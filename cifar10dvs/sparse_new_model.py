import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import layer, surrogate
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

__all__ = ['spikformer','spikformerteacher']

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,C,N = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,N).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,N).contiguous()
        x = self.fc2_lif(x)
        return x

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = x.transpose(-1,-2) # T*B ,C,N
        x = self.in_conv(x)
        B, N, C = x.size()  # T*B ,N, C
        local_x = x[:,:, :C//2]  # T*B, N, C//2
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)  # B, 1, C//2
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)  # B, N, C
        return self.out_conv(x) # B, N, 2
    
class Predictor(nn.Module):

    def __init__(self, embed_dim=384):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim//2, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        return self.linear(x) # B, N, 2


class SSA(nn.Module):   # Spiking Self Attention
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."   # hidden dim 应该被num_head 整除
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        
        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def attention_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()  # T*B , N , 1
        B, H, N, N = attn.size() # T*B , H , N , N
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32) * attn_policy.to(torch.float32)
        return attn.type_as(max_att)

    def forward(self, x, policy=None):
        T, B, C, N = x.shape     # spike feature

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # SSA' (Q,K,V) = SNN(Q*k.T*V * s)
        # SSA (Q,K,V) = SNN(BN(linear(SSA'(Q.K,V))))
        attn = (q @ k.transpose(-2, -1))   # Q*K*scale  # (T,B,Head,N,N)

        if policy is not None:
            t,b,h,n,n = attn.shape
            attn = attn.flatten(0,1)
            attn = self.attention_policy(attn, policy)
            attn = attn.reshape(t,b,h,n,n)

        x = ( attn @ v ) * self.scale                                 # Q*k*scale*V
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)                           # SNN(Q*K*scale)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,N)) # SNN(BN(Linear(Q*k*scale*V)))
        
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, policy = None):
        x_attn = self.attn(x , policy) # SSA
        x = x + x_attn        # x = x + SSA
        x = x + self.mlp(x)   # x = x + MLP

        return x


class SPS(nn.Module):  # spiking patch spliting   脉冲patch分割
    # 给定一个二维图像序列 I∈R^(T x C x H x W)，脉冲patch分割（SPS）模块将其线性投影到一个D维的脉冲形式特征向量上，并将其分割成N个脉冲形式的patches序列X∈R^(T x N x D)
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w] # [128, 128]
        patch_size = to_2tuple(patch_size)         # (4,4)
        self.patch_size = patch_size               # 4
        self.C = in_channels                       # 2
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]  # 128/4 = 32  
        self.num_patches = self.H * self.W         # 32 * 32 = 1024
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)  # conv(in,out,kernel_size,stride,padding) (4,32,3,1,1)  -> (out,H,W) (32,128,128)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)  # 32
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False) # conv(in,out,kernel_size,stride,padding) (32,64,3,1,1) -> (out,H,W) (64,128,128)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4) # 64
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False) # conv(in,out,kernel_size,stride,padding) (64,128,3,1,1) -> (out,H,W) (128,128,128)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2) # 128
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)  # MaxPool(kernel_size,stride,padding) (3,2,1) -> (out,H,W) (128,64,64)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False) # conv(in,out,kernel_size,stride,padding) (128,256,3,1,1) -> (out,H,W) (256,64,64)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims) # 256
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) #  MaxPool(kernel_size,stride,padding) (3,2,1) -> (out,H,W) (256,32,32)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)  # conv(in,out,kernel_size,stride,padding) (256,256,3,1,1) -> (out,H,W) (256,32,32)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape                                   #  x = float(?)
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value   Conv2d
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()  #  BN
        x = self.proj_lif(x).flatten(0, 1).contiguous()           #  Lif   -> x = spike(?)
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, 64, 64).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, 32, 32).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, 16, 16).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_rpe = self.rpe_bn(self.rpe_conv(x)).reshape(T,B,-1,H//16,W//16).contiguous()
        x_rpe = self.rpe_lif(x_rpe).flatten(0,1)
        x = x + x_rpe
        x = x.reshape(T, B, -1, (H//16)*(H//16)).contiguous()
        return x

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

class SpikformerPrune(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4 , pruning_loc = None , token_ratio = None,  distill = False
                 ):
        super().__init__()
        self.num_classes = num_classes # 11
        self.depths = depths       # [6, 8, 6]
        # predictor_list = [PredictorLG(embed_dims) for _ in range(len(pruning_loc))]
        # self.score_predictor = nn.ModuleList(predictor_list)
        predictor_list = [Predictor(embed_dims) for _ in range(len(pruning_loc))]
        self.predictor = nn.ModuleList(predictor_list)

        self.distill = distill
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule  
        # e.g depths = 8  drop_path_rate = 1.
        # dpr = [0.0, 0.1428571492433548, 0.2857142984867096, 0.4285714626312256, 0.5714285373687744, 0.7142857313156128, 0.8571428656578064, 1.0]

        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"pos_embed", pos_embed)
        setattr(self, f"block", block)

        # classification head
        self.norm = norm_layer(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        
        pos_embed = getattr(self, f"pos_embed")
        trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x)

        T,B,C,N = x.shape
        p_count = 0
        out_pred_prob = []
        init_n = N
        prev_decision = torch.ones(T*B, init_n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(T*B, init_n, 1, dtype=x.dtype, device=x.device)

        decisions = [[] for _ in self.pruning_loc]


        #for blk in block:
        for i, blk in enumerate(block):
            if i in self.pruning_loc: 
                # spatial_x = x.flatten(0,1)  # TB,C,N
                spatial_x = x.mean(0)  # B,C,N
                spatial_x = spatial_x.transpose(-2,-1)
                # spatial_x = x[:, 1:]
                pred_score = self.predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)  # B,N,2
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1]
                    hard_keep_decision = (hard_keep_decision.unsqueeze(0).repeat(T,1,1,1)).reshape(T*B,-1,1)
                    hard_keep_decision = hard_keep_decision * prev_decision
                    out_pred_prob.append(hard_keep_decision.reshape(T*B, init_n))
                    policy =  hard_keep_decision
                    x = blk(x, policy=policy)
                    prev_decision = hard_keep_decision
                else:
                    score = pred_score[:,:,0]
                    num_keep_node = int(init_n * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                    keep_policy = (keep_policy.unsqueeze(0).repeat(T,1,1)).reshape(T*B,-1)
                    # now_policy = keep_policy + 1
                    x = x.flatten(0,1)  # TB,C,N
                    x = x.transpose(-1,-2)  # TB,N,C
                    decisions[p_count].append(keep_policy)
                    x = batch_index_select(x, keep_policy)
                    x = x.reshape(T,B,-1,C) # T,B,New,C
                    x = x.transpose(-1,-2) # T,B,C,New
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    x = blk(x)
                p_count += 1
            else:
                if self.training:
                    x = blk(x, policy)
                else:
                    x = blk(x)

        x = x.mean(3)  # t b c  -> t b c
        x = self.head(x.mean(0)) # t b c -> b c

        if self.training:
            if self.distill:
                return x, prev_decision.detach(), out_pred_prob
            else:
                return x, out_pred_prob
        else:
            return x, decisions


class SpikformerTeacher(nn.Module):
    '''SpikeFormer'''
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2]
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"pos_embed", pos_embed)
        setattr(self, f"block", block)

        self.norm = norm_layer(embed_dims)
        # classification head 这里不需要脉冲，因为输入的是在T时长平均发射值
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        pos_embed = getattr(self, f"pos_embed")
        trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.mean(3)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

@register_model
def spikformer(pretrained=False, **kwargs):
    model = SpikformerPrune(
        patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
        in_channels=2, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        pruning_loc = [1,2,3],
        token_ratio = [0.8 , 0.8**2, 0.8**3],
        distill = True,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def spikformerteacher(pretrained=False, **kwargs):
    model = SpikformerTeacher(
        patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
        in_channels=2, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


