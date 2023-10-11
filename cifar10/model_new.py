import torch
import copy
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import numpy as np
cupyname='torch'
# from wfunc import *
def dec2bin(seednum,BL):
    return bin(seednum)[2:].zfill(BL)
def hwbin(a,BL):
    if a <0:
        a = 2**BL + a
    fbin = dec2bin(a,BL)
    x=''
    for i in range(BL):
        x += fbin[BL-i-1]
    return x
def sc_mul5(a,b):
    BL= 5
    u = [int(i) for i in hwbin(a,BL)]
    v = [int(i) for i in hwbin(b,BL)]
    b4 = int(b/2)
    b3 = int(b4/2)
    b2 = int(b3/2)
    d = (u[0] * v[4] ) + (u[1] * v[4]) + (u[1] * v[3]) + u[2] * b2 + u[3]*b3 + u[4]*b4
    return d
def sc_mul6(a,b):
    BL= 6
    u = [int(i) for i in hwbin(a,BL)]
    v = [int(i) for i in hwbin(b,BL)]
    b5 = int(b/2)
    b4 = int(b5/2)
    b3 = int(b4/2)
    b2 = int(b3/2)
    d = (u[0] * v[5]) + (u[1] * v[5]) + (u[1] * v[4]) + u[2] * b2 + u[3]*b3 + u[4]*b4 + + u[5]*b5
    return d  
def sc_mul4(a,b):
    BL= 4
    u = [int(i) for i in hwbin(a,BL)]
    v = [int(i) for i in hwbin(b,BL)]
    b3 = int(b/2)
    b2 = int(b3/2)
    d = (u[0] * v[3] ) + (u[1] * v[3]) + (u[1] * v[2]) + u[2] * b2 + u[3]*b3  
    return d  
def scmul(aa,bb,BL):
    N = 2**BL
    a = abs(aa)
    b = abs(bb)
    if a >= N-1:
        f = b
    elif a == 0:
        f= 0
    else:
        if BL==4:
            f = sc_mul4(a,b)
        elif BL==5:
            f = sc_mul5(a,b)
        else:
            f = sc_mul6(a,b)
    d = f*N 
    if (aa<0 and bb>0) or (aa > 0 and bb <0):
        return -d
    else:
        return d
AA = {}
for BL in range(4,7):
    N = 2**BL+1
    B = torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            B[i,j] = i*j
    if BL not in AA:
        AA[BL]=B.int()
# BL = 6
# N = 2**BL
# A =  AA[BL] 
AB = {}
for BL in range(4,7):
    N = 2**BL+1
    A5 = torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            A5[i,j]=scmul(i,j,BL)
    if BL not in AB:
        AB[BL]=A5
torch.save(AB,'scmul_new.pt')

BL = 5
N = 2**BL
A =  AB[BL] 
A = A.cuda()

def SC_MAT_MAC_p(a,b,c,N):

     

    # a = xxx.reshape(-1,b.size(1)) # 
    # d = torch.matmul(a,b)+c
    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    b = torch.where(torch.isnan(b), torch.full_like(b, 0), b)
    c = torch.where(torch.isnan(c), torch.full_like(c, 0), c)
    a = torch.where(torch.isinf(a), torch.full_like(a, 1), a)
    b = torch.where(torch.isinf(b), torch.full_like(b, 1), b)
    c = torch.where(torch.isinf(c), torch.full_like(c, 1), c)

    kkk=torch.tensor([a.data.abs().max(),b.abs().max(),c.abs().max()])
    # print(f"max {kkk}  {kkk[0]} {kkk[1:]}")
    # if torch.isnan(kkk).sum()>0:
    #     print(f"a {a},b{b},c {c}")
    # print(f"a {a.size()},b{b.size()},c {c.size()}")
    dmax = kkk[1:].max()
    if kkk[0]==0:
        kkk[0]=1 
    SBL2 = int(np.log2(int(N/kkk[0])))
    if dmax==0:
        dmax = 1
    SBL1 = int(np.log2(int(N/dmax)))
    SN1 = 2**SBL1
    SN2=  2**SBL2
    aa = (a*SN2).int()
    bb = (b*SN1).int()
    cc = (c*SN1).int()
    x3 = aa.repeat_interleave(b.size(0), dim=0)
    x4 = bb.repeat(a.size(0), 1)
    # print(f"aa {aa} bb {bb} cc {cc}")
    # print(f"x3 {x3.size()} aa {aa} x3 {x3}")
    x5 = x3.view(-1).int()
    x6 = x4.view(-1).int()    
    # print(f"x5 {x5} x6 {x6}")
    x5 = x5.long()
    x6 = x6.long()
    x7 = A[[x5.abs(),x6.abs()]]
    x5 = x5.cuda()
    x6 = x6.cuda()
    x7 = x7.cuda()
    x7 = (((x5>0)*(x6<0) +  (x5<0)*(x6>0))*-1 + ((x5>0)*(x6>0) + (x5<0)*(x6<0)))*x7
    x8 = x7.view(-1,a.size(1))
    x9 = x8.sum(1).view(a.size(0),-1) 
    # print(f"aa {aa} bb {bb} cc {cc} x7 {x7}")
    # print(f"x8 {x8} x9 {x9}  ")
    d = (x9/SN2).int()+ cc
    p = d/SN1
    # p = p.reshape(xxx.shape)
    return p  
def SC_RBN_p(x,x_mean,u,w,b,N):

    #print(f"deubg {b.size()}")
    x = x.reshape(-1,b.size(0))
    B = torch.tensor(x.size(0))
    cb = 1/(torch.sqrt(2*torch.log(B))) 
    if x_mean == None:
        B = torch.tensor(x.size(0))
        cb = 1/(torch.sqrt(2*torch.log(B))) 
        x_mean = torch.mean(x,dim=0,keepdim=True)
    if u == None:
        x_max = x.max(0)[0]
        x_min = x.min(0)[0]
        qu = (x - x_mean)
        d = x_max - x_min
        u = (cb * d) 


    q = (x - x_mean)
    # print(f"cb {cb} d {u} \n {q/u}")
    q = torch.where(torch.isnan(q), torch.full_like(q, 0), q)
    u = torch.where(torch.isnan(u), torch.full_like(u, 0), u)
    q = torch.where(torch.isinf(q), torch.full_like(q, 1), q)
    u = torch.where(torch.isinf(u), torch.full_like(u, 1), u)
    dmax=torch.tensor([q.abs().max(),u.abs().max()]).max()
    if dmax == 0:
        dmax = 1
    BL1 = int(np.log2(int(N/dmax)))
    SN1 = 2**BL1
    qq = (q*SN1).int()
    uu = (u*SN1).int()
    wmax = w.abs().max()
    bmax = b.abs().max()
    if wmax == 0 :
        wmax = 1
    if bmax == 0:
        bmax = 1
    BL2 = int(np.log2(int(N/wmax)))
    SN2 = 2**BL2
    BL3 = int(np.log2(int(N/bmax)))
    SN3 = 2**BL3

    ww = (w*SN2).int()
    bb = (b*SN3).int()
    # w*(q/u) + b with ESC div 
    x5 = ww.repeat_interleave(q.size(0), dim=0).view(-1)
    x6 = qq.view(-1)

    x5 = x5.long()
    x6 = x6.long()
    uu = uu.long()
    bb = bb.long()

    x5 = x5.cuda()
    x6 = x6.cuda()
    uu = uu.cuda()
    bb = bb.cuda()

    u = u.cuda()
    b = b.cuda()
    # print(f"x5 {x5.size()} x6 {x6.size()} x {x.size()}")
    x7 = A[x5.abs(),x6.abs()]
    x7 = x7.cuda()
    x7 = (((x5>0)*(x6<0) +  (x5<0)*(x6>0))*-1 + ((x5>0)*(x6>0) + (x5<0)*(x6<0)))*x7
    x7 = x7.view(x.size(0),-1)
    x8 = A[uu.abs(),bb.abs()]
    x8 = x8.cuda()
    x8 = (((u>0)*(b<0) +  (u<0)*(b>0))*-1 + ((u>0)*(b>0) + (u<0)*(b<0)))*x8
    rr = x7*SN3 + x8*SN2
    ss = uu*SN2*SN3
    p = rr/ss
    return p
def f_batch_norm(is_training,X,gamma,beta,moving_mean,moving_var,eps,momentum):
    if not is_training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X-mean)**2).mean(dim=0)
        else:
            mean = X.mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
            var = ((X-mean)**2).mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
        X_hat = (X-mean) / torch.sqrt(var+eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) *var
    Y = gamma * X_hat + beta
    return Y,moving_mean,moving_var
def f_rbn(is_training,X,gamma,beta,moving_mean,moving_var,eps,momentum):
    # (x-u)/(cb*(x.max()-x.min()))
    B = torch.tensor(X.size(0))
    cb = 1/(torch.sqrt(2*torch.log(B))) 
    x_mean = torch.mean(X,dim=0,keepdim=True)
    x_max = X.max(0)[0]
    x_min = X.min(0)[0]
    qu = (X - x_mean)
    d = x_max - x_min
    du = (cb * d) 
    # print(f"x_mean {x_mean},x_max {x_max} x_min {x_min} \nd {d} du {du}")
    if not is_training:
        X_hat = (X - moving_mean) / moving_var
    else:
        X_hat = (X-x_mean) / du
        moving_mean = momentum * moving_mean + (1.0 - momentum) * x_mean
        moving_var = momentum * moving_var + (1.0 - momentum) * du
    Y = gamma * X_hat + beta
    return Y,moving_mean,moving_var

class SCLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True,BL=6):
        super(SCLinear, self).__init__()
        self.N = 2**BL
        self.linear = nn.Linear(input_features, output_features, bias=bias)  # 输入和输出的维度都是
    def forward(self, x):
        out = self.linear(x)
        out.data = SC_MAT_MAC_p(x,self.linear.weight.data,self.linear.bias.data,self.N)
        return out

class MyBatchNorm1d(nn.Module):
    def __init__(self,num_features):
        super(MyBatchNorm1d,self).__init__()
        shape = (1,num_features)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    def forward(self,X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        X = X.reshape(-1,self.moving_var.size(1))
        Y,self.moving_mean,self.moving_var = f_batch_norm(self.training,X,self.gamma,self.beta,self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)
        return Y
class MyRBN(nn.Module):
    def __init__(self,num_features):
        super(MyRBN,self).__init__()
        shape = (1,num_features)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    def forward(self,X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        #X = X.reshape(-1,self.moving_var.size(1))
        Y,self.moving_mean,self.moving_var = f_rbn(self.training,X,self.gamma,self.beta,self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)
        return Y
class SCRBN1(nn.Module):
    def __init__(self,num_features,BL=5):
        super(SCRBN,self).__init__()
        self.dim = num_features
        self.bn = nn.BatchNorm1d(num_features)#MyRBN(num_features)
        self.N = 2**BL
    def forward(self,X):
        N = self.N
        print("x.shape",X.shape)
        Y = self.bn(X)
         
        if not self.training:
            Y.data =  (SC_RBN_p(X,self.bn.running_mean.data,self.bn.running_var.data,self.bn.weight.data,self.bn.bias.data,N)).reshape(X.shape)
        else:
            Y.data =  (SC_RBN_p(X,None,None,self.bn.weight.data,self.bn.bias.data,N)).reshape(X.shape)
        return Y
class SCRBN(nn.Module):
    def __init__(self,num_features,BL=5):
        super(SCRBN,self).__init__()
        self.bn = MyRBN(num_features)
        self.N = 2**BL
    def forward(self,X):
        N = self.N
        Y = self.bn(X)
        X = X.reshape(-1,self.bn.gamma.size(1))
        B = torch.tensor(X.size(0))
        cb = 1/(torch.sqrt(2*torch.log(B))) 
        x_mean = torch.mean(X,dim=0,keepdim=True)
        x_max = X.max(0)[0]
        x_min = X.min(0)[0]
        d = x_max - x_min
        du = (cb * d)
        if not self.training:
            Y.data =  SC_RBN_p(X,self.bn.moving_mean,self.bn.moving_var,self.bn.gamma.data,self.bn.beta.data,N)
        else:
            Y.data =  SC_RBN_p(X,x_mean,du,self.bn.gamma.data,self.bn.beta.data,N)

        return Y
class SSA_SL(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,BL=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = SCLinear(dim, dim,BL=BL)
        self.q_bn = MyBatchNorm1d(dim) 
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.k_linear = SCLinear(dim, dim,BL=BL)
        self.k_bn = MyBatchNorm1d(dim) 
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.v_linear = SCLinear(dim, dim,BL=BL)
        self.v_bn = MyBatchNorm1d(dim) 
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=cupyname)
        self.proj_linear = SCLinear(dim, dim,BL=BL)
        self.proj_bn = MyBatchNorm1d(dim) 
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
    def forward(self, x):
        T,B,N,C = x.shape
        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        proj_linear_out = self.proj_linear(x)
        proj_linear_out = self.proj_bn(proj_linear_out).reshape(T, B, N, C)
        x = self.proj_lif(proj_linear_out)
        return x
class SSA_R(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,BL=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = MyRBN(dim) 
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = MyRBN(dim) 
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = MyRBN(dim) 
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=cupyname)
        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = MyRBN(dim) 
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
    def forward(self, x):
        T,B,N,C = x.shape
        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        proj_linear_out = self.proj_linear(x)
        proj_linear_out = self.proj_bn(proj_linear_out).reshape(T, B, N, C)
        x = self.proj_lif(proj_linear_out)
        return x
    
class SSA_SLR(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,BL=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = SCLinear(dim, dim,BL=BL)
        self.q_bn = MyRBN(dim) 
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.k_linear = SCLinear(dim, dim,BL=BL)
        self.k_bn = MyRBN(dim) 
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.v_linear = SCLinear(dim, dim,BL=BL)
        self.v_bn = MyRBN(dim) 
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=cupyname)
        self.proj_linear = SCLinear(dim, dim,BL=BL)
        self.proj_bn = MyRBN(dim) 
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
    def forward(self, x):
        T,B,N,C = x.shape
        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        proj_linear_out = self.proj_linear(x)
        proj_linear_out = self.proj_bn(proj_linear_out).reshape(T, B, N, C)
        x = self.proj_lif(proj_linear_out)
        return x
class SSA_SR(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,BL=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = SCRBN(dim,BL=BL) 
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = SCRBN(dim,BL=BL) 
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = SCRBN(dim,BL=BL) 
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=cupyname)
        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = SCRBN(dim,BL=BL) 
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
    def forward(self, x):
        T,B,N,C = x.shape
        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        proj_linear_out = self.proj_linear(x)
        proj_linear_out = self.proj_bn(proj_linear_out).reshape(T, B, N, C)
        x = self.proj_lif(proj_linear_out)
        return x
    
class SSA_SC(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,BL=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = SCLinear(dim, dim,BL=BL)
        self.q_bn = SCRBN(dim,BL=BL) 
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.k_linear = SCLinear(dim, dim,BL=BL)
        self.k_bn = SCRBN(dim,BL=BL) 
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.v_linear = SCLinear(dim, dim,BL=BL)
        self.v_bn = SCRBN(dim,BL=BL) 
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=cupyname)
        self.proj_linear = SCLinear(dim, dim,BL=BL)
        self.proj_bn = SCRBN(dim,BL=BL) 
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=cupyname)
    def forward(self, x):
        T,B,N,C = x.shape
        x_for_qkv = x.reshape(-1, C)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.reshape(-1, C)
        proj_linear_out = self.proj_linear(x)
        proj_linear_out = self.proj_bn(proj_linear_out).reshape(T, B, N, C)
        x = self.proj_lif(proj_linear_out)
        return x

BL = 5
A = AB[BL]

# from visualizer import get_local
__all__ = ['spikformer']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        #self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        #self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape
        x_ = x.flatten(0, 1)  # T*B,N,C
        x = self.fc1_linear(x_)     # 1.linear  T*B,N,C -> T*B,N,H
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous() # 2.bn  T*B,N,H ->
        x = self.fc1_lif(x)         # 3.lif

        x = self.fc2_linear(x.flatten(0,1)) # 4.linear
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous() # 5.bn
        x = self.fc2_lif(x)                 # 6.lif
        return x



class SSA(nn.Module):   # Spiking Self Attention
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."   # hidden dim 应该被num_head 整除
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        #self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        #self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        #self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        #self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        #self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    # @get_local('attn')
    def forward(self, x):
        T,B,N,C = x.shape     # spike feature

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]   1. X*Wq
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()  # 2. Bn
        q_linear_out = self.q_lif(q_linear_out)   #             3. Lif 
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()  # (T, B, num_head, N, C//num_head)  Q = SNN(BN(X*Wq))

        k_linear_out = self.k_linear(x_for_qkv)  # [TB, N, C]
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B,C,N).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()  # (T, B, num_head, N, C//num_head)  K = SNN(BN(X*Wk))

        v_linear_out = self.v_linear(x_for_qkv)  # [TB, N, C]
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B,C,N).contiguous() 
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()  # (T, B, num_head, N, C//num_head)  V = SNN(BN(X*Wv))

        # SSA' (Q,K,V) = SNN(Q*k.T*V * s)
        # SSA (Q,K,V) = SNN(BN(linear(SSA'(Q.K,V))))
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Q*K*scale
        x = attn @ v                                   # Q*k*scale*V
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)                           # SNN(Q*K*scale)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))  # SNN(BN(Linear(Q*k*scale*V)))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA_SC(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,BL=BL)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x_attn = self.attn(x) # SSA
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
        #self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False) # conv(in,out,kernel_size,stride,padding) (32,64,3,1,1) -> (out,H,W) (64,128,128)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4) # 64
        #self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False) # conv(in,out,kernel_size,stride,padding) (64,128,3,1,1) -> (out,H,W) (128,128,128)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2) # 128
        #self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)  # MaxPool(kernel_size,stride,padding) (3,2,1) -> (out,H,W) (128,64,64)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False) # conv(in,out,kernel_size,stride,padding) (128,256,3,1,1) -> (out,H,W) (256,64,64)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims) # 256
        #self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) #  MaxPool(kernel_size,stride,padding) (3,2,1) -> (out,H,W) (256,32,32)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)  # conv(in,out,kernel_size,stride,padding) (256,256,3,1,1) -> (out,H,W) (256,32,32)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        #self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, C, H, W = x.shape                                   #  x = float(?)
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value   Conv2d
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()  #  BN
        x = self.proj_lif(x).flatten(0, 1).contiguous()           #  Lif   -> x = spike(?)

        x = self.proj_conv1(x)                                    #  Conv2d 
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        return x


class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4
                 ):
        super().__init__()
        self.T = T  # time step  4
        self.num_classes = num_classes # 11
        self.depths = depths       # [6, 8, 6]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule  
        # e.g depths = 8  drop_path_rate = 1.
        # dpr = [0.0, 0.1428571492433548, 0.2857142984867096, 0.4285714626312256, 0.5714285373687744, 0.7142857313156128, 0.8571428656578064, 1.0]

        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)

        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
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
        # path = "prune sparsity.txt"
        # file = open(path,"a+")
        # file.write(str(x.size()))

        # # 统计tensor中0的个数
        # num_zeros = torch.sum(x == 0)
        # file.write(f"The SPS contains {num_zeros} zeros." + "\n")

        # num_elements = x.numel()
        # sparsity = num_zeros / num_elements
        # file.write(f"The sparsity of SPS is {sparsity}." + "\n")

        count = 0
        for blk in block:
            x = blk(x)
            # count += 1
            # num_zeros = torch.sum(x == 0)
            # file.write(f"Block {count} contains {num_zeros} zeros." + "\n")

            # num_elements = x.numel()
            # sparsity = num_zeros / num_elements
            # file.write(f"The sparsity of Block {count} is {sparsity}." + "\n")

        return x.mean(2)  # t b n c -> t b c

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0)) # t b c -> b c
        return x


@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(
        # img_size_h=224, img_size_w=224,
        # patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        # in_channels=3, num_classes=1000, qkv_bias=False,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


