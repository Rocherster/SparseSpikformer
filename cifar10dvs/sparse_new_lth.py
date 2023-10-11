import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from torch.cuda import amp
import sparse_new_model, utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy
from loss import DistillDiffPruningLoss_dynamic
import autoaugment
from utils_for_lth import *
from utl import data_transforms
import copy
import pickle
import time
import utl
import csv
import os


_seed_ = 2021
import random
random.seed(2021)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(_seed_)
writer = SummaryWriter("./")

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='Sparse_Spikformer_new_lth', help='model')
    parser.add_argument('--dataset', default='cifar10dvs', help='dataset')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--data-path', default='./data', help='dataset')
    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--log-wandb', action='store_true', default=True,
                    help='log training and validation metrics to wandb')
    parser.add_argument('--experiment', default='Sparse_spikformer_cifar10dvs', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        # default=True,
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True,  action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    # parser.add_argument('--adam', default=True, action='store_true',
    #                     help='Use Adam')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    #Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--ratio_weight',type=float,default=2.0)
    parser.add_argument('--base_weight',type=float,default=0.9)

    # for lth
    parser.add_argument("--prune_percent", default=25, type=float, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=15, type=int, help="Pruning iterations count")
    parser.add_argument('--round', type=int, default=1, help='for mean and std === 1 to 5')
    parser.add_argument("--pruning_scope", default="global", type=str)
    parser.add_argument("--end_iter", default=300, type=int)
    parser.add_argument('--valid_freq', type=int, default=50, help='test for SNN')
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--rewinding_epoch", default=20, type=int)
    parser.add_argument("--sparsity_round", default=0, type=int)
    parser.add_argument("--iter", default=0, type=int)

    args = parser.parse_args()
    return args

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None, aug=None, trival_aug=None, mixup_fn=None):

    torch.cuda.set_device(1)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    EPS = 1e-6

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]
        N,T,C,H,W = image.shape
        if aug != None:
            image = torch.stack([(aug(image[i])) for i in range(N)])
        if trival_aug != None:
            image = torch.stack([(trival_aug(image[i])) for i in range(N)])

        if mixup_fn is not None:
            image, target = mixup_fn(image, target)
            target_for_compu_acc = target.argmax(dim=-1)


        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss,loss_part = criterion(image, output, target)
        else:
            output = model(image)
            loss,loss_part = criterion(image, output, target)

        pred, mask, out_pred_score = output

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()

            if not iter == 0:
                # Freezing Pruned weights by making their gradients Zero
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        tensor = p.data
                        if (len(tensor.size())) == 1:
                            continue
                        grad_tensor = p.grad
                        grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                        p.grad.data = grad_tensor

            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()

            if not iter == 0:
                # Freezing Pruned weights by making their gradients Zero
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        tensor = p.data
                        if (len(tensor.size())) == 1:
                            continue
                        grad_tensor = p.grad
                        grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                        p.grad.data = grad_tensor

            optimizer.step()

        functional.reset_net(model)
        if mixup_fn is not None:
            acc1, acc5 = utils.accuracy(pred, target_for_compu_acc, topk=(1, 5))
        else:
            acc1, acc5 = utils.accuracy(pred, target, topk=(1, 5))
        batch_size = image.shape[0]
        #loss_s = loss.item()
        loss_s = loss
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):

    torch.cuda.set_device(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            output = model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=T, split_by='number')
    dataset_train, dataset_test = split_to_train_test_set(0.9, origin_set, 10)
    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()

    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def main(args):

    # if args.log_wandb:   # wandb记录
    #     if has_wandb:
    #         wandb.init(project=args.experiment, config=args)
    #     else:
    #         _logger.warning("You've requested to log metrics to wandb but package not found. "
    #                         "Metrics not being logged to wandb, try `pip install wandb`")
    torch.cuda.set_device(1)
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_T{args.T}')

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    if args.opt == 'adamw':
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    device = torch.device(args.device)
    data_path = args.data_path
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(data_path, args.distributed, args.T)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True)

    model = create_model(
        'spikformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None
    )
    print("Creating student model")

    teacher_model = create_model(
        'spikformerteacher',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None
    )
    print("Creating teacher model")

    pretrained = torch.load('/home/ly22216501/SNN/transformer/SparseSpikformer/cifar10dvs/pretrained/checkpoint_max_test_acc1.pth', map_location='cpu')

    load_state_dict(model, pretrained)
    load_state_dict(teacher_model, pretrained)

    print("Loading pretrained model")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    model.to(device)
    teacher_model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # criterion_train = LabelSmoothingCrossEntropy()
    criterion_train = SoftTargetCrossEntropy().cuda()
    criterion = nn.CrossEntropyLoss()

    criterion_train = DistillDiffPruningLoss_dynamic(
            teacher_model, criterion_train, clf_weight=1.0, keep_ratio=[0.7,0.7**2,0.7**3], mse_token=True, ratio_weight=args.ratio_weight, distill_weight=0.5
        )

    optimizer = create_optimizer(args, model)
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:

        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')


    train_snn_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5)
                    ])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)

    # Copying and Saving Initial State
    # initial_state_dict = copy.deepcopy(model.state_dict())  # 初始权重参数
    # utl.checkdir(f"{os.getcwd()}/sparse_new_snn_laterewind_lth/")
    # torch.save(model.state_dict(), f"{os.getcwd()}/sparse_new_snn_laterewind_lth/initial_state_dict.pth.tar")

    initial_state_dict = torch.load("/home/ly22216501/SNN/transformer/SparseSpikformer/cifar10dvs/sparse_new_snn_laterewind_lth/laterewinding_state_dict.pth.tar",map_location='cuda:1')

    checkpoint = torch.load("/home/ly22216501/SNN/transformer/SparseSpikformer/cifar10dvs/logs/Sparse_new_lth_b16_T16_wd0.06_adamw_cnf_ADD/lr0.001_Iteration1/checkpoint_299.pth",map_location='cuda:1')
    model.load_state_dict(checkpoint['model'])

    # Making Initial Mask
    with open(f"{os.getcwd()}/dumps/sparse_new_snn_laterewind_lth/mask_75.2.pkl", 'rb') as fp:  # 加载mask 保证和 lth使用相同的mask
            mask = pickle.load(fp)

    # mask = make_mask(model)

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    max_acc = np.zeros(ITERATION, float)
    dir = output_dir
    rewinding_epoch = 20


    print("Start training")
    start_time = time.time()

    for _ite in range(2, ITERATION):

        max_test_acc1 = 0.
        test_acc5_at_max_test_acc1 = 0.

        train_tb_writer = None
        te_tb_writer = None

        args.iter = _ite

        output_dir = dir + f'_Iteration{_ite}'
        if not os.path.exists(output_dir):
            utils.mkdir(output_dir)

        if args.tb and utils.is_main_process():
            purge_step_train = args.start_epoch
            purge_step_te = args.start_epoch
            train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
            te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
            with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
                args_txt.write(str(args))

            print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

        if not _ite == 0:  # 1,2,3 ... 14
            model, mask = prune_by_percentile(args, args.prune_percent, mask, model)  # prune_percent=25
            model = original_initialization(mask, initial_state_dict, model)

            optimizer = create_optimizer(args, model)
            if args.amp:
                scaler = amp.GradScaler()
            else:
                scaler = None
            lr_scheduler, num_epochs = create_scheduler(args, optimizer)

        print(f"\n--- Pruning Level [round{args.round}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utl.print_nonzeros(model)
        comp[_ite] = comp1

        # Dumping mask
        utl.checkdir(f"{os.getcwd()}/dumps/sparse_new_snn_laterewind_lth/")
        with open(f"{os.getcwd()}/dumps/sparse_new_snn_laterewind_lth/mask_{comp1}.pkl",
                'wb') as fp:
            pickle.dump(mask, fp)

        for epoch in range(args.end_iter):

            comp1 = utl.print_nonzeros(model)
            save_max = False
            if args.distributed:
                train_sampler.set_epoch(epoch)
            if epoch >= 75:
                mixup_fn.mixup_enabled = False
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, criterion_train, optimizer, data_loader, device, epoch,
                args.print_freq, scaler, args.T_train,
                train_snn_aug, train_trivalaug, mixup_fn)
            if utils.is_main_process():
                train_tb_writer.add_scalar('train_loss', train_loss, epoch)
                train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
            lr_scheduler.step(epoch + 1)

            test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
            if te_tb_writer is not None:
                if utils.is_main_process():

                    te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                    te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                    te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


            if max_test_acc1 < test_acc1:
                max_test_acc1 = test_acc1
                test_acc5_at_max_test_acc1 = test_acc5
                save_max = True


            if output_dir:

                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_test_acc1': max_test_acc1,
                    'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
                }

                if save_max:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
                
                output_file = os.path.join(output_dir, 'summary.csv')
               
                # 检查输出文件是否存在，如果不存在则创建一个新文件
                if not os.path.exists(output_file):
                    with open(output_file, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['epoch', 'test_acc1', 'test_acc5', 'max_test_acc1'])

                # 将数据写入 CSV 文件
                with open(output_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, test_acc1, test_acc5, max_test_acc1])

            # print(args)

            # TODO Late rewinding init weight at 20epoch
            if _ite == 0 and epoch == args.rewinding_epoch:  # rewinding_epoch=20
                print('--------find laterewinding weight--------')
                initial_state_dict = copy.deepcopy(model.state_dict())
                rewinding_epoch = args.rewinding_epoch
                utl.checkdir(f"{os.getcwd()}/sparse_new_snn_laterewind_lth/")
                torch.save(initial_state_dict, f"{os.getcwd()}/sparse_new_snn_laterewind_lth/laterewinding_state_dict.pth.tar")

            print('Iteration: ', _ite, 'Epoch: ', epoch, 'Prune rate: ', comp1, 'test_acc1: ', test_acc1,
                    'max_test_acc1: ', max_test_acc1)

            if output_dir:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

        max_acc[_ite] = max_test_acc1
        np.savetxt("sparse_lth_best_acc.txt", max_acc)

    return max_test_acc1

if __name__ == "__main__":
    args = parse_args()
    main(args)
