import argparse
import yaml
from easydict import EasyDict
import os
import glob

from utils.sse import sse_input_path_validated, sse_output_path_validated
from utils.yaml_rw import load_yaml, save_yaml
from train import train
from test import test
from fit import fit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./input', help='input path')
    parser.add_argument('--output_path', type=str, default='./output', help='output path')
    
    parser.add_argument('--process', type=str, default='train', choices=['train', 'test', 'fit'], help='process name')
    
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False, help='resume from checkpoint for train or fit')
    
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='device')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers (per RANK if DDP)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'radam', 'lbfgs', 'sgd_nesterov', 'asgd', 'rprop'], help='optimizer type')
    parser.add_argument('--scheduler', type=str, default='steplr', choices=['steplr', 'multisteplr', 'exponential', 'cosine', 'cyclic', 'onecycle', 'lambda', 'cosine_warm'], help='learning rate scheduler type')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=['cross_entropy', 'mse', 'l1', 'binary_cross_entropy'], help='loss_function')
    
    
    # 通用优化器参数
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor')
    
    # Adam系列参数
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='betas for Adam/AdamW/Adamax/NAdam')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for numerical stability')
    parser.add_argument('--amsgrad', action='store_true', help='use AMSGrad variant for Adam')
    
    # SGD参数
    parser.add_argument('--dampening', type=float, default=0, help='dampening for momentum')
    parser.add_argument('--nesterov', action='store_true', help='enables Nesterov momentum')
    
    # RMSprop参数
    parser.add_argument('--alpha', type=float, default=0.99, help='smoothing constant for RMSprop')
    parser.add_argument('--centered', action='store_true', help='compute centered RMSProp')
    
    # Adadelta参数
    parser.add_argument('--rho', type=float, default=0.9, help='coefficient for computing running averages')
    
    # LBFGS参数
    parser.add_argument('--max_iter', type=int, default=20, help='max iterations per optimization step for LBFGS')
    parser.add_argument('--history_size', type=int, default=100, help='update history size for LBFGS')
    
    # ASGD参数
    parser.add_argument('--lambd', type=float, default=1e-4, help='decay term for ASGD')
    parser.add_argument('--alpha_asgd', type=float, default=0.75, help='power for eta update in ASGD')
    parser.add_argument('--t0', type=float, default=1e6, help='point at which to start averaging for ASGD')
    
    
    # 调度器
    # 通用调度器参数
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate decay step')
    parser.add_argument('--lr_decay_min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum epochs')
    
    # MultiStepLR 特定参数
    parser.add_argument('--milestones',type=list, default=[30, 60, 90], help='milestones for MultiStepLR')
    
    # CosineAnnealingLR 特定参数
    parser.add_argument('--T_max', type=int, default=50, help='T_max for CosineAnnealingLR')
    
    # ReduceLROnPlateau 特定参数
    parser.add_argument('--plateau_mode', type=str, default='min', choices=['min', 'max'], help='mode for ReduceLROnPlateau')
    parser.add_argument('--patience', type=int, default=5, help='patience for ReduceLROnPlateau')
    parser.add_argument('--threshold', type=float, default=1e-4, help='threshold for ReduceLROnPlateau')
    
    # CosineAnnealingWarmRestarts 特定参数
    parser.add_argument('--T_0', type=int, default=10, help='T_0 for CosineAnnealingWarmRestarts')
    parser.add_argument('--T_mult', type=int, default=2, help='T_mult for CosineAnnealingWarmRestarts')
    
    # OneCycleLR 特定参数
    parser.add_argument('--max_lr', type=float, default=0.01, help='max learning rate for OneCycleLR')
    parser.add_argument('--total_steps', type=int, default=10000, help='total steps for OneCycleLR')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='steps per epoch for OneCycleLR')
    parser.add_argument('--pct_start', type=float, default=0.3, help='percentage of rising phase for OneCycleLR')
    parser.add_argument('--anneal_strategy', type=str, default='cos', choices=['cos', 'linear'], help='annealing strategy for OneCycleLR')
    
    args = parser.parse_args()
    # print(args)
    args_dict = vars(args)
    args_dict_environ = {}
    for key, value in args_dict.items():
        if key in ['input_path', 'output_path']:
            args_dict_environ[key] = type_switch(os.getenv(key.upper(), value), value)
        else:
            args_dict_environ[key] = type_switch(os.getenv(key, value), value)
    args_easydict = EasyDict(args_dict_environ)
    return args_easydict
    
def type_switch(environ_value, value):
    if isinstance(value, bool):
        return bool(environ_value)
    elif isinstance(value, int):
        return int(environ_value)
    elif isinstance(value, float):
        return float(environ_value)
    elif isinstance(value, str):
        return environ_value
    elif isinstance(value, tuple):
        return environ_value
    elif isinstance(value, list):
        return environ_value
    
def add_args(args):
    model_yaml = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*/'), '*.yaml'))[0]
    # print(model_yaml)
    model_name = os.path.splitext(os.path.basename(model_yaml))[0]
    # print(model_name)
    args.model_yaml = model_yaml
    args.model_name = model_name
    if args.process == 'test' or ((args.process == 'train' or args.process == 'fit') and args.resume_from_checkpoint):
        model_path = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*/'), '*.pth'))[0]
        # print(model_path)
        args.model_path = model_path
    else:
        args.model_path = None
    
    data_yaml = glob.glob(os.path.join(os.path.join(f'{args.input_path}/data', '*/'), '*.yaml'))[0]
    # print(data_yaml)
    data_name = os.path.splitext(os.path.basename(data_yaml))[0]
    # print(data_name)
    args.data_yaml = data_yaml
    args.data_name = data_name
    data_path = glob.glob(os.path.join(os.path.join(f'{args.input_path}/data', '*/'), '*/'))[0]
    args.data_path = data_path
    
    if data_name == 'mnist':
        args.num_classes = 10
    elif data_name == 'cifar10':
        args.num_classes = 10
    elif data_name == 'cifar100':
        args.num_classes = 100
    elif data_name == 'imagenet':
        args.num_classes = 1000
        
    return args


def load_yaml(args):
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)


def main(args):
    args = add_args(args)
    if args.process == 'train':
        train(args)
    elif args.process == 'test':
        test(args)
    elif args.process == 'fit':
        fit(args)
    else:
        raise ValueError('任务不支持.')
    
if __name__ == '__main__':
    args = parse_args()
    # print(args)
    sse_input_path_validated(args)
    sse_output_path_validated(args)
    main(args)
    
    
    
    
    
    
