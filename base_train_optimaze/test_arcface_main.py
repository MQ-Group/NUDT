#!/usr/bin/env python3
"""
按照 main.py 的方式测试 ArcFace 模型，检查 SSE 输出是否正常
"""
import os
import sys
import argparse
from easydict import EasyDict
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sse import sse_input_path_validated, sse_output_path_validated
from test import test

def create_test_args():
    """创建测试参数，模拟 main.py 的参数解析"""
    base_dir = Path('/data6/user23215430/nudt/base_train_optimaze')
    
    # 创建 args 对象，模拟 main.py 的参数
    args = EasyDict({
        'input_path': str(base_dir / 'input'),
        'output_path': str(base_dir / 'output'),
        'process': 'test',
        'resume_from_checkpoint': True,  # test 模式需要加载权重
        'epochs': 1,
        'batch': 8,  # 使用较小的 batch size
        'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
        'workers': 0,
        'optimizer': 'adam',
        'scheduler': 'steplr',
        'loss_function': 'cross_entropy',
        
        # 优化器参数
        'lr': 0.001,
        'weight_decay': 0.0,
        'momentum': 0.9,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'amsgrad': False,
        'dampening': 0,
        'nesterov': False,
        'alpha': 0.99,
        'centered': False,
        'rho': 0.9,
        'max_iter': 20,
        'history_size': 100,
        'lambd': 1e-4,
        'alpha_asgd': 0.75,
        't0': 1e6,
        
        # 调度器参数
        'lr_decay_rate': 0.1,
        'lr_decay_step': 10,
        'lr_decay_min_lr': 1e-6,
        'max_epochs': 100,
        'milestones': [30, 60, 90],
        'T_max': 50,
        'plateau_mode': 'min',
        'patience': 5,
        'threshold': 1e-4,
        'T_0': 10,
        'T_mult': 2,
        'max_lr': 0.01,
        'total_steps': 10000,
        'steps_per_epoch': 100,
        'pct_start': 0.3,
        'anneal_strategy': 'cos',
    })
    
    # 添加模型和数据集的路径（模拟 add_args 函数）
    import glob
    model_dir = Path(args.input_path) / 'model' / 'ArcFace'
    args.model_yaml = str(model_dir / 'ArcFace.yaml')
    args.model_name = 'ArcFace'
    args.model_path = str(model_dir / 'ArcFace.pth')
    
    data_dir = Path(args.input_path) / 'data' / 'cifar10'
    args.data_yaml = str(data_dir / 'cifar10.yaml')
    args.data_name = 'cifar10'
    
    # 查找实际的数据路径（CIFAR10 数据在 cifar10/cifar10/ 下）
    data_path_candidates = [
        data_dir / 'cifar10',
        data_dir,
    ]
    args.data_path = None
    for candidate in data_path_candidates:
        if candidate.exists():
            args.data_path = str(candidate)
            break
    
    if args.data_path is None:
        raise ValueError(f"找不到 CIFAR10 数据集路径")
    
    args.num_classes = 10  # CIFAR10 有 10 个类别
    
    return args

def main():
    """主函数"""
    print("="*60)
    print("测试 ArcFace 模型的 SSE 输出")
    print("="*60)
    
    try:
        # 创建测试参数
        args = create_test_args()
        
        print(f"\n配置信息:")
        print(f"  输入路径: {args.input_path}")
        print(f"  输出路径: {args.output_path}")
        print(f"  模型名称: {args.model_name}")
        print(f"  模型路径: {args.model_path}")
        print(f"  数据集名称: {args.data_name}")
        print(f"  数据集路径: {args.data_path}")
        print(f"  设备: {args.device}")
        print(f"  Batch Size: {args.batch}")
        print()
        
        # 验证路径（模拟 main.py 的验证步骤）
        print("="*60)
        print("验证输入输出路径...")
        print("="*60)
        sse_input_path_validated(args)
        sse_output_path_validated(args)
        
        print()
        print("="*60)
        print("开始测试 ArcFace 模型...")
        print("="*60)
        print()
        
        # 运行测试
        test(args)
        
        print()
        print("="*60)
        print("测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

