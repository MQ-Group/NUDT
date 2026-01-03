import torch.optim.lr_scheduler as lr_scheduler

def configure_scheduler(optimizer, args):
    """
    配置学习率调度器
    
    参数:
        optimizer: 优化器对象
        args: 包含所有配置参数的对象
        
    返回:
        配置好的学习率调度器
    """
    if args.scheduler == 'steplr':
        # StepLR: 固定步长调整
        # 参数: step_size=每隔多少epoch调整, gamma=调整系数
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_decay_step, 
            gamma=args.lr_decay_rate
        )
        
    elif args.scheduler == 'multisteplr':
        # MultiStepLR: 多步调整
        # 参数: milestones=在哪些epoch调整, gamma=调整系数
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.milestones,  # 例如: [30, 60, 90]
            gamma=args.lr_decay_rate
        )
        
    elif args.scheduler == 'exponential':
        # ExponentialLR: 指数衰减
        # 参数: gamma=指数衰减系数
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.lr_decay_rate
        )
        
    elif args.scheduler == 'cosine':
        # CosineAnnealingLR: 余弦退火
        # 参数: T_max=最大迭代次数, eta_min=最小学习率
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.T_max or args.max_epochs,
            eta_min=args.lr_decay_min_lr or 0
        )
        
    elif args.scheduler == 'cyclic':
        # CyclicLR: 循环学习率
        # 参数: base_lr=基础学习率, max_lr=最大学习率
        #       step_size_up=上升步数, step_size_down=下降步数
        scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.base_lr or args.lr * 0.1,
            max_lr=args.max_lr or args.lr * 1.0,
            step_size_up=args.step_size_up or 2000,
            step_size_down=args.step_size_down or None,
            mode=args.cyclic_mode or 'triangular',  # 'triangular', 'triangular2', 'exp_range'
            gamma=args.cyclic_gamma or 1.0,
            scale_fn=None,
            scale_mode='cycle',
            cycle_momentum=True,
            base_momentum=0.8,
            max_momentum=0.9,
            last_epoch=-1
        )
        
    elif args.scheduler == 'onecycle':
        # OneCycleLR: 单周期学习率
        # 参数: max_lr=最大学习率, total_steps=总步数
        #       pct_start=上升阶段比例, anneal_strategy=退火策略
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr or args.lr * 10,
            total_steps=args.total_steps or args.max_epochs * args.steps_per_epoch,
            epochs=args.epochs or args.max_epochs,
            steps_per_epoch=args.steps_per_epoch or 1000,
            pct_start=args.pct_start or 0.3,
            anneal_strategy=args.anneal_strategy or 'cos',  # 'cos' 或 'linear'
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4,
            last_epoch=-1
        )
        
    elif args.scheduler == 'lambda':
        # LambdaLR: 自定义函数调整
        # 参数: lr_lambda=自定义函数或函数列表
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=args.lr_lambda
        )
        
    elif args.scheduler == 'cosine_warm':
        # CosineAnnealingWarmRestarts: 带重启的余弦退火
        # 参数: T_0=第一次重启周期长度, T_mult=周期增长倍数
        #       eta_min=最小学习率
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0 or 10,  # 第一次重启的周期长度
            T_mult=args.T_mult or 1,  # 周期增长倍数
            eta_min=args.lr_decay_min_lr or 0,
            last_epoch=-1
        )
        
    else:
        raise ValueError(f'Invalid scheduler type: {args.scheduler}! '
                        f'Supported types: steplr, multisteplr, exponential, cosine, '
                        f'plateau, cyclic, onecycle, lambda, cosine_warm')
    
    return scheduler