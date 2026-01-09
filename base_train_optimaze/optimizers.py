
import torch.optim as optim

def configure_optimizer(model, args):
    """
    配置优化器
    
    参数:
        model: 模型对象
        args: 包含所有配置参数的对象
        
    返回:
        配置好的优化器
    """
    # 从args中获取参数，如果不存在则使用默认值
    lr = getattr(args, 'lr', 0.001)
    weight_decay = getattr(args, 'weight_decay', 0.0)
    momentum = getattr(args, 'momentum', 0.9)
    
    if args.optimizer == 'sgd':
        # SGD: 随机梯度下降
        # 参数: momentum=动量, dampening=阻尼, nesterov=使用Nesterov动量
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=momentum,
            dampening=getattr(args, 'dampening', 0),
            weight_decay=weight_decay,
            nesterov=getattr(args, 'nesterov', False)
        )
        
    elif args.optimizer == 'adam':
        # Adam: 自适应矩估计
        # 参数: betas=(beta1, beta2), eps=数值稳定性常数
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr,
            betas=getattr(args, 'betas', (0.9, 0.999)),
            eps=getattr(args, 'eps', 1e-8),
            weight_decay=weight_decay,
            amsgrad=getattr(args, 'amsgrad', False)
        )
        
    elif args.optimizer == 'adamw':
        # AdamW: Adam with decoupled weight decay
        # 更适合Transformer等模型
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=getattr(args, 'betas', (0.9, 0.999)),
            eps=getattr(args, 'eps', 1e-8),
            weight_decay=weight_decay,
            amsgrad=getattr(args, 'amsgrad', False)
        )
        
    elif args.optimizer == 'rmsprop':
        # RMSprop: 均方根传播
        # 参数: alpha=平滑常数, momentum=动量
        optimizer = optim.RMSprop(
            model.parameters(), 
            lr=lr,
            alpha=getattr(args, 'alpha', 0.99),
            eps=getattr(args, 'eps', 1e-8),
            weight_decay=weight_decay,
            momentum=momentum,
            centered=getattr(args, 'centered', False)
        )
        
    elif args.optimizer == 'adagrad':
        # Adagrad: 自适应梯度算法
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=lr,
            lr_decay=getattr(args, 'lr_decay', 0),
            weight_decay=weight_decay,
            initial_accumulator_value=getattr(args, 'initial_accumulator_value', 0)
        )
        
    elif args.optimizer == 'adadelta':
        # Adadelta: 自适应学习率方法
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=lr,
            rho=getattr(args, 'rho', 0.9),
            eps=getattr(args, 'eps', 1e-6),
            weight_decay=weight_decay
        )
        
    elif args.optimizer == 'adamax':
        # Adamax: Adam的无穷范数变体
        optimizer = optim.Adamax(
            model.parameters(),
            lr=lr,
            betas=getattr(args, 'betas', (0.9, 0.999)),
            eps=getattr(args, 'eps', 1e-8),
            weight_decay=weight_decay
        )
        
    elif args.optimizer == 'nadam':
        # NAdam: Nesterov加速的Adam
        # 注意：PyTorch 1.10+ 支持
        try:
            optimizer = optim.NAdam(
                model.parameters(),
                lr=lr,
                betas=getattr(args, 'betas', (0.9, 0.999)),
                eps=getattr(args, 'eps', 1e-8),
                weight_decay=weight_decay,
                momentum_decay=getattr(args, 'momentum_decay', 4e-3)
            )
        except AttributeError:
            raise ImportError("NAdam requires PyTorch >= 1.10.0")
            
    elif args.optimizer == 'radam':
        # RAdam: 修正的Adam
        # 需要实现RAdam优化器或从外部导入
        try:
            from radam import RAdam
            optimizer = RAdam(
                model.parameters(),
                lr=lr,
                betas=getattr(args, 'betas', (0.9, 0.999)),
                eps=getattr(args, 'eps', 1e-8),
                weight_decay=weight_decay
            )
        except ImportError:
            raise ImportError("RAdam is not installed. Install with: pip install radam")
            
    elif args.optimizer == 'lbfgs':
        # L-BFGS: 准牛顿方法
        # 注意：L-BFGS在每次迭代中需要多次计算函数
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=getattr(args, 'max_iter', 20),
            max_eval=getattr(args, 'max_eval', None),
            tolerance_grad=getattr(args, 'tolerance_grad', 1e-5),
            tolerance_change=getattr(args, 'tolerance_change', 1e-9),
            history_size=getattr(args, 'history_size', 100),
            line_search_fn=getattr(args, 'line_search_fn', None)
        )
        
    elif args.optimizer == 'sgd_nesterov':
        # SGD with Nesterov momentum
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
        
    elif args.optimizer == 'asgd':
        # ASGD: 平均随机梯度下降
        optimizer = optim.ASGD(
            model.parameters(),
            lr=lr,
            lambd=getattr(args, 'lambd', 1e-4),
            alpha=getattr(args, 'alpha', 0.75),
            t0=getattr(args, 't0', 1e6),
            weight_decay=weight_decay
        )
        
    elif args.optimizer == 'rprop':
        # Rprop: 弹性反向传播
        optimizer = optim.Rprop(
            model.parameters(),
            lr=lr,
            etas=getattr(args, 'etas', (0.5, 1.2)),
            step_sizes=getattr(args, 'step_sizes', (1e-6, 50))
        )
        
    else:
        raise ValueError(f'Invalid optimizer type: {args.optimizer}! '
                        f'Supported types: sgd, adam, adamw, rmsprop, adagrad, adadelta, '
                        f'adamax, nadam, radam, lbfgs, sgd_nesterov, asgd, rprop')
    
    return optimizer