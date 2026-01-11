import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, '..')
import torchattacks

sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10, load_cifar100
from robustbench.utils import load_model, clean_accuracy

from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from torchattacks import FGSM, PGD, BIM, CW, DeepFool, MIFGSM, VMIFGSM, GN, Jitter, Boundary, ZOO, HSJA, NES, PGDRS
from torchdefends import YOPO, TRADES, FREE, FAST
from attacks.utils import imshow, get_pred
from torchvision import transforms

import torchvision.datasets as datasets


from utils.sse import sse_print

def defend(args):
    device = args.device
    
    try:
        # print(args.data_name)
        if args.data_name == 'cifar10':
            # dataset = datasets.CIFAR10(
            #                     root=args.data_path,
            #                     train=True,
            #                     transform=transforms.Compose([transforms.ToTensor()]),
            #                     download=False)
            dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transforms.Compose([transforms.ToTensor()]))
            train_loader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=args.batch,
                                shuffle=True,
                                num_workers=args.workers)
            benchmark_dataset = BenchmarkDataset.cifar_10
        elif args.data_name == 'cifar100':
            # dataset = datasets.CIFAR10(
            #                     root=args.data_path,
            #                     train=True,
            #                     transform=transforms.Compose([transforms.ToTensor()]),
            #                     download=False)
            dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transforms.Compose([transforms.ToTensor()]))
            train_loader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=args.batch,
                                shuffle=True,
                                num_workers=args.workers)
            benchmark_dataset = BenchmarkDataset.cifar_100
        else:
            raise ValueError('数据集不支持, 仅支持数据集: cifar10, cifar100.')

        event = "data_load"
        data = {
            "status": "success",
            "message": "数据集加载完毕.",
            "data name": args.data_name,
            "data path": args.data_path
        }
        sse_print(event, data)
    except Exception as e:
        event = "data_load"
        data = {
            "status": "failure",
            "message": f"{e}",
            "data name": args.data_name,
            "data path": args.data_path
        }
        sse_print(event, data)
        
    # print(images.shape)
    # print(labels.shape)
    
    '''
    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it overrides ``threat_model``
    '''
    model = load_model(
        model_name = args.model_name,
        model_path = '', # 不传模型权重进行训练
        dataset = benchmark_dataset,
        threat_model = ThreatModel.Linf, # 默认值
        norm='Linf'
        ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    try:
        # print(args.defend_method)
        if args.defend_method == 'fgsm':
            atk = FGSM(model, eps=args.epsilon)
        elif args.defend_method == 'mifgsm':
            atk = MIFGSM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, decay=args.decay)
        elif args.defend_method == 'vmifgsm':
            atk = VMIFGSM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, decay=args.decay, N=args.sampled_examples, beta=3/2)
        elif args.defend_method == 'pgd':
            atk = PGD(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, random_start=args.random_start)
        elif args.defend_method == 'bim':
            atk = BIM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.defend_method == 'cw':
            atk = CW(model, c=1, kappa=0, steps=args.max_iterations, lr=args.lr)
        elif args.defend_method == 'deepfool':
            atk = DeepFool(model, steps=args.max_iterations, overshoot=0.02)
        elif args.defend_method == 'gn':
            atk = GN(model, std=args.std)
        elif args.defend_method == 'jitter':
            atk = Jitter(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, scale=args.scale, std=args.std, random_start=args.random_start)
        elif args.defend_method == 'boundary':
            atk = Boundary(model, max_queries=args.max_queries, init_epsilon=args.epsilon, spherical_step=0.01, orthogonal_step=args.step_size, binary_search_steps=args.binary_search_steps)
        elif args.defend_method == 'zoo':
            atk = ZOO(model, max_iterations=args.max_iterations, learning_rate=args.lr, binary_search_steps=args.binary_search_steps, init_const=0.01, beta=0.001, batch_size=128, resolution=1, early_stop_iters=10, abort_early=True)
        elif args.defend_method == 'hsja':
            atk = HSJA(model, max_queries=args.max_queries, norm=args.norm, gamma=0.01, init_num_evals=10, max_num_evals=10, stepsize_search='geometric_progression', num_iterations=args.max_iterations, constraint='L2', batch_size=128)
        elif args.defend_method == 'nes':
            atk = NES(model, max_queries=args.max_queries, epsilon=args.epsilon, learning_rate=args.lr, samples_per_draw=10, sigma=0.001, decay_factor=0.9, norm=args.norm, early_stop=True, loss_func='cross_entropy')
        elif args.defend_method == 'yopo':
            atk = YOPO(model, eps=args.epsilon)
        elif args.defend_method == 'pgdrs':
            atk = PGDRS(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, noise_type=args.noise_type, noise_sd=args.noise_sd, noise_batch_size=5, batch_max=2048)
        elif args.defend_method == 'trades':
            atk = TRADES(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.defend_method == 'free':
            atk = FREE(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.defend_method == 'fast':
            atk = FAST(model, eps=args.epsilon)
        else:
            raise ValueError('不支持的攻击方法.')

        event = "defend_init"
        data = {
            "status": "success",
            "message": "防御初始化完成.",
            "defend_method": args.defend_method
        }
        sse_print(event, data)
    except Exception as e:
        event = "defend_init"
        data = {
            "status": "failure",
            "message": f"{e}",
            "defend_method": args.defend_method
        }
        sse_print(event, data)    
    
    total_batch = len(train_loader)
    for epoch in range(args.epochs):
        # 记录训练统计信息
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_i, batch_data in enumerate(train_loader):
            images, labels = batch_data
            # print(images.shape)
            # print(labels.shape)
            images, labels = images.to(device), labels.to(device)
            
            if (batch_i/total_batch)*100 <= args.adversarial_sample_proportion: # 对抗样本占比
                model.eval()
                images = atk(images, labels) # 使用adv_images进行训练
            
            model.train()
            # 前向传播
            optimizer.zero_grad()  # 清除梯度
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_loss = running_loss / (batch_i + 1)
            current_acc = 100. * correct / total
            
            import math
            if batch_i % math.ceil(total_batch / 200.0) == 0:
                event = "defend_train"
                data = {
                    "message": "正在执行防御训练...",
                    "progress": int(epoch/args.epochs*100),
                    "log": f"[{int(epoch/args.epochs*100)}%] 正在防御执行训练...",
                    "details": {
                        "epoch": f"{epoch + 1}/{args.epochs}",
                        "batch": f"{batch_i + 1}/{total_batch}",
                        "loss": f"{current_loss:.4f}", 
                        "accuracy": f"{current_acc:.2f}%", 
                        "batch_size": args.batch,
                        "image_size": images.shape[-1]
                    }
                }
                sse_print(event, data)
    
    model_weight_save_path = f"{args.output_path}/defend_trained_{args.model_name}.pt"
    torch.save(model.state_dict(), model_weight_save_path)
    
    os.system(f"cp {args.model_yaml} {args.output_path}")
    
    event = "final_result"
    data = {
        "message": "防御训练执行完成.",
        "progress": 100,
        "log": f"[100%] 防御训练执行完成.",
        "details": {
            "defend_method": args.defend_method,
            "model_name": args.model_name,
            "checkpoint": model_weight_save_path, 
            "loss": f"{current_loss:.4f}", 
            "accuracy": f"{current_acc:.2f}%",
            "eopchs": args.epochs, 
            "batch_size": args.batch,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
    }
    sse_print(event, data)
    
