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

from torchattacks import FGSM, PGD, BIM, CW, DeepFool
from attacks.utils import imshow, get_pred
from torchvision import transforms

from utils.sse import sse_print

def defend(args):
    device = args.device
    
    try:
        # print(args.data_name)
        if args.data_name == 'cifar10':
            dataset = BenchmarkDataset.cifar_10
        elif args.data_name == 'cifar100':
            dataset = BenchmarkDataset.cifar_100
        else:
            raise ValueError('数据集不支持, 仅支持数据集: cifar10, cifar100.')
        
        event = "data_load_validated"
        data = {
            "status": "success",
            "message": "数据集加载完毕.",
            "data name": args.data_name,
            "data path": args.data_path
        }
        sse_print(event, data)
    except Exception as e:
        event = "data_load_validated"
        data = {
            "status": "failure",
            "message": f"{e}",
            "data name": args.data_name,
            "data path": args.data_path
        }
        sse_print(event, data)
        
    
    '''
    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it overrides ``threat_model``
    '''
    model = load_model(
        model_name = args.model_name,
        model_path = '',
        dataset = dataset,
        threat_model = ThreatModel.Linf, # 默认值
        norm='Linf'
        ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # 加入if选择防御方法
    event = "defend_init"
    data = {
        "status": "success",
        "message": "防御初始化完成.",
        "defend_method": args.defend_method
    }
    sse_print(event, data)
    
    atk = FGSM(model, eps=args.epsilon)
    ori_adv_loader = atk.load(
        load_path=args.data_path,
        batch_size=args.batch,
        shuffle=False,
        normalize=None,
        load_predictions=False,
        load_clean_inputs=True,
        )

    for epoch in range(args.epochs):
        # 记录训练统计信息
        running_loss = 0.0
        correct = 0
        total = 0
        total_batch = len(ori_adv_loader)
        for batch_i, batch_data in enumerate(ori_adv_loader):
            adv_images, labels, ori_images = batch_data
            # print(adv_images.shape)
            # print(labels.shape)
            # print(ori_images.shape)
            adv_images, labels = adv_images.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()  # 清除梯度
            outputs = model(adv_images)  # 使用adv_images进行训练
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
                    "image_size": adv_images.shape[-1]
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
            "loss": f"{current_loss:.4f}", 
            "accuracy": f"{current_acc:.2f}%"
        }
    }
    sse_print(event, data)
    
