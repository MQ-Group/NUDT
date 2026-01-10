import os
import torch
from datasets import get_dataset
from models import get_model, load_weight
from loss_functions import get_loss_fn
from optimizers import configure_optimizer
from schedulers import configure_scheduler
from torch.utils.data import DataLoader

from utils.sse import sse_print

def train(args):
    device = args.device
    
    train_dataset, num_classes = get_dataset(data_name=args.data_name, data_path=args.data_path, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    
    event = "data_load"
    data = {
        "status": "success",
        "message": "数据集加载完毕.",
        "data_name": args.data_name,
        "data_path": args.data_path
    }
    sse_print(event, data)
    
    model = get_model(model_name=args.model_name, num_classes=num_classes)
    
    event = "model_build"
    data = {
        "status": "success",
        "message": "模型创建完毕.",
        "model_name": args.model_name,
        "num_classes": num_classes
    }
    sse_print(event, data)
    
    if args.resume_from_checkpoint:
        model = load_weight(model=model, pretrained_path=args.model_path)
        
        event = "model_load"
        data = {
            "status": "success",
            "message": "模型加载完毕.",
            "model_name": args.model_name,
            "num_path": args.model_path
        }
        sse_print(event, data)
    
    loss_fn = get_loss_fn(loss_fn_name=args.loss_function)
    
    optimizer = configure_optimizer(model, args)
    scheduler = configure_scheduler(optimizer, args)
    
    model.to(device)
    model.train()
    
    total_batch = len(train_loader)
    for epoch in range(args.epochs):
        for batch_i, batch_data in enumerate(train_loader):
            inputs, labels = batch_data
            # print(inputs.shape)
            # print(labels.shape)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # 统计信息
            _, predicted = torch.max(outputs, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = 100. * correct / total
            
            import math
            if batch_i % math.ceil(total_batch / 200.0) == 0:
                event = "train"
                data = {
                    "message": "正在执行训练...",
                    "progress": int(epoch/args.epochs*100),
                    "log": f"[{int(epoch/args.epochs*100)}%] 正在执行训练...",
                    "details": {
                        "epoch": f"{epoch + 1}/{args.epochs}",
                        "batch": f"{batch_i + 1}/{total_batch}",
                        "loss": f"{loss.item():.4f}", 
                        "accuracy": f"{accuracy:.2f}%", 
                        "batch_size": args.batch,
                        "image_size": inputs.shape[-1]
                    }
                }
                sse_print(event, data)

        # 更新学习率
        scheduler.step()
    
        model_weight_save_path = f"{args.output_path}/trained_{args.model_name}.pth"
        torch.save(model.state_dict(), model_weight_save_path)
        os.system(f"cp {args.model_yaml} {args.output_path}")
        
    
    event = "final_result"
    data = {
        "message": "训练执行完成.",
        "progress": 100,
        "log": f"[100%] 训练执行完成.",
        "details": {
            "model_name": args.model_name,
            "checkpoint": model_weight_save_path,
            "eopchs": args.epochs, 
            "batch_size": args.batch,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
    }
    sse_print(event, data)