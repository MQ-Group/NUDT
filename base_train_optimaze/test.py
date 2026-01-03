
import torch
from datasets import get_dataset
from models import get_model, load_weight
from loss_functions import get_loss_fn
from optimizers import configure_optimizer
from schedulers import configure_scheduler
from torch.utils.data import DataLoader

from utils.sse import sse_print

def test(args):
    device = args.device
    
    test_dataset, num_classes = get_dataset(data_name=args.data_name, data_path=args.data_path, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    
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
    
    model.to(device)
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    total_batch = len(test_loader)
    for batch_i, batch_data in enumerate(test_loader):
        inputs, labels = batch_data
        # print(inputs.shape)
        # print(labels.shape)
        inputs, labels = inputs.to(device), labels.to(device)
                
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        current_loss = running_loss / (batch_i + 1)
        current_acc = 100. * correct / total
        
        event = "test"
        data = {
            "message": "正在执行测试...",
            "progress": int(batch_i/total_batch*100),
            "log": f"[{int(batch_i/total_batch*100)}%] 正在执行测试...",
            "details": {
                "batch": f"{batch_i + 1}/{total_batch}",
                "loss": f"{current_loss:.4f}", 
                "accuracy": f"{current_acc:.2f}%", 
                "batch_size": args.batch,
                "image_size": inputs.shape[-1]
            }
        }
        sse_print(event, data)
    
    
    event = "final_result"
    data = {
        "message": "测试执行完成.",
        "progress": 100,
        "log": f"[100%] 测试执行完成.",
        "details": {
            "loss": f"{current_loss:.4f}", 
            "accuracy": f"{current_acc:.2f}%"
        }
    }
    sse_print(event, data)