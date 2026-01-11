import torch
from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

from data.VOCDatasetAdapter import VOCDatasetAdapter

import os
from torchattacks import FGSM, MIFGSM, VMIFGSM, PGD, BIM, CW, DeepFool, GN, Jitter
from sse.sse import sse_print

def collate_fn(batch):
    """
    自定义批处理函数，因为图像大小不同，目标字典结构不同。
    """
    return tuple(zip(*batch))



def adv_train(args):
    device = args.device
    
    train_dataset_raw = VOCDetection(
        root=args.data_path,
        year='2007',
        image_set='train',
        download=False,
        transform=None  # 我们将在适配器中处理变换
    )
    train_dataset = VOCDatasetAdapter(train_dataset_raw)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    
    event = "data_load"
    data = {
        "status": "success",
        "message": "数据集加载完毕.",
        "data_name": args.data_name,
        "data_path": args.data_path
    }
    sse_print(event, data)
    
    if args.model_name == 'ssd':
        model = ssd300_vgg16(
            weights=None,
            progress=True,
            num_classes=len(train_dataset.classes),
            weights_backbone=None,
            trainable_backbone_layers=None,
            score_thresh=0.1,
            nms_thresh=0.45,
            detections_per_img=10,
            iou_thresh=0.5,
            topk_candidates=400,
            positive_fraction=0.25,
        )
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            progress=True,
            num_classes=len(train_dataset.classes),
            weights_backbone=None,
            trainable_backbone_layers=None,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
            bbox_reg_weights=None,
        )
    
    event = "model_build"
    data = {
        "status": "success",
        "message": "模型创建完毕.",
        "model_name": args.model_name,
        "class_name": train_dataset.classes
    }
    sse_print(event, data)
    
    model.load_state_dict(torch.load(args.model_path, weights_only=False))
    model.to(device)
    
    event = "model_load"
    data = {
        "status": "success",
        "message": "模型加载完毕.",
        "model_name": args.model_name,
        "num_path": args.model_path
    }
    sse_print(event, data)
    
    
    try:
        # print(args.attack_method)
        if args.attack_method == 'fgsm':
            atk = FGSM(model, eps=args.epsilon)
        elif args.attack_method == 'mifgsm':
            atk = MIFGSM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, decay=args.decay)
        elif args.attack_method == 'vmifgsm':
            atk = VMIFGSM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, decay=args.decay, N=args.sampled_examples, beta=3/2)
        elif args.attack_method == 'pgd':
            atk = PGD(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, random_start=args.random_start)
        elif args.attack_method == 'bim':
            atk = BIM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.attack_method == 'cw':
            atk = CW(model, c=1, kappa=0, steps=args.max_iterations, lr=args.lr)
        elif args.attack_method == 'deepfool':
            # atk = DeepFool(model, steps=args.max_iterations, overshoot=0.02)
            atk = CW(model, c=1, kappa=0, steps=args.max_iterations, lr=args.lr)
        elif args.attack_method == 'gn':
            atk = GN(model, std=args.std)
        elif args.attack_method == 'jitter':
            # atk = Jitter(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, scale=args.scale, std=args.std, random_start=args.random_start)
            atk = PGD(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, random_start=args.random_start)
        else:
            raise ValueError('不支持的攻击方法.')

        event = "adversarial_samples_generation_init"
        data = {
            "status": "success",
            "message": "攻击方法初始化完成.",
            "attack_method": args.attack_method
        }
        sse_print(event, data)
    except Exception as e:
        event = "adversarial_samples_generation_init"
        data = {
            "status": "failure",
            "message": f"{e}",
            "attack_method": args.attack_method
        }
        sse_print(event, data)
        import sys
        sys.exit()
        
    # print(atk)
    # adv_images = atk(batch)
    
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[80000, 100000], gamma=0.1)
    
    total_batch = len(train_loader)
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_i, (images, targets) in enumerate(train_loader):
            # print(len(images))
            # print(len(targets))
            # print(images[0].shape) # (C, H, W)
            # print(images[0]) # 数值0~1
            # print(targets[0].keys())
            # print(targets[0]['boxes'].shape)
            # print(targets[0]['labels'].shape)
            
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            images = atk(images, targets)
            
            # 前向传播，计算损失
            loss_dict = model(images=images, targets=targets) # images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
            loss = loss_dict['bbox_regression'] + loss_dict['classification']
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            import math
            if batch_i % math.ceil(total_batch / 200.0) == 0:
                event = "train"
                data = {
                    "message": "正在执行对抗训练...",
                    "progress": int(epoch/args.epochs*100),
                    "log": f"[{int(epoch/args.epochs*100)}%] 正在执行对抗训练...",
                    "details": {
                        "epoch": f"{epoch + 1}/{args.epochs}",
                        "batch": f"{batch_i + 1}/{total_batch}",
                        "avg_loss": f"{total_loss / (batch_i + 1):.4f}", 
                        # "loss": f"{loss.item():.4f}", 
                        "box_loss": f"{loss_dict['bbox_regression'].item():.4f}", 
                        "class_loss": f"{loss_dict['classification'].item():.4f}", 
                        "batch_size": args.batch,
                        "image_size": images[0].shape[-1]
                    }
                }
                sse_print(event, data)

        # 更新学习率
        scheduler.step()

        model_weight_save_path = f"{args.output_path}/adversarial_trained_{args.model_name}.pt"
        torch.save(model.state_dict(), model_weight_save_path)
        os.system(f"cp {args.model_yaml} {args.output_path}")
        
    event = "final_result"
    data = {
        "message": "对抗训练执行完成.",
        "progress": 100,
        "log": f"[100%] 对抗训练执行完成.",
        "details": {
            "model_name": args.model_name,
            "checkpoint": model_weight_save_path, 
            "eopchs": args.epochs, 
            "batch_size": args.batch,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "avg_loss": f"{total_loss / total_batch:.4f}", 
        }
    }
    sse_print(event, data)