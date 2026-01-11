import torch
from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

from data.VOCDatasetAdapter import VOCDatasetAdapter

import os
from torchattacks import FGSM, MIFGSM, VMIFGSM, PGD, BIM, CW, DeepFool, GN, Jitter
from torchvision import transforms
from sse.sse import sse_print

def collate_fn(batch):
    """
    自定义批处理函数，因为图像大小不同，目标字典结构不同。
    """
    return tuple(zip(*batch))

def adv(args):
    device = args.device
    
    classes = args.classes
    
    test_dataset_raw = VOCDetection(
        root=args.data_path,
        year='2007',
        image_set='val',
        download=False,
        transform=None  # 我们将在适配器中处理变换
    )
    # print(test_dataset_raw[0])
    test_dataset = VOCDatasetAdapter(test_dataset_raw)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=False,
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
            num_classes=len(classes),
            weights_backbone=None,
            trainable_backbone_layers=None,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            detections_per_img=args.detections_per_img,
            iou_thresh=args.iou_thresh,
            topk_candidates=args.topk_candidates,
            positive_fraction=args.positive_fraction,
        )
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            progress=True,
            num_classes=len(classes),
            weights_backbone=None,
            trainable_backbone_layers=None,
            box_score_thresh=args.score_thresh,
            box_nms_thresh=args.nms_thresh,
            box_detections_per_img=args.detections_per_img,
            box_fg_iou_thresh=args.iou_thresh,
            box_bg_iou_thresh=args.iou_thresh,
            box_batch_size_per_image=512,
            box_positive_fraction=args.positive_fraction,
            bbox_reg_weights=None,
        )

    event = "model_build"
    data = {
        "status": "success",
        "message": "模型创建完毕.",
        "model_name": args.model_name,
        "class_name": classes
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
            "message": "对抗样本生成初始化完成.",
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
    
    ori_images_flod = f"{args.output_path}/ori_images"
    adv_images_flod = f"{args.output_path}/adv_images"
    os.makedirs(ori_images_flod, exist_ok=True)
    os.makedirs(adv_images_flod, exist_ok=True)
    
    if len(test_loader) < args.selected_samples:
        total_iamges = len(test_loader)
        event = "adversarial_samples_generation_number"
        data = {
            "status": "warning",
            "message": f"警告: 数据集的样本数{len(test_loader)}小于所要生成的对抗样本数量{args.selected_samples}"
        }
        sse_print(event, data)
    else:
        total_iamges = args.selected_samples
        
    event = "adversarial_samples_generation_run"
    for i, (images, targets) in enumerate(test_loader):
        # print(len(images))
        # print(len(targets))
        # print(images[0].shape) # (C, H, W)
        # print(images[0]) # 数值0~1
        # print(targets[0].keys())
        # print(targets[0]['boxes'].shape)
        # print(targets[0]['labels'].shape)
        
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        adv_images = atk(images, targets)
        
        
        ori_img_save_path = f"{ori_images_flod}/ori_img_{i}_obj_{targets[0]['labels'].shape[0]}.jpg"
        adv_img_save_path = f"{adv_images_flod}/adv_img_{i}_obj_{targets[0]['labels'].shape[0]}.jpg"
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(images[0])
        pil_image.save(ori_img_save_path)
        pil_image = to_pil(adv_images[0])
        pil_image.save(adv_img_save_path)
        
        data = {
            "status": "success",
            "message": "生成对抗样本...",
            "progress": int(i/total_iamges*100),
            "log": f"[{int(i/total_iamges*100)}%] 正在生成第{i}张对抗样本, 总共需要生成{total_iamges}张."
        }
        sse_print(event, data)
        
        if i == total_iamges - 1:
            break

    os.system(f"cp {args.data_yaml} {args.output_path}")
    
    event = "final_result"
    data = {
        "status": "success",
        "message": "对抗样本生成完成",
        "progress": 100,
        "log": f"[100%] 对抗样本生成完成, 共生成{total_iamges}张.",
        "attack_method": args.attack_method,
        "original_samples": ori_images_flod,
        "adversarial_samples": adv_images_flod
    }
    sse_print(event, data)
    