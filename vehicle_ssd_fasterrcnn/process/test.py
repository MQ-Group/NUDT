import torch
from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

from data.VOCDatasetAdapter import VOCDatasetAdapter

from sse.sse import sse_print

def collate_fn(batch):
    """
    自定义批处理函数，因为图像大小不同，目标字典结构不同。
    """
    return tuple(zip(*batch))


def test(args):
    device = args.device
    
    test_dataset_raw = VOCDetection(
        root=args.data_path,
        year='2007',
        image_set='val',
        download=False,
        transform=None  # 我们将在适配器中处理变换
    )
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
            num_classes=len(test_dataset.classes),
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
            num_classes=len(test_dataset.classes),
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
        "class_name": test_dataset.classes
    }
    sse_print(event, data)
    
    model.load_state_dict(torch.load(args.model_path, weights_only=False))
    model.to(device)
    
    results_dict = {}
    total_batch = len(test_loader)
    
    total_loss = 0
    total_scores = 0
    for batch_i, (images, targets) in enumerate(test_loader):
        # print(len(images))
        # print(len(targets))
        # print(images[0].shape) # (C, H, W)
        # print(images[0]) # 数值0~1
        # print(targets[0].keys())
        # print(targets[0]['boxes'].shape)
        # print(targets[0]['labels'].shape)
        
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        model.train()
        loss_dict = model(images=images, targets=targets) # images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        # print(loss_dict) # ssd 和 fasterrcnn不一样
        loss = sum(loss for loss in loss_dict.values())
        total_loss += loss.item()
        
        model.eval()
        predictions = model(images=images, targets=None)
        # print(len(predictions))
        # print(predictions[0].keys())
        # print(predictions[0]['boxes'].shape)
        # print(predictions[0]['scores'].shape)
        # print(predictions[0]['labels'].shape)
        
        scores = sum([predictions[i]['scores'].mean() for i in range(len(predictions))]) / args.batch
        total_scores += scores.item()
        
        
        import math
        if batch_i % math.ceil(total_batch / 200.0) == 0:
            event = "test"
            data = {
                "message": "正在执行测试...",
                "progress": int(batch_i/total_batch*100),
                "log": f"[{int(batch_i/total_batch*100)}%] 正在执行测试...",
                "details": {
                    "batch": f"{batch_i + 1}/{total_batch}",
                    "total_loss": f"{loss.item():.4f}", 
                    "loss": {
                        key: f"{val.item():.4f}" for key, val in loss_dict.items()
                    },
                    "confidence": f"{scores.item():.4f}", 
                    "batch_size": args.batch,
                    "image_size": images[0].shape[-1]
                }
            }
            sse_print(event, data)

    
    event = "final_result"
    data = {
        "message": "测试执行完成.",
        "progress": 100,
        "log": f"[100%] 测试执行完成.",
        "details": {
            "avg_loss": f"{total_loss/total_batch:.4f}", 
            "avg_confidence": f"{total_scores/total_batch:.4f}"
        }
    }
    sse_print(event, data)