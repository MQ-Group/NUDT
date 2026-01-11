import torch
from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from PIL import Image
import os
import glob
from sse.sse import sse_print


def predict(args):
    device = args.device
    
    classes = [
            '__background__',  # 索引 0 保留给背景
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    
    if args.model_name == 'ssd':
        model = ssd300_vgg16(
            weights=None,
            progress=True,
            num_classes=len(classes),
            weights_backbone=None,
            trainable_backbone_layers=None,
            score_thresh=0.1,
            nms_thresh=0.45,
            # detections_per_img=200,
            detections_per_img=2,
            iou_thresh=0.5,
            topk_candidates=400,
            positive_fraction=0.25,
        )
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            progress=True,
            num_classes=len(classes),
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
        "class_name": classes
    }
    sse_print(event, data)
    
    model.load_state_dict(torch.load(args.model_path, weights_only=False))
    model.to(device)
    
    event = "model_load"
    data = {
        "status": "success",
        "message": "模型加载完成.",
        "model_name": args.model_name,
        "model_path": args.model_path
    }
    sse_print(event, data)
    
    # images_flod = f"{args.data_path}ori_images"
    images_flod = glob.glob(os.path.join(f'{args.data_path}', '*/'))[0]
    # print(images_flod)
    
    images_paths = glob.glob(os.path.join(images_flod, '*.jpg'))
    # print(len(image_paths))
    
    event = "data_load"
    data = {
        "status": "success",
        "message": "样本加载完毕.",
        "data_name": args.data_name,
        "samples_number": len(images_paths),
        "data_path": args.data_path
    }
    sse_print(event, data)

    pred_images_flod = f"{args.output_path}/pred_images"
    os.makedirs(pred_images_flod, exist_ok=True)

    model.eval()

    total_iamges = len(images_paths)
    # ori_pred_conf_sum = 0.0
    for i in range(total_iamges):
        # print(images_paths[i])
        
        images = Image.open(images_paths[i])
        to_tensor = transforms.ToTensor()
        images = to_tensor(images)
        
        images = images.to(device)
        predictions = model(images=[images], targets=None)
        
        # print(len(predictions))
        # print(predictions[0].keys())
        # print(predictions[0]['boxes'].shape)
        # print(predictions[0]['scores'].shape)
        # print(predictions[0]['labels'].shape)
        
        labels = [classes[i] for i in predictions[0]["labels"]]
        images = draw_bounding_boxes(
                                images, 
                                boxes=predictions[0]["boxes"],
                                labels=labels,
                                colors="red",
                                width=2,
                                font_size=16)
        
        pred_images_path = f"{pred_images_flod}/img_{i}.jpg"
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(images)
        pil_image.save(pred_images_path)
        
        
        event = "predict"
        data = {
            "message": "正在执行预测...",
            "progress": int(i/total_iamges*100),
            # "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击... 原始样本: {ori_img_path}, 原始样本预测准确率: {ori_pred_conf*100:.2f}%, 原始样本预测类别: {ori_pred_cls.item()}, 对抗样本: {adv_img_path}, 对抗样本预测准确率: {adv_pred_conf*100:.2f}%, 对抗样本预测类别: {ori_pred_cls.item()}, 原始样本实际类别: {labels[i].item()}, 攻击结果: {attack_result}"
            "log": f"[{int(i/total_iamges*100)}%] 正在执行预测...",
            "details": {
                # "confidence": f"{ori_pred_conf*100:.2f}%",
                # "predict_class": ori_pred_cls,
                "object_number": len(labels),
                "predict_class": labels,
                "confidence": predictions[0]['scores'].tolist(),
                "file_path": pred_images_path
            }
        }
        sse_print(event, data)
        
    
    event = "final_result"
    data = {
        "message": "预测执行完成.",
        "progress": 100,
        "log": f"[100%] 预测执行完成.",
        "details": {
            "samples": pred_images_flod,
            "total_iamges": total_iamges,
            # "summary": {
            #     "total_iamges": total_iamges,
            #     "samples_pred_confidence": f"{ori_pred_conf_sum/total_iamges*100:.2f}%"
            # }
        }
    }
    sse_print(event, data)