import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, '..')
import torchattacks

sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10, load_cifar100
from robustbench.utils import load_model, clean_accuracy

from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from torchattacks import FGSM, PGD, BIM, CW, DeepFool, GN, Jitter, Boundary, ZOO, HSJA, NES
from torchdefends import YOPO, TRADES, FREE, FAST
from .utils import imshow, get_pred
from torchvision import transforms

from PIL import Image
import glob

from utils.sse import sse_print

def attack_use_adv(args):
    device = args.device
    
    try:
        # print(args.data_name)
        if args.data_name == 'cifar10':
            dataset = BenchmarkDataset.cifar_10
        elif args.data_name == 'cifar100':
            dataset = BenchmarkDataset.cifar_100
        else:
            raise ValueError('数据集不支持, 仅支持数据集: cifar10, cifar100.')
        
        ori_images_flod = f"{args.data_path}ori_images"
        adv_images_flod = f"{args.data_path}adv_images"
        
        ori_image_paths = glob.glob(os.path.join(ori_images_flod, '*.jpg'))
        adv_image_paths = glob.glob(os.path.join(adv_images_flod, '*.jpg'))
        ori_image_paths.sort()
        adv_image_paths.sort()
    
        ori_images = []
        adv_images = []
        labels = []
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        for ori_image_path in ori_image_paths:
            # print(ori_image_path)
            ori_image = Image.open(ori_image_path)
            ori_image = transform(ori_image)
            ori_images.append(ori_image)
            
            # print(os.path.basename(ori_image_path))
            # print(os.path.splitext(os.path.basename(ori_image_path)))
            # print(os.path.splitext(os.path.basename(ori_image_path))[0].split('_'))
            label = int(os.path.splitext(os.path.basename(ori_image_path))[0].split('_')[-1])
            # print(label)
            labels.append(label)
            
        for adv_image_path in adv_image_paths:
            adv_image = Image.open(adv_image_path)
            adv_image = transform(adv_image)
            adv_images.append(adv_image)
                    
        ori_images = torch.stack(ori_images, dim=0)
        adv_images = torch.stack(adv_images, dim=0)
        labels = torch.Tensor(labels)
        
        # print(ori_images.shape)
        # print(labels.shape)
        # print(adv_images.shape)
        
        event = "data_load_validated"
        data = {
            "status": "success",
            "message": "对抗样本和原始样本加载完毕.",
            "data_name": args.data_name,
            "samples_number": adv_images.shape[0],
            "data_path": args.data_path
        }
        sse_print(event, data)
    except Exception as e:
        event = "data_load_validated"
        data = {
            "status": "failure",
            "message": f"{e}",
            "data_name": args.data_name,
            "data_path": args.data_path
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
        model_path = args.model_path,
        dataset = dataset,
        threat_model = ThreatModel.Linf, # 默认值
        norm='Linf'
        ).to(device)
    
    event = "attack_init"
    data = {
        "status": "success",
        "message": "攻击初始化完成."
    }
    sse_print(event, data)
    
    # print(atk)
    
    total_iamges = ori_images.shape[0]
    ori_acc_sum = 0.0
    adv_acc_sum = 0.0
    attack_success_count = 0
    attack_failure_count = 0
    
    for i in range(total_iamges):
        ori_acc = clean_accuracy(model, ori_images[i:i+1].to(device), labels[i:i+1].to(device))
        adv_acc = clean_accuracy(model, adv_images[i:i+1].to(device), labels[i:i+1].to(device))
        # print('Acc: %2.2f %%'%(ori_acc*100))
        # print('Acc: %2.2f %%'%(adv_acc*100))
        ori_acc_sum += ori_acc*100
        adv_acc_sum += adv_acc*100
        
        ori_pred_cls = get_pred(model, ori_images[i:i+1], device)
        adv_pred_cls = get_pred(model, adv_images[i:i+1], device)
        # print(ori_pred_cls)
        # print(adv_pred_cls)
        
        if ori_pred_cls != adv_pred_cls:
            # attack_result = '成功'
            attack_result = 'success'
            attack_success_count += 1
        else:
            # attack_result = '失败'
            attack_result = 'failure'
            attack_failure_count += 1
            

        ori_img_path = ori_image_paths[i]
        adv_img_path = adv_image_paths[i]
        
        event = "attack_inference"
        data = {
            "message": "正在执行攻击推理...",
            "progress": int(i/total_iamges*100),
            # "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击推理... 原始样本: {ori_img_path}, 原始样本预测准确率: {ori_acc*100:.2f}%, 原始样本预测类别: {ori_pred_cls.item()}, 对抗样本: {adv_img_path}, 对抗样本预测准确率: {adv_acc*100:.2f}%, 对抗样本预测类别: {ori_pred_cls.item()}, 原始样本实际类别: {labels[i].item()}, 攻击结果: {attack_result}"
            "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击推理...",
            "details": {
                "original_sample": {
                    "accuracy": f"{ori_acc*100:.2f}%",
                    "predict_class": ori_pred_cls.item(),
                    "file_path": ori_img_path
                },
                "adversarial_sample": {
                    "accuracy": f"{adv_acc*100:.2f}%",
                    "predict_class": adv_pred_cls.item(),
                    "file_path": adv_img_path
                },
                "actual_class": labels[i].item(),
                "attack_result": attack_result
            }
        }
        sse_print(event, data)
    

    event = "final_result"
    data = {
        "message": "攻击推理执行完成.",
        "progress": 100,
        "log": f"[100%] 攻击推理执行完成.",
        "details": {
            "original_samples": ori_images_flod,
            "adversarial_samples": adv_images_flod,
            "summary": {
                "total_iamges": total_iamges,
                "task_success_count": attack_success_count,
                "task_failure_count": attack_failure_count,
                "original_samples_accuracy": f"{ori_acc_sum/total_iamges:.2f}%",
                "adversarial_samples_accuracy": f"{adv_acc_sum/total_iamges:.2f}%"
            }
        }
    }
    sse_print(event, data)


    
    '''
    ori_adv_loader = atk.load(
        load_path=args.data_path,
        batch_size=1,
        shuffle=False,
        normalize=None,
        load_predictions=False,
        load_clean_inputs=True,
        )
    
    ori_images_flod = f"{args.output_path}/ori_images"
    adv_images_flod = f"{args.output_path}/adv_images"
    os.makedirs(ori_images_flod, exist_ok=True)
    os.makedirs(adv_images_flod, exist_ok=True)
    
    total_iamges = len(ori_adv_loader)
    ori_acc_sum = 0.0
    adv_acc_sum = 0.0
    attack_success_count = 0
    attack_failure_count = 0
    
    for i, batch_data in enumerate(ori_adv_loader):
        adv_images, labels, ori_images = batch_data
        # print(adv_images.shape)
        # print(labels.shape)
        # print(ori_images.shape)
        
        ori_acc = clean_accuracy(model, ori_images.to(device), labels.to(device))
        adv_acc = clean_accuracy(model, adv_images.to(device), labels.to(device))
        # print('Acc: %2.2f %%'%(ori_acc*100))
        # print('Acc: %2.2f %%'%(adv_acc*100))
        ori_acc_sum += ori_acc*100
        adv_acc_sum += adv_acc*100
        
        ori_pred_cls = get_pred(model, ori_images, device)
        adv_pred_cls = get_pred(model, adv_images, device)
        # print(ori_pred_cls)
        # print(adv_pred_cls)
        
        if ori_pred_cls != adv_pred_cls:
            # attack_result = '成功'
            attack_result = 'success'
            attack_success_count += 1
        else:
            # attack_result = '失败'
            attack_result = 'failure'
            attack_failure_count += 1
            
        ori_img_save_path = f"{ori_images_flod}/ori_img_{i}_pred_cls_{ori_pred_cls.item()}.jpg"
        adv_img_save_path = f"{adv_images_flod}/adv_img_{i}_pred_cls_{adv_pred_cls.item()}.jpg"
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(ori_images[0])
        pil_image = to_pil(adv_images[0])
        pil_image.save(ori_img_save_path)
        pil_image.save(adv_img_save_path)
        
        event = "attack_inference"
        data = {
            "message": "正在执行攻击推理...",
            "progress": int(i/total_iamges*100),
            # "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击推理... 原始样本: {ori_img_save_path}, 原始样本预测准确率: {ori_acc*100:.2f}%, 原始样本预测类别: {ori_pred_cls.item()}, 对抗样本: {adv_img_save_path}, 对抗样本预测准确率: {adv_acc*100:.2f}%, 对抗样本预测类别: {ori_pred_cls.item()}, 攻击结果: {attack_result}"
            "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击推理...",
            "details": {
                "original_sample": {
                    "accuracy": f"{ori_acc*100:.2f}%",
                    "predict_class": ori_pred_cls.item(),
                    "file_path": ori_img_save_path
                },
                "adversarial_sample": {
                    "accuracy": f"{adv_acc*100:.2f}%",
                    "predict_class": adv_pred_cls.item(),
                    "file_path": adv_img_save_path
                },
                "attack_result": attack_result
            }
        }
        sse_print(event, data)
    

    event = "final_result"
    data = {
        "message": "攻击推理执行完成.",
        "progress": 100,
        "log": f"[100%] 攻击推理执行完成.",
        "details": {
            "original_samples": ori_images_flod,
            "adversarial_samples": adv_images_flod,
            "summary": {
                "total_iamges": total_iamges,
                "task_success_count": attack_success_count,
                "task_failure_count": attack_failure_count,
                "original_samples_accuracy": f"{ori_acc_sum/total_iamges:.2f}%",
                "adversarial_samples_accuracy": f"{adv_acc_sum/total_iamges:.2f}%"
            }
        }
    }
    sse_print(event, data)
    '''