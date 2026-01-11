import os
from pathlib import Path
os.environ['YOLO_VERBOSE'] = 'false'
cfg_dir = "./cfgs"
YOLO_CONFIG_DIR = str(Path(cfg_dir).resolve())
# print(YOLO_CONFIG_DIR)
# print('-'*100)
os.environ['YOLO_CONFIG_DIR'] = YOLO_CONFIG_DIR
from ultralytics import YOLO

import glob

from utils.sse import sse_print

from defends import Scale, Compression, NeuralCleanse, PGDPurifier, FGSMDenoise
from torchvision import transforms
from PIL import Image
import torch

# https://docs.ultralytics.com/zh/modes/predict/
def defend_use_adv(args):
    yolo = YOLO(model=args.model_path, task='detect', verbose=True) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    
    event = "model_load"
    data = {
        "status": "success",
        "message": "模型加载完成.",
        "model_name": args.model_name,
        "model_path": args.model_path
    }
    sse_print(event, data)
    
    # ori_images_flod = f"{args.data_path}ori_images"
    adv_images_flod = f"{args.data_path}adv_images"
    # print(ori_images_flod)
    # print(adv_images_flod)
    
    # ori_image_paths = glob.glob(os.path.join(ori_images_flod, '*.jpg'))
    adv_image_paths = glob.glob(os.path.join(adv_images_flod, '*.jpg'))
    # ori_image_paths.sort()
    adv_image_paths.sort()
    # print(len(ori_image_paths))
    # print(len(adv_image_paths))
    
    event = "data_load"
    data = {
        "status": "success",
        "message": "对抗样本和原始样本加载完毕.",
        "data_name": args.data_name,
        "samples_number": len(adv_image_paths),
        "data_path": args.data_path
    }
    sse_print(event, data)
    
    try:
        # print(args.defend_method)
        if args.defend_method == 'scale':
            defend = Scale( # 算法输入图像像素为0~255
                            scale=args.scaling_factor,
                            interp=args.interpolate_method
                            )
        elif args.defend_method == 'compression':
            defend = Compression( # 算法输入图像像素为0~255
                                clip_values=(0, 255),
                                quality=args.image_quality,
                                channels_first=False,
                                apply_fit=True,
                                apply_predict=True,
                                verbose=False,
                            )
        elif args.defend_method == 'neural_cleanse':
            defend = NeuralCleanse(kernel_size=args.filter_kernel_size) # 算法输入图像像素为0~255
        elif args.defend_method == 'pgd_purifier':
            defend = PGDPurifier(
                                steps=args.max_iterations, 
                                alpha=args.step_size, 
                                epsilon=args.epsilon*255 # 算法输入图像像素为0~255
                                )
        elif args.defend_method == 'fgsm_denoise':
            defend = FGSMDenoise(epsilon=args.epsilon*255) # 算法输入图像像素为0~255
        else:
            raise ValueError('不支持的防御方法.')

        event = "defend_init"
        data = {
            "status": "success",
            "message": "防御方法初始化完成.",
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
        import sys
        sys.exit()



    total_iamges = len(adv_image_paths)
    ori_pred_conf_sum = 0.0
    adv_pred_conf_sum = 0.0
    defend_success_count = 0
    defend_failure_count = 0
    
    
    ori_images_flod = f"{args.output_path}/def_images"
    adv_images_flod = f"{args.output_path}/adv_images"
    os.makedirs(ori_images_flod, exist_ok=True)
    os.makedirs(adv_images_flod, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor() # 会将像素值范围变为0~1
    ])
    
    for i in range(total_iamges):
        # print(ori_image_paths[i])
        # print(adv_image_paths[i])
        # ori_results = yolo.predict(source=ori_image_paths[i])
        adv_results = yolo.predict(source=adv_image_paths[i])
        
        adv_image = Image.open(adv_image_paths[i])
        adv_image = transform(adv_image)
        # print(adv_image.shape)
        adv_image = adv_image.unsqueeze(0)
        adv_image = (adv_image * 255).to(torch.uint8)
        adv_image, _ = defend(adv_image)
        # print(adv_image.shape)
        adv_image = adv_image.to(torch.float)
        ori_results = yolo.predict(source=adv_image) # 将防御后的对抗样本作为原始
        
        # print(results)
        # print(len(results))
        
        '''
            Results 对象具有以下属性：
            属性	类型	描述
            orig_img	np.ndarray	原始图像，以 numpy 数组形式呈现。
            orig_shape	tuple	原始图像的形状，格式为（高度，宽度）。
            boxes	Boxes, optional	一个 Boxes 对象，包含检测到的边界框。
            masks	Masks, optional	一个 Masks 对象，包含检测到的掩码。
            probs	Probs, optional	一个 Probs 对象，包含 分类任务 中每个类别的概率。
            keypoints	Keypoints, optional	一个 Keypoints 对象，包含每个对象检测到的关键点。
            obb	OBB, optional	包含旋转框检测的 OBB 对象。
            speed	dict	一个字典，包含预处理、推理和后处理的速度，单位为毫秒/图像。
            names	dict	一个将类索引映射到类名称的字典。
            path	str	图像文件的路径。
            save_dir	str, optional	用于保存结果的目录。
        '''
        # result = results[0]
        # print(result.boxes)  # Boxes object for bounding box outputs
        
        # print(result.boxes.conf)
        # print(result.boxes.cls)
        '''
            Boxes 类的方法和属性表，包括它们的名称、类型和描述：
            名称	类型	描述
            cpu()	方法	将对象移动到 CPU 内存。
            numpy()	方法	将对象转换为 numpy 数组。
            cuda()	方法	将对象移动到 CUDA 内存。
            to()	方法	将对象移动到指定的设备。
            xyxy	属性 (torch.Tensor)	返回 xyxy 格式边界框。
            conf	属性 (torch.Tensor)	返回边界框的置信度值。
            cls	属性 (torch.Tensor)	返回边界框的类别值。
            id	属性 (torch.Tensor)	返回边界框的跟踪 ID（如果可用）。
            xywh	属性 (torch.Tensor)	返回 xywh 格式边界框。
            xyxyn	属性 (torch.Tensor)	返回按原始图像尺寸归一化的 xyxy 格式边界框。
            xywhn	属性 (torch.Tensor)	返回按原始图像尺寸归一化的 xywh 格式边界框。
        '''

        ori_result = ori_results[0]
        adv_result = adv_results[0]
        
        ori_pred_cls = int(ori_result.boxes.cls.item()) if ori_result.boxes.cls.nelement() != 0 else -1
        ori_pred_conf = ori_result.boxes.conf.item() if ori_result.boxes.conf.nelement() != 0 else 0.0
        
        adv_pred_cls = int(adv_result.boxes.cls.item()) if adv_result.boxes.cls.nelement() != 0 else -1
        adv_pred_conf = adv_result.boxes.conf.item() if adv_result.boxes.conf.nelement() != 0 else 0.0

        # print(ori_pred_cls)
        # print(ori_pred_conf)
        # print(adv_pred_cls)
        # print(adv_pred_conf)
        
        actual_class = int(os.path.splitext(os.path.basename(adv_image_paths[i]))[0].split('_')[-1])
        
        if (actual_class == ori_pred_cls and ori_pred_cls != adv_pred_cls) or (actual_class == adv_pred_cls and ori_pred_cls == adv_pred_cls and ori_pred_conf - adv_pred_conf >= args.confidence_threshold):
            # defend_result = '成功'
            defend_result = 'success'
            defend_success_count += 1
        else:
            # defend_result = '失败'
            defend_result = 'failure'
            defend_failure_count += 1
        
        ori_pred_conf_sum += ori_pred_conf
        adv_pred_conf_sum += adv_pred_conf
        
        
        ori_img_path = f"{ori_images_flod}/def_img_{i}_cls_{actual_class}_pred_cls_{ori_pred_cls}.jpg"
        adv_img_path = f"{adv_images_flod}/adv_img_{i}_cls_{actual_class}_pred_cls_{adv_pred_cls}.jpg"
        ori_results[0].save(filename=ori_img_path)
        adv_results[0].save(filename=adv_img_path)
        
        
        event = "defend"
        data = {
            "message": "正在执行防御...",
            "progress": int(i/total_iamges*100),
            # "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击... 原始样本: {ori_img_path}, 原始样本预测准确率: {ori_pred_conf*100:.2f}%, 原始样本预测类别: {ori_pred_cls.item()}, 对抗样本: {adv_img_path}, 对抗样本预测准确率: {adv_pred_conf*100:.2f}%, 对抗样本预测类别: {ori_pred_cls.item()}, 原始样本实际类别: {labels[i].item()}, 攻击结果: {defend_result}"
            "log": f"[{int(i/total_iamges*100)}%] 正在执行防御...",
            "details": {
                "defend_sample": {
                    "confidence": f"{ori_pred_conf*100:.2f}%",
                    "predict_class": ori_pred_cls,
                    "file_path": ori_img_path
                },
                "adversarial_sample": {
                    "confidence": f"{adv_pred_conf*100:.2f}%",
                    "predict_class": adv_pred_cls,
                    "file_path": adv_img_path
                },
                "actual_class": actual_class,
                "defend_result": defend_result
            }
        }
        sse_print(event, data)
        
    
    event = "final_result"
    data = {
        "message": "防御执行完成.",
        "progress": 100,
        "log": f"[100%] 防御执行完成.",
        "details": {
            "defend_samples": ori_images_flod,
            "adversarial_samples": adv_images_flod,
            "summary": {
                "total_iamges": total_iamges,
                "task_success_count": defend_success_count,
                "task_failure_count": defend_failure_count,
                "defend_samples_pred_confidence": f"{ori_pred_conf_sum/total_iamges*100:.2f}%",
                "adversarial_samples_pred_confidence": f"{adv_pred_conf_sum/total_iamges*100:.2f}%"
            }
        }
    }
    sse_print(event, data)
        
