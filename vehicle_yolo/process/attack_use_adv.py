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

# https://docs.ultralytics.com/zh/modes/predict/
def attack_use_adv(args):
    yolo = YOLO(model=args.model_path, task='detect', verbose=True) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    # print(yolo)
    # print(dir(yolo))
    # print(dir(yolo.model))
    # print(yolo.model.names)
    
    event = "model_load"
    data = {
        "status": "success",
        "message": "模型加载完成.",
        "model_name": args.model_name,
        "model_path": args.model_path
    }
    sse_print(event, data)
    
    ori_images_flod = f"{args.data_path}ori_images"
    adv_images_flod = f"{args.data_path}adv_images"
    # print(ori_images_flod)
    # print(adv_images_flod)
    
    ori_image_paths = glob.glob(os.path.join(ori_images_flod, '*.jpg'))
    adv_image_paths = glob.glob(os.path.join(adv_images_flod, '*.jpg'))
    # print(len(ori_image_paths))
    # print(len(adv_image_paths))
    
    event = "data_load"
    data = {
        "status": "success",
        "message": "对抗样本和原始样本加载完毕.",
        "data_name": args.data_name,
        "samples_number": len(ori_image_paths),
        "data_path": args.data_path
    }
    sse_print(event, data)

    total_iamges = len(ori_image_paths)
    # ori_pred_conf_sum = 0.0
    # adv_pred_conf_sum = 0.0
    attack_success_count = 0
    attack_failure_count = 0
    
    
    ori_images_flod = f"{args.output_path}/ori_images"
    adv_images_flod = f"{args.output_path}/adv_images"
    os.makedirs(ori_images_flod, exist_ok=True)
    os.makedirs(adv_images_flod, exist_ok=True)
    
    for i in range(total_iamges):
        # print(ori_image_paths[i])
        # print(adv_image_paths[i])
        ori_results = yolo.predict(source=ori_image_paths[i])
        adv_results = yolo.predict(source=adv_image_paths[i])
        
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
        # print(ori_result.boxes.cls)
        # print(ori_result.boxes.conf)
        # print(adv_result.boxes.cls)
        # print(adv_result.boxes.conf)
        
        # ori_pred_cls = int(ori_result.boxes.cls.item()) if ori_result.boxes.cls.nelement() != 0 else -1
        # ori_pred_conf = ori_result.boxes.conf.item() if ori_result.boxes.conf.nelement() != 0 else 0.0
        ori_pred_cls = ori_result.boxes.cls
        ori_pred_conf = ori_result.boxes.conf
        
        # adv_pred_cls = int(adv_result.boxes.cls.item()) if adv_result.boxes.cls.nelement() != 0 else -1
        # adv_pred_conf = adv_result.boxes.conf.item() if adv_result.boxes.conf.nelement() != 0 else 0.0
        adv_pred_cls = adv_result.boxes.cls
        adv_pred_conf = adv_result.boxes.conf

        # print(ori_pred_cls)
        # print(ori_pred_conf)
        # print(adv_pred_cls)
        # print(adv_pred_conf)
        
        
        # if ori_pred_cls != adv_pred_cls or (ori_pred_cls == adv_pred_cls and ori_pred_conf - adv_pred_conf >= args.confidence_threshold):
        #     # 1）如果识别类别不同
        #     # 2）如果识别类别相同，但有置信度降低大于阈值
        #     # attack_result = '成功'
        #     attack_result = 'success'
        #     attack_success_count += 1
        # esle:
        #     # attack_result = '失败'
        #     attack_result = 'failure'
        #     attack_failure_count += 1
            
        if ori_pred_cls.nelement() == adv_pred_cls.nelement() and ori_pred_cls.nelement() != 0 and (ori_pred_cls == adv_pred_cls).all() and (ori_pred_conf - adv_pred_conf < args.confidence_threshold).all(): # tensor.all()功能: 如果张量tensor中所有元素都是True, 才返回True; 否则返回False
            # attack_result = '失败'
            attack_result = 'failure'
            attack_failure_count += 1
        else:
            # attack_result = '成功'
            attack_result = 'success'
            attack_success_count += 1
        
        
        # ori_pred_conf_sum += ori_pred_conf
        # adv_pred_conf_sum += adv_pred_conf
            
        # actual_class = int(os.path.splitext(os.path.basename(ori_image_paths[i]))[0].split('_')[-1])
        # ori_img_path = f"{ori_images_flod}/ori_img_{i}_cls_{actual_class}_pred_cls_{ori_pred_cls}.jpg"
        # adv_img_path = f"{adv_images_flod}/adv_img_{i}_cls_{actual_class}_pred_cls_{adv_pred_cls}.jpg"
        actual_object_number = int(os.path.splitext(os.path.basename(ori_image_paths[i]))[0].split('_')[-1])
        ori_img_path = f"{ori_images_flod}/ori_img_{i}_obj_{actual_object_number}_pred_obj_{ori_pred_cls.nelement()}.jpg"
        adv_img_path = f"{adv_images_flod}/adv_img_{i}_obj_{actual_object_number}_pred_obj_{adv_pred_cls.nelement()}.jpg"
        
        ori_results[0].save(filename=ori_img_path)
        adv_results[0].save(filename=adv_img_path)
        
        
        event = "attack"
        data = {
            "message": "正在执行攻击...",
            "progress": int(i/total_iamges*100),
            # "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击推理... 原始样本: {ori_img_path}, 原始样本预测准确率: {ori_pred_conf*100:.2f}%, 原始样本预测类别: {ori_pred_cls.item()}, 对抗样本: {adv_img_path}, 对抗样本预测准确率: {adv_pred_conf*100:.2f}%, 对抗样本预测类别: {ori_pred_cls.item()}, 原始样本实际类别: {labels[i].item()}, 攻击结果: {attack_result}"
            "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击...",
            "details": {
                "original_sample": {
                    # "confidence": f"{ori_pred_conf*100:.2f}%",
                    # "predict_class": ori_pred_cls,
                    "object_number": ori_pred_cls.nelement(),
                    "predict_class": [yolo.model.names[int(ori_pred_cls_i)] for ori_pred_cls_i in ori_pred_cls.tolist()],
                    "confidence": ori_pred_conf.tolist(),
                    "file_path": ori_img_path
                },
                "adversarial_sample": {
                    # "confidence": f"{adv_pred_conf*100:.2f}%",
                    # "predict_class": adv_pred_cls,
                    "object_number": adv_pred_cls.nelement(),
                    "predict_class": [yolo.model.names[int(adv_pred_cls_i)] for adv_pred_cls_i in adv_pred_cls.tolist()],
                    "confidence": adv_pred_conf.tolist(),
                    "file_path": adv_img_path
                },
                # "actual_class": actual_class,
                "actual_object_number": actual_object_number,
                "attack_result": attack_result
            }
        }
        sse_print(event, data)
        
    
    event = "final_result"
    data = {
        "message": "攻击执行完成.",
        "progress": 100,
        "log": f"[100%] 攻击执行完成.",
        "details": {
            "original_samples": ori_images_flod,
            "adversarial_samples": adv_images_flod,
            "summary": {
                "total_iamges": total_iamges,
                "task_success_count": attack_success_count,
                "task_failure_count": attack_failure_count,
                # "original_samples_pred_confidence": f"{ori_pred_conf_sum/total_iamges*100:.2f}%",
                # "adversarial_samples_pred_confidence": f"{adv_pred_conf_sum/total_iamges*100:.2f}%"
            }
        }
    }
    sse_print(event, data)
        
        
    
    