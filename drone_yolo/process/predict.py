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
def predict(args):
    yolo = YOLO(model=args.model_path, task='detect', verbose=True) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    # print(yolo)
    
    event = "model_load"
    data = {
        "status": "success",
        "message": "模型加载完成.",
        "model_name": args.model_name,
        "model_path": args.model_path
    }
    sse_print(event, data)
    
    ori_images_flod = f"{args.data_path}ori_images"
    # print(ori_images_flod)
    
    ori_image_paths = glob.glob(os.path.join(ori_images_flod, '*.jpg'))
    ori_image_paths.sort()
    # print(len(ori_image_paths))
    
    event = "data_load"
    data = {
        "status": "success",
        "message": "样本加载完毕.",
        "data_name": args.data_name,
        "samples_number": len(ori_image_paths),
        "data_path": args.data_path
    }
    sse_print(event, data)

    total_iamges = len(ori_image_paths)
    ori_pred_conf_sum = 0.0
    
    
    ori_images_flod = f"{args.output_path}/ori_images"
    os.makedirs(ori_images_flod, exist_ok=True)
    
    for i in range(total_iamges):
        # print(ori_image_paths[i])
        ori_results = yolo.predict(source=ori_image_paths[i])
        
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
        
        ori_pred_cls = int(ori_result.boxes.cls.item()) if ori_result.boxes.cls.nelement() != 0 else -1
        ori_pred_conf = ori_result.boxes.conf.item() if ori_result.boxes.conf.nelement() != 0 else 0.0
        # print(ori_pred_cls)
        # print(ori_pred_conf)
        
        ori_pred_conf_sum += ori_pred_conf
        
        ori_img_path = f"{ori_images_flod}/ori_img_{i}_pred_cls_{ori_pred_cls}.jpg"
        ori_results[0].save(filename=ori_img_path)
        
        event = "predict"
        data = {
            "message": "正在执行预测...",
            "progress": int(i/total_iamges*100),
            # "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击... 原始样本: {ori_img_path}, 原始样本预测准确率: {ori_pred_conf*100:.2f}%, 原始样本预测类别: {ori_pred_cls.item()}, 对抗样本: {adv_img_path}, 对抗样本预测准确率: {adv_pred_conf*100:.2f}%, 对抗样本预测类别: {ori_pred_cls.item()}, 原始样本实际类别: {labels[i].item()}, 攻击结果: {attack_result}"
            "log": f"[{int(i/total_iamges*100)}%] 正在执行预测...",
            "details": {
                "confidence": f"{ori_pred_conf*100:.2f}%",
                "predict_class": ori_pred_cls,
                "file_path": ori_img_path
            }
        }
        sse_print(event, data)
        
    
    event = "final_result"
    data = {
        "message": "预测执行完成.",
        "progress": 100,
        "log": f"[100%] 预测执行完成.",
        "details": {
            "samples": ori_images_flod,
            "summary": {
                "total_iamges": total_iamges,
                "samples_pred_confidence": f"{ori_pred_conf_sum/total_iamges*100:.2f}%"
            }
        }
    }
    sse_print(event, data)
        
        
    
    