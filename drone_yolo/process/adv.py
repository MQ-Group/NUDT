import os
from pathlib import Path
os.environ['YOLO_VERBOSE'] = 'false'
cfg_dir = "./cfgs"
YOLO_CONFIG_DIR = str(Path(cfg_dir).resolve())
# print(YOLO_CONFIG_DIR)
# print('-'*100)
os.environ['YOLO_CONFIG_DIR'] = YOLO_CONFIG_DIR
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.detect.train import DetectionTrainer

from torchattacks import FGSM, PGD, BIM, CW, DeepFool, GN, Jitter
# import glob

from torchvision import transforms

from utils.sse import sse_print

def adv(args):
    
    overrides = {"session": None}
    yolo = DetectionTrainer(cfg=DEFAULT_CFG, overrides=overrides, _callbacks=None)
    # print(yolo) # <ultralytics.models.yolo.detect.train.DetectionTrainer object at 0x7f9b14537fa0>
    # print(dir(yolo))
    # print(yolo.data) # {'path': PosixPath('input/data/drone_type/drone_type'), 'train': '/data6/user23215430/nudt/drone_recognition/input/data/drone_type/drone_type/images/train', 'val': '/data6/user23215430/nudt/drone_recognition/input/data/drone_type/drone_type/images/val', 'test': None, 'nc': 6, 'names': {0: 'Bebop', 1: 'None', 2: 'None', 3: 'Emax', 4: 'None', 5: 'Mambo'}, 'yaml_file': './cfgs/data.yaml', 'channels': 3}
    
    event = "model_load"
    data = {
        "status": "success",
        "message": "模型加载完成.",
        "model_name": args.model_name,
        "model_path": args.model_path
    }
    sse_print(event, data)
    
    yolo._setup_train()
    
    event = "data_load"
    data = {
        "status": "success",
        "message": "数据集加载完毕.",
        "data_name": args.data_name,
        "data_path": args.data_path
    }
    sse_print(event, data)
    
    try:
        # print(args.attack_method)
        if args.attack_method == 'fgsm':
            atk = FGSM(yolo.model, eps=args.epsilon)
        elif args.attack_method == 'pgd':
            atk = PGD(yolo.model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, random_start=args.random_start)
        elif args.attack_method == 'bim':
            atk = BIM(yolo.model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.attack_method == 'cw':
            atk = CW(yolo.model, c=1, kappa=0, steps=args.max_iterations, lr=args.lr)
        elif args.attack_method == 'deepfool':
            # atk = DeepFool(yolo.model, steps=args.max_iterations, overshoot=0.02)
            atk = CW(yolo.model, c=1, kappa=0, steps=args.max_iterations, lr=args.lr)
        elif args.attack_method == 'gn':
            atk = GN(yolo.model, std=args.std)
        elif args.attack_method == 'jitter':
            # atk = Jitter(yolo.model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, scale=args.scale, std=args.std, random_start=args.random_start)
            atk = PGD(yolo.model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, random_start=args.random_start)
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
    
    total_iamges = args.selected_samples
    event = "adversarial_samples_generation_run"
    for i, batch in enumerate(yolo.test_loader):
        # print(batch.keys()) # dict_keys(['batch_idx', 'bboxes', 'cls', 'im_file', 'img', 'ori_shape', 'ratio_pad', 'resized_shape'])
        # print(batch['img'].shape)
        # print(batch['cls'].shape)
        # print(batch['img'][0,0,0,0])
        batch = yolo.preprocess_batch(batch)
        # loss, loss_items = yolo.model(batch)
        # print(batch['img'][0,0,0,0])
        
        adv_images = atk(batch)
        
        # ori_img_save_path = f"{ori_images_flod}/ori_img_{i}_cls_{yolo.data['names'][int(batch['cls'][0].item())]}.jpg"
        # adv_img_save_path = f"{adv_images_flod}/adv_img_{i}_cls_{yolo.data['names'][int(batch['cls'][0].item())]}.jpg"
        ori_img_save_path = f"{ori_images_flod}/ori_img_{i}_cls_{int(batch['cls'][0].item())}.jpg"
        adv_img_save_path = f"{adv_images_flod}/adv_img_{i}_cls_{int(batch['cls'][0].item())}.jpg"
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(batch['img'][0])
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

    
    
    