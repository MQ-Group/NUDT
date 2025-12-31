import os
from pathlib import Path
os.environ['YOLO_VERBOSE'] = 'false'
cfg_dir = "./cfgs"
YOLO_CONFIG_DIR = str(Path(cfg_dir).resolve())
# print(YOLO_CONFIG_DIR)
# print('-'*100)
os.environ['YOLO_CONFIG_DIR'] = YOLO_CONFIG_DIR

import glob
import yaml
from easydict import EasyDict

from nudt_ultralytics.callbacks.callbacks import callbacks_dict


def load_yaml(load_path):
    with open(load_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
    
def main(args):
    from ultralytics import YOLO
    
    cfg = load_yaml(args.cfg_yaml)
    cfg = EasyDict(cfg)
    
    # if cfg.mode == 'train':
    #     if cfg.pretrained is None:
    #         model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     else:
    #         model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.train(cfg=args.cfg_yaml)
    # if cfg.mode == 'train':
    #     if cfg.pretrained is None:
    #         model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     else:
    #         model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.train(cfg=args.cfg_yaml)
    # if cfg.mode == 'train':
    #     model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     if cfg.pretrained is None:
    #         results = model.train(cfg=args.cfg_yaml)
    #     else:
    #         results = model.train(cfg=args.cfg_yaml, pretrained=cfg.pretrained)
    # elif cfg.mode == 'validate':
    #     model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.val(data=cfg.data, imgsz=cfg.imgsz, batch=cfg.batch, conf=cfg.conf, iou=cfg.iou, device=cfg.device, project=cfg.project, name=cfg.name)
    # elif cfg.mode == 'predict':
    #     model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.predict(source="path/to/image.jpg", conf=cfg.conf)
    # elif cfg.mode == 'tune':
    #     model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.tune(data=cfg.data, iterations=5)
    
    if args.process == 'train':
        model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        # print(model)
        # param_num = sum(p.numel() for p in model.parameters())
        # print(f'Number of Parameters: {param_num}')
        for (event, func) in callbacks_dict.items():
            model.add_callback(event, func)
        results = model.train(cfg=args.cfg_yaml)
    # elif args.process == 'defend':
    #     if cfg.pretrained is None:
    #         model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     else:
    #         model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.train(cfg=args.cfg_yaml)
    elif args.process == 'defend':
        model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        for (event, func) in callbacks_dict.items():
            model.add_callback(event, func)
        results = model.val(data=cfg.data)
    elif args.process == 'attack':
        model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        for (event, func) in callbacks_dict.items():
            model.add_callback(event, func)
        results = model.val(data=cfg.data)
    elif args.process == 'adv':
        from attacks.attacks import attacks
        att = attacks(args)
        att.run_adv()
        import glob
        # ori_dataset_name = glob.glob(os.path.join(f'{args.input_path}/data', '*'))[0].split('/')[-1]
        # adv_data_path = f'{args.output_path}/adv_{ori_dataset_name}'
        # os.system(f"cp {args.data_yaml} {adv_data_path}")
        os.system(f"cp {args.data_yaml} {args.output_path}")
        from utils.sse import sse_print
        event = "final_result"
        data = {
            "message": "对抗样本生成完毕.",
            "progress": 100,
            # "log": f"[100%] 对抗样本保存在{args.output_path}/adv_{ori_dataset_name}, 包含{args.selected_samples}个对抗样本."
            "log": f"[100%] 对抗样本保存在{args.output_path}/{args.data_name}, 包含{args.selected_samples}个对抗样本."
        }
        sse_print(event, data)
    elif args.process == 'test':
        model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        for (event, func) in callbacks_dict.items():
            model.add_callback(event, func)
        results = model.val(data=cfg.data)
    elif args.process == 'sample':
        from sample.sample import sample_dataset
        sampled_data_path = f'{args.output_path}/sampled_' + args.data_name
        sample_dataset(source_dir=args.data_path, target_dir=sampled_data_path, train_count=args.selected_samples, val_count=args.selected_samples, seed=None)
        os.system(f"cp {args.data_yaml} {args.output_path}")
        
        from utils.sse import sse_print
        event = "final_result"
        data = {
            "message": "数据集抽取完毕.",
            "progress": 100,
            "log": f"[100%] 从{args.data_path}数据集中抽取{args.selected_samples}张样本生成小数据集保存在{sampled_data_path}"
        }
        sse_print(event, data)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfgs/yolo.yaml', help='cfg yaml file')

    args = parser.parse_args()
    return EasyDict(vars(args))

if __name__ == '__main__':
    args = parse_args()
    main(args)