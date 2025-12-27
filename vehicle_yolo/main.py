import argparse
import yaml
from easydict import EasyDict
import os
import glob

from utils.sse import sse_input_path_validated, sse_output_path_validated
from utils.yaml_rw import load_yaml, save_yaml
from nudt_ultralytics.main import main as yolo

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./input', help='input path')
    parser.add_argument('--output_path', type=str, default='./output', help='output path')
    
    parser.add_argument('--process', type=str, default='defend', choices=['adv', 'attack', 'defend', 'train', 'test', 'sample'], help='process name')
    # parser.add_argument('--model', type=str, default='yolov5', choices=['yolov5', 'yolov8', 'yolov10'], help='model name')
    # parser.add_argument('--data', type=str, default='kitti', choices=['kitti', 'bdd100k', 'ua-detrac', 'dawn', 'special_vehicle', 'flir_adas', 'imagenet10'], help='data name')
    # parser.add_argument('--class_number', type=int, default=8, choices=[8, 10, 4, 1, 1000], help='number of class. 8 for kitti, 10 for bdd100k, 4 for ua-detrac, 5 for special_vehicle, 1 for dawn, 1 for flir_adas')
    
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'classify'], help='task name. detect for kitti, bdd100k, ua-detrac, special_vehicle, dawn, flir_adas')
    
    parser.add_argument('--attack_method', type=str, default='fgsm', choices=['pgd', 'fgsm', 'bim', 'deepfool', 'cw'], help='attack method')
    parser.add_argument('--defend_method', type=str, default='scale', choices=['scale', 'compression', 'fgsm_denoise', 'neural_cleanse', 'pgd_purifier'], help='defend method')
    
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    # parser.add_argument('--device', type=int, default=0, help='which gpu for cuda')
    parser.add_argument('--device', type=str, default='cpu', help='which gpu for cuda')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers (per RANK if DDP)')
    
    parser.add_argument('--selected_samples', type=int, default=10, help='the number of generated adversarial sample for attack method')
    parser.add_argument('--epsilon', type=float, default=8/255, help='epsilon for attack method')
    parser.add_argument('--step_size', type=float, default=2/255, help='epsilon for attack method')
    parser.add_argument('--max_iterations', type=int, default=50, help='epsilon for attack method')
    parser.add_argument('--random_start', type=bool, default=False, help='initial random start for attack method')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate of optimization for attack method')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=['cross_entropy', 'mse', 'l1', 'binary_cross_entropy'], help='loss function for attack method')
    parser.add_argument('--optimization_method', type=str, default='adam', choices=['adam', 'sgd'], help='optimization for attack method')
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_environ = {}
    for key, value in args_dict.items():
        if key in ['input_path', 'output_path']:
            args_dict_environ[key] = type_switch(os.getenv(key.upper(), value), value)
        else:
            args_dict_environ[key] = type_switch(os.getenv(key, value), value)
    args_easydict = EasyDict(args_dict_environ)
    return args_easydict
    
def type_switch(environ_value, value):
    if isinstance(value, bool):
        return bool(environ_value)
    elif isinstance(value, int):
        return int(environ_value)
    elif isinstance(value, float):
        return float(environ_value)
    elif isinstance(value, str):
        return environ_value
    
def yolo_cfg(args):
    model_yaml = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*'), '*.yaml'))[0]
    # print(model_yaml)
    model_name = os.path.splitext(os.path.basename(model_yaml))[0]
    # print(model_name)
    args.model_yaml = model_yaml
    args.model_name = model_name
    data_yaml = glob.glob(os.path.join(os.path.join(f'{args.input_path}/data', '*'), '*.yaml'))[0]
    # print(data_yaml)
    data_name = os.path.splitext(os.path.basename(data_yaml))[0]
    # print(data_name)
    args.data_yaml = data_yaml
    args.data_name = data_name
    
    model_cfg = load_yaml(model_yaml)
    data_cfg = load_yaml(data_yaml)
    
    model_cfg['nc'] = data_cfg['nc']
    data_path = glob.glob(os.path.join(os.path.join(f'{args.input_path}/data', '*'), '*'))[0]
    # print(data_path)
    data_cfg['path'] = data_path
    
    args.nc = data_cfg['nc']
    args.data_path = data_path
    
    model_yaml = './cfgs/model.yaml'
    save_yaml(model_cfg, model_yaml)
    data_yaml = './cfgs/data.yaml'
    save_yaml(data_cfg, data_yaml)
    
    cfg_yaml = f'./nudt_ultralytics/cfgs/models/{args.task}/default.yaml'
    cfg = load_yaml(cfg_yaml)
    cfg = EasyDict(cfg)
    cfg.task = args.task
    cfg.model = model_yaml
    if args.task == 'classify':
        cfg.data = data_cfg['path']
    else:
        cfg.data = data_yaml
    cfg.save_dir = args.output_path
    if args.process == 'adv':
        cfg.mode = 'predict'
        cfg.batch = 1
        cfg.pretrained = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*'), '*.pt'))[0]
        cfg.device = args.device
        cfg.workers = args.workers
    elif args.process == 'attack':
        cfg.mode = 'validate'
        cfg.batch = args.batch
        cfg.pretrained = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*'), '*.pt'))[0]
        cfg.device = args.device
        cfg.workers = args.workers
        cfg.attack_or_defend = 'attack'
        cfg.attack_method = args.attack_method
    elif args.process == 'defend':
        cfg.mode = 'predict'
        # cfg.batch = 1
        cfg.pretrained = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*'), '*.pt'))[0]
        cfg.attack_or_defend = 'defend'
        cfg.defend_method = args.defend_method
        cfg.workers = args.workers
    elif args.process == 'train':
        cfg.mode = 'train'
        cfg.epochs = args.epochs
        cfg.batch = args.batch
        cfg.device = args.device
        cfg.workers = args.workers
    elif args.process == 'test':
        cfg.mode = 'validate'
        cfg.batch = args.batch
        cfg.pretrained = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*'), '*.pt'))[0]
        cfg.device = args.device
        cfg.workers = args.workers
        
    
    cfg = dict(cfg)
    args.cfg_yaml = './cfgs/default.yaml'
    save_yaml(cfg, args.cfg_yaml)
    
    return args

def main(args):
    args = yolo_cfg(args)
    yolo(args)
        
if __name__ == '__main__':
    args = parse_args()
    
    sse_input_path_validated(args)
    sse_output_path_validated(args)
    # sse_working_path_created(args.working_path)
    # sse_source_unzip_completed(args.dataset_path, args.working_path)
    main(args)
    
    
    
    
    
    
