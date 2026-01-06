import argparse
import yaml
from easydict import EasyDict
import os
import glob

from utils.sse import sse_input_path_validated, sse_output_path_validated
from utils.yaml_rw import load_yaml, save_yaml

from attacks.adv import adv
from attacks.attack_use_adv import attack_use_adv
from attacks.attack import adv_attack
from defends.defend import defend
from defends.defend_use_adv import defend_use_adv
from detects.detect import detect

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./input', help='input path')
    parser.add_argument('--output_path', type=str, default='./output', help='output path')
    
    parser.add_argument('--process', type=str, default='adv', choices=['adv', 'attack', 'defend', 'detect'], help='process name')
    # parser.add_argument('--process', type=str, default='attack', choices=['adv', 'attack', 'defend', 'train', 'test', 'sample'], help='process name')
    # parser.add_argument('--model', type=str, default='yolov5', choices=['yolov5', 'yolov8', 'yolov10'], help='model name')
    # parser.add_argument('--data', type=str, default='kitti', choices=['kitti', 'bdd100k', 'ua-detrac', 'dawn', 'special_vehicle', 'flir_adas', 'imagenet10'], help='data name')
    # parser.add_argument('--class_number', type=int, default=8, choices=[8, 10, 4, 1, 1000], help='number of class. 8 for kitti, 10 for bdd100k, 4 for ua-detrac, 5 for special_vehicle, 1 for dawn, 1 for flir_adas')
    
    parser.add_argument('--attack_method', type=str, default='fgsm', choices=['fgsm', 'pgd', 'bim', 'cw', 'deepfool', 'gn' 'jitter', 'boundary', 'zoo', 'hsja', 'nes', 'yopo', 'pgdrs', 'trades', 'free', 'fast'], help='attack method')
    parser.add_argument('--defend_method', type=str, default='yopo', choices=['yopo', 'pgdrs', 'trades', 'free', 'fast', 'fgsm', 'pgd', 'bim', 'cw', 'deepfool', 'gn' 'jitter', 'boundary', 'zoo', 'hsja', 'nes'], help='defend method')
    parser.add_argument('--detect_method', type=str, default='spatial_smoothing', choices=['spatial_smoothing', 'feature_squeezing', 'local_intrinsic_dimensionality'], help='detect method')
    
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='device')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers (per RANK if DDP)')
    
    parser.add_argument('--selected_samples', type=int, default=64, help='the number of generated adversarial sample for attack method')
    
    parser.add_argument('--epsilon', type=float, default=8/255, help='epsilon for attack method and defend medthod')
    parser.add_argument('--step_size', type=float, default=2/255, help='epsilon for attack method and defend medthod')
    parser.add_argument('--max_iterations', type=int, default=50, help='epsilon for attack method and defend medthod')
    
    parser.add_argument('--random_start', type=bool, default=True, help='initial random start for attack method')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate of optimization for attack method')
    
    parser.add_argument('--std', type=float, default=0.1, help='standard deviation for gn and jitter attack method')
    parser.add_argument('--scale', type=int, default=10, help='scale for jitter attack method')
    
    parser.add_argument('--max_queries', type=int, default=10, help='max queries for attack method')
    parser.add_argument('--binary_search_steps', type=int, default=10, help='binary search steps for attack method')
    parser.add_argument('--norm', type=str, default='L2', choices=['L2', 'Linf'], help='norm for attack method')

    # defend method
    parser.add_argument('--noise_type', type=str, default='guassian', choices=['guassian', 'uniform'], help='pgdrs parameter for defend method')
    parser.add_argument('--noise_sd', type=float, default=0.5, help='pgdrs parameter noise standard deviation for defend method')
    # detect method
    parser.add_argument('--kernel_size', type=int, default=3, help='ss and fs for detect method')
    parser.add_argument('--bit_depth', type=int, default=4, help='fs for detect method')
    parser.add_argument('--k_nearest', type=int, default=20, help='k_nearest must less than selected_samples lid for detect method')
    parser.add_argument('--detection_threshold', type=float, default=0.5, help='detection threshold for detect method')
    
    
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
    
def add_args(args):
    model_yaml = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*/'), '*.yaml'))[0]
    # print(model_yaml)
    model_name = os.path.splitext(os.path.basename(model_yaml))[0]
    # print(model_name)
    args.model_yaml = model_yaml
    args.model_name = model_name
    if args.process != 'defend':
        model_path = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*/'), '*.pt'))[0]
        # print(model_path)
        args.model_path = model_path
    
    data_yaml = glob.glob(os.path.join(os.path.join(f'{args.input_path}/data', '*/'), '*.yaml'))[0]
    # print(data_yaml)
    data_name = os.path.splitext(os.path.basename(data_yaml))[0]
    # print(data_name)
    args.data_yaml = data_yaml
    args.data_name = data_name
    if args.process == 'attack':
        data_path = glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
    else:
        data_path = glob.glob(os.path.join(os.path.join(f'{args.input_path}/data', '*/'), '*/'))[0]
    args.data_path = data_path
    
    return args

def main(args):
    args = add_args(args)
    if args.process == 'adv':
        adv(args)
    elif args.process == 'attack':
        attack_use_adv(args)
        # attack(args)
    elif args.process == 'defend':
        # defend_use_adv(args)
        defend(args)
    elif args.process == 'detect':
        detect(args)
    else:
        raise ValueError('任务不支持.')
    
if __name__ == '__main__':
    args = parse_args()
    
    sse_input_path_validated(args)
    sse_output_path_validated(args)
    main(args)
    
    
    
    
    
    
