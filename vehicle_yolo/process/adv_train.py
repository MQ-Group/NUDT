import os
from pathlib import Path
os.environ['YOLO_VERBOSE'] = 'false'
cfg_dir = "./cfgs"
YOLO_CONFIG_DIR = str(Path(cfg_dir).resolve())
# print(YOLO_CONFIG_DIR)
# print('-'*100)
os.environ['YOLO_CONFIG_DIR'] = YOLO_CONFIG_DIR
from ultralytics import YOLO
from process.callbacks import callbacks_dict

# https://docs.ultralytics.com/zh/modes/train/#usage-examples
def adv_train(args):
    yolo = YOLO(model=args.model_yaml, task='detect', verbose=True)
    
    for (event, func) in callbacks_dict.items():
        yolo.add_callback(event, func)
        
    # 修改ultralytics/engine/trainer.py
    results = yolo.train(cfg=args.cfg_yaml, attack_method=args.attack_method, defend_method=args.defend_method, adversarial_sample_proportion=args.adversarial_sample_proportion)
    
        