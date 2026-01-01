import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc

import argparse

import torch
import yaml
from easydict import EasyDict
import os
import glob
import zipfile
import json

from nudt_pytorch_lightning.models.model_interface import MInterface
from nudt_pytorch_lightning.datasets.data_interface import DInterface
from nudt_pytorch_lightning.callbacks.callback import Callback

# os.environ["NCCL_DEBUG"] = "INFO"

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfgs/badnet/BadNet_cifar10_poison.yaml', help='the config_file')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)



def main(args):
    
    data = DInterface(args)
    model = MInterface(**vars(args))
    
    
    if getattr(args, 'train', False):
        trainer = pl.Trainer(gpus=-1, 
                            accelerator='ddp', 
                            profiler='simple',
                            max_epochs=args.epoch, 
                            resume_from_checkpoint=args.checkpoint,
                            callbacks=[Callback], 
                            default_root_dir=args.log_dir,
                            check_val_every_n_epoch=1, 
                            num_sanity_val_steps=2,
                            gradient_clip_val=0.0)
        trainer.fit(model=model, datamodule=data)
    elif getattr(args, 'test', False):
        trainer = pl.Trainer(gpus=-1, 
                            accelerator='ddp', 
                            profiler='simple',
                            default_root_dir=args.log_dir,
                            resume_from_checkpoint=args.checkpoint)
        results = trainer.test(model=model, datamodule=data)
    elif getattr(args, 'validate', False):
        trainer = pl.Trainer(gpus=-1, 
                            accelerator='ddp', 
                            profiler='simple',
                            default_root_dir=args.log_dir,
                            resume_from_checkpoint=args.checkpoint)
        results = trainer.validate(model=model, datamodule=data)
    elif getattr(args, 'predict', False):
        trainer = pl.Trainer(gpus=-1, 
                            accelerator='ddp', 
                            profiler='simple',
                            default_root_dir=args.log_dir,
                            resume_from_checkpoint=args.checkpoint)
        results = trainer.predict(model=model, datamodule=data)
    elif getattr(args, 'tune', False):
        trainer = pl.Trainer(gpus=-1, 
                            accelerator='ddp', 
                            profiler='simple',
                            max_epochs=args.epoch, 
                            resume_from_checkpoint=args.checkpoint,
                            callbacks=[Callback], 
                            default_root_dir=args.log_dir,
                            check_val_every_n_epoch=1, 
                            num_sanity_val_steps=2,
                            gradient_clip_val=0.0)
        results = trainer.tune(model=model, datamodule=data)
    else:
        trainer = pl.Trainer(gpus=-1, 
                            accelerator='ddp', 
                            profiler='simple',
                            default_root_dir=args.log_dir,
                            resume_from_checkpoint=args.checkpoint)
        results = trainer.test(model=model, datamodule=data)


if __name__ == '__main__':
    args = parse_config()
    
    env_cp = os.environ.copy()
    try:
        node_rank, local_rank, world_size = env_cp['NODE_RANK'], env_cp['LOCAL_RANK'], env_cp['WORLD_SIZE']

        is_in_ddp_subprocess = env_cp['PL_IN_DDP_SUBPROCESS']
        pl_trainer_gpus = env_cp['PL_TRAINER_GPUS']
        print(node_rank, local_rank, world_size, is_in_ddp_subprocess, pl_trainer_gpus)

        if int(local_rank) == int(world_size) - 1:
            print(args)
    except KeyError:
        pass
    
    
    main(args)
    
    