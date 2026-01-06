
# https://docs.ultralytics.com/zh/usage/callbacks/#how-do-i-attach-a-custom-callback-for-the-prediction-mode-in-ultralytics-yolo

import os
from utils.sse import sse_print

# Trainer callbacks ----------------------------------------------------------------------------------------------------

def on_pretrain_routine_start(trainer):
    """Called before the pretraining routine starts."""
    event = "train_initial"
    data = {
        "status": "success",
        "message": "训练任务初始化完成.",
        "parameters": dict(trainer.args)
    }
    sse_print(event, data)


def on_pretrain_routine_end(trainer):
    """Called after the pretraining routine ends."""
    # print(trainer.data) # {'path': PosixPath('input/data/drone_type/drone_type'), 'train': '/data6/user23215430/nudt/drone_recognition/input/data/drone_type/drone_type/images/train', 'val': '/data6/user23215430/nudt/drone_recognition/input/data/drone_type/drone_type/images/val', 'test': None, 'nc': 6, 'names': {0: 'Bebop', 1: 'None', 2: 'None', 3: 'Emax', 4: 'None', 5: 'Mambo'}, 'yaml_file': './cfgs/data.yaml', 'channels': 3}
    event = "data_load"
    data = {
        "status": "success",
        "message": "数据集加载完成.",
        "data_name": trainer.data['names'],
        "data_path": str(trainer.data['path'])
    }
    sse_print(event, data)
    
    if trainer.args.pretrained != False:
        event = "model_load"
        data = {
            "status": "success",
            "message": "模型加载完成.",
            "model_name": os.path.basename(trainer.args.model).split('.')[0],
            "model_path": trainer.args.pretrained
        }
        sse_print(event, data)
    
    
def on_train_start(trainer):
    """Called when the training starts."""
    event = "train_start"
    data = {
        "message": "训练开始.",
        "total_epoch": trainer.args.epochs,
        "total_batch": len(trainer.train_loader),
        "start_epoch": trainer.start_epoch,
        "batch_size": trainer.args.batch,
        "image_size": trainer.args.imgsz,
    }
    sse_print(event, data)


def on_train_epoch_start(trainer):
    """Called at the start of each training epoch."""
    pass


def on_train_batch_start(trainer):
    """Called at the start of each training batch."""
    pass


def optimizer_step(trainer):
    """Called when the optimizer takes a step."""
    pass


def on_before_zero_grad(trainer):
    """Called before the gradients are set to zero."""
    pass


def on_train_batch_end(trainer):
    """Called at the end of each training batch."""
    event = "train"
    data = {
        "message": "正在执行训练...",
        "progress": int(trainer.epoch/trainer.epochs*100),
        "log": f"[{int(trainer.epoch/trainer.epochs*100)}%] 正在执行训练...",
        "details": {
            "epoch": f"{trainer.epoch + 1}/{trainer.epochs}",
            "batch": f"{trainer.batch_i + 1}/{len(trainer.train_loader)}",
            "GPU_memory_util": f"{trainer._get_memory():.3g}G",
            "box_loss": trainer.loss_items[0].item(), 
            "cls_loss": trainer.loss_items[1].item(), 
            "df1_loss": trainer.loss_items[2].item(),
        },
    }
    sse_print(event, data)


def on_train_epoch_end(trainer):
    """Called at the end of each training epoch."""
    event = "train"
    data = {
        "message": "正在执行训练...",
        "progress": int(trainer.epoch/trainer.epochs*100),
        "log": f"[{int(trainer.epoch/trainer.epochs*100)}%] 正在执行训练...",
        "details": {
            "epoch": f"{trainer.epoch + 1}/{trainer.epochs}",
            "GPU_memory_util": f"{trainer._get_memory():.3g}G",
            "box_loss": trainer.loss_items[0].item(), 
            "cls_loss": trainer.loss_items[1].item(), 
            "df1_loss": trainer.loss_items[2].item(),
            "lr": trainer.lr, 
        }
    }
    sse_print(event, data)


def on_fit_epoch_end(trainer):
    """Called at the end of each fit epoch (train + val)."""
    pass


def on_model_save(trainer):
    """Called when the model is saved."""
    event = "train"
    data = {
        "message": "正在执行训练...",
        "progress": int(trainer.epoch/trainer.epochs*100),
        "log": f"[{int(trainer.epoch/trainer.epochs*100)}%] 训练模型权重保存: {trainer.wdir}/last.pt",
    }
    sse_print(event, data)


def on_train_end(trainer):
    """Called when the training ends."""
    event = "final_result"
    data = {
        "message": "训练完成, 结果信息已保存",
        "progress": 100,
        "log": f"[100%] 训练完成, 结果信息已保存",
        "details": {
            "labels_image": f"{trainer.save_dir}/labels.jpg",
            "results_image": f"{trainer.save_dir}/results.png",
            "results_csv": f"{trainer.save_dir}/results.csv",
            "display_images": [f"{trainer.save_dir}/train_batch{i}.png" for i in trainer.plot_idx],
            "weight": {
                "last": f"{trainer.wdir}/last.pt",
                "best": f"{trainer.wdir}/best.pt",
            },
            "summary": trainer.metrics,
        }
    }
    sse_print(event, data)


def on_params_update(trainer):
    """Called when the model parameters are updated."""
    pass


def teardown(trainer):
    """Called during the teardown of the training process."""
    pass

# Validator callbacks --------------------------------------------------------------------------------------------------


def on_val_start(validator):
    """Called when the validation starts."""
    event = "val_initial"
    data = {
        "status": "success",
        "message": "测试任务初始化完成.",
        "parameters": dict(validator.args)
    }
    sse_print(event, data)
    
    # print(trainer.data) # {'path': PosixPath('input/data/drone_type/drone_type'), 'train': '/data6/user23215430/nudt/drone_recognition/input/data/drone_type/drone_type/images/train', 'val': '/data6/user23215430/nudt/drone_recognition/input/data/drone_type/drone_type/images/val', 'test': None, 'nc': 6, 'names': {0: 'Bebop', 1: 'None', 2: 'None', 3: 'Emax', 4: 'None', 5: 'Mambo'}, 'yaml_file': './cfgs/data.yaml', 'channels': 3}
    event = "data_load"
    data = {
        "status": "success",
        "message": "数据集加载完成.",
        "data_name": validator.data['names'],
        "data_path": str(validator.data['path'])
    }
    sse_print(event, data)
    
    event = "model_load"
    data = {
        "status": "success",
        "message": "模型加载完成.",
        "model_name": os.path.basename(validator.args.model).split('.')[0],
        "model_path": validator.args.pretrained
    }
    sse_print(event, data)
    
    event = "val_start"
    data = {
        "message": "测试开始.",
        "total_batch": len(validator.dataloader),
        "batch_size": validator.args.batch,
        "image_size": validator.args.imgsz
    }
    sse_print(event, data)


def on_val_batch_start(validator):
    """Called at the start of each validation batch."""
    pass


def on_val_batch_end(validator):
    """Called at the end of each validation batch."""
    # print(validator.args)
    event = "test"
    data = {
        "message": "正在执行测试...",
        "progress": int(validator.batch_i/len(validator.dataloader)*100),
        "log": f"[{int(validator.batch_i/len(validator.dataloader)*100)}%] 正在执行测试...",
        "details": {
            "batch": f"{validator.batch_i + 1}/{len(validator.dataloader)}",
        },
    }
    sse_print(event, data)


def on_val_end(validator):
    """Called when the validation ends."""
    event = "final_result"
    data = {
        "message": "测试执行完成, 结果信息已保存",
        "progress": 100,
        "log": f"[100%] 测试执行完成, 结果信息已保存",
        "details": {
            "curve": {
                "混淆矩阵": f"{validator.save_dir}/confusion_matrix.png",
                "归一化混淆矩阵": f"{validator.save_dir}/confusion_matrix_normalized.png",
                "F1-置信度曲线": f"{validator.save_dir}/BoxF1_curve.png",
                "精度-置信度曲线": f"{validator.save_dir}/BoxP_curve.png",
                "精度-回调曲线": f"{validator.save_dir}/BoxPR_curve.png",
                "回调-置信度曲线": f"{validator.save_dir}/BoxR_curve.png"
            },
            "speed/s": validator.metrics.speed,
            "summary": {
                f"{validator.data['names'][i]}": class_summary for i, class_summary in enumerate(validator.metrics.summary())
            }
        }
    }
    sse_print(event, data)


# Predictor callbacks --------------------------------------------------------------------------------------------------


def on_predict_start(predictor):
    """Called when the prediction starts."""
    pass

def on_predict_batch_start(predictor):
    """Called at the start of each prediction batch."""
    pass
    

def on_predict_batch_end(predictor):
    """Called at the end of each prediction batch."""
    pass


def on_predict_postprocess_end(predictor):
    """Called after the post-processing of the prediction ends."""
    pass


def on_predict_end(predictor):
    """Called when the prediction ends."""
    # event = "task_completed"
    # data = {
    #     "status": "success",
    #     "message": "Task completed successfully.",
    #     "summary": predictor.results
    # }
    # sse_print(event, data)
    pass
    


# Exporter callbacks ---------------------------------------------------------------------------------------------------


def on_export_start(exporter):
    """Called when the model export starts."""
    pass


def on_export_end(exporter):
    """Called when the model export ends."""
    pass



callbacks_dict = {
    # Run in trainer
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_pretrain_routine_end": on_pretrain_routine_end,
    "on_train_start": on_train_start,
    "on_train_epoch_start": on_train_epoch_start,
    "on_train_batch_start": on_train_batch_start,
    "optimizer_step": optimizer_step,
    "on_before_zero_grad": on_before_zero_grad,
    "on_train_batch_end": on_train_batch_end,
    "on_train_epoch_end": on_train_epoch_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_model_save": on_model_save,
    "on_train_end": on_train_end,
    "on_params_update": on_params_update,
    "teardown": teardown,
    # Run in validator
    "on_val_start": on_val_start,
    "on_val_batch_start": on_val_batch_start,
    "on_val_batch_end": on_val_batch_end,
    "on_val_end": on_val_end,
    # Run in predictor
    "on_predict_start": on_predict_start,
    "on_predict_batch_start": on_predict_batch_start,
    "on_predict_postprocess_end": on_predict_postprocess_end,
    "on_predict_batch_end": on_predict_batch_end,
    "on_predict_end": on_predict_end,
    # Run in exporter
    "on_export_start": on_export_start,
    "on_export_end": on_export_end,
}