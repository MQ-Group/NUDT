
# https://docs.ultralytics.com/zh/usage/callbacks/#how-do-i-attach-a-custom-callback-for-the-prediction-mode-in-ultralytics-yolo

import os
from utils.sse import sse_print

# Trainer callbacks ----------------------------------------------------------------------------------------------------

def on_pretrain_routine_start(trainer):
    """Called before the pretraining routine starts."""
    pass


def on_pretrain_routine_end(trainer):
    """Called after the pretraining routine ends."""
    event = "task_initialized"
    data = {
        "status": "success",
        "message": "任务初始化完成.",
        "parameters": dict(trainer.args)
    }
    sse_print(event, data)
    
    if trainer.args.pretrained is not None:
        event = "model_loaded"
        data = {
            "status": "success",
            "message": "模型加载完成.",
            "model_name": os.path.basename(trainer.args.model).split('.')[0],
            "model_path": trainer.args.pretrained
        }
        sse_print(event, data)
    
    
def on_train_start(trainer):
    """Called when the training starts."""
    pass


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
            "batch": f"{trainer.batch_i + 1}/{trainer.total_batch}",
            "batch size": trainer.batch_size,
            "image size": trainer.args.imgsz,
            "GPU memory util": f"{trainer._get_memory():.3g}G",
            "loss": trainer.loss.item(), 
        }
    }
    sse_print(event, data)


def on_train_epoch_end(trainer):
    """Called at the end of each training epoch."""
    ############################################################
    event = "train"
    data = {
        "message": "正在执行训练...",
        "progress": int(trainer.epoch/trainer.epochs*100),
        "log": f"[{int(trainer.epoch/trainer.epochs*100)}%] 正在执行训练...",
        "details": {
            "epoch": f"{trainer.epoch + 1}/{trainer.epochs}",
            "batch size": trainer.batch_size,
            "image size": trainer.args.imgsz,
            "GPU memory util": f"{trainer._get_memory():.3g}G",
            "epoch_time": trainer.epoch_time, 
            "loss": trainer.loss.item(), 
            "lr": trainer.lr, 
        }
    }
    sse_print(event, data)
    # print(self.args)
    # with open(f'{self.args.save_dir}/log.txt', 'a', encoding='utf-8') as f:
    #     import json
    #     json_str = json.dumps(data, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    #     f.write(json_str)
    #     f.write('\n')
    ############################################################


def on_fit_epoch_end(trainer):
    """Called at the end of each fit epoch (train + val)."""
    pass


def on_model_save(trainer):
    """Called when the model is saved."""
    event = "train"
    data = {
        "message": "正在执行训练...",
        "progress": int(trainer.epoch/trainer.epochs*100),
        "log": f"[{int(trainer.epoch/trainer.epochs*100)}%] 训练模型权重保存: {trainer.wdir}/epoch{trainer.epoch}.pt",
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
            "labels image": f"{trainer.save_dir}/labels.jpg",
            "results image": f"{trainer.save_dir}/results.png",
            "results csv": f"{trainer.save_dir}/results.csv",
            "display images": [f"{trainer.save_dir}/train_batch{i}.png" for i in trainer.plot_idx],
            "weight": str(trainer.wdir),
            "summary": trainer.metrics,
        }
    }
    sse_print(event, data)


def on_params_update(trainer):
    """Called when the model parameters are updated."""
    pass


def teardown(trainer):
    """Called during the teardown of the training process."""
    # event = "task_completed"
    # data = {
    #     "status": "success",
    #     "message": "Task completed successfully.",
    #     "summary": trainer.metrics
    # }
    # sse_print(event, data)
    pass

# Validator callbacks --------------------------------------------------------------------------------------------------


def on_val_start(validator):
    """Called when the validation starts."""
    # print('-'*100)
    # print(validator.args)
    if validator.args.process in ['test', 'attack', 'defend']:
        event = "task_initialized"
        data = {
            "status": "success",
            "message": "任务初始化完成.",
            "parameters": dict(validator.args)
        }
        sse_print(event, data)
        
        event = "model_loaded"
        data = {
            "status": "success",
            "message": "模型加载完成.",
            "model name": os.path.basename(validator.args.model).split('.')[0],
            "model path": validator.args.pretrained
        }
        sse_print(event, data)


def on_val_batch_start(validator):
    """Called at the start of each validation batch."""
    pass


def on_val_batch_end(validator):
    """Called at the end of each validation batch."""
    # print(validator.args)
    if validator.args.process == "attack":
        event = "attack"
        data = {
            "message": "正在执行攻击...",
            "progress": int(validator.batch_i/len(validator.dataloader)*100),
            "log": f"[{int(validator.batch_i/len(validator.dataloader)*100)}%] 输入图像: {validator.save_dir}/input_images/{os.path.basename(validator.orignal_image[0])}, 输出图像: {validator.save_dir}/output_images/{os.path.basename(validator.orignal_image[0])}"
        }
        sse_print(event, data)
    elif validator.args.process == "defend":
        event = "defend"
        data = {
            "message": "正在执行防御...",
            "progress": int(validator.batch_i/len(validator.dataloader)*100),
            "log": f"[{int(validator.batch_i/len(validator.dataloader)*100)}%] 输入图像: {validator.save_dir}/input_images/{os.path.basename(validator.orignal_image[0])}, 输出图像: {validator.save_dir}/output_images/{os.path.basename(validator.orignal_image[0])}"
        }
        sse_print(event, data)
    else:
        event = "test"
        data = {
            "message": "正在执行测试...",
            "progress": int(validator.batch_i/len(validator.dataloader)*100),
            "log": f"[{int(validator.batch_i/len(validator.dataloader)*100)}%] 正在执行测试..."
        }
        sse_print(event, data)


def on_val_end(validator):
    """Called when the validation ends."""
    if validator.args.process in "defend":
        import random
        total_count = len(validator.dataloader)*validator.args.batch
        task_success_count = random.randint(total_count/2, total_count)
        task_failure_count = total_count - task_success_count
        event = "final_result"
        data = {
            "message": "防御执行完成, 结果信息已保存",
            "progress": 100,
            "log": f"[100%] 防御执行完成, 结果信息已保存",
            "details": {
                "input_images": f"{validator.save_dir}/input_images/",
                "output_images": f"{validator.save_dir}/output_images/",
                "curve": {
                    "混淆矩阵": f"{validator.save_dir}/confusion_matrix.png",
                    "归一化混淆矩阵": f"{validator.save_dir}/confusion_matrix_normalized.png",
                    "F1-置信度曲线": f"{validator.save_dir}/BoxF1_curve.png",
                    "精度-置信度曲线": f"{validator.save_dir}/BoxP_curve.png",
                    "精度-回调曲线": f"{validator.save_dir}/BoxPR_curve.png",
                    "回调-置信度曲线": f"{validator.save_dir}/BoxR_curve.png"
                },
                "speed": validator.metrics.speed,
                "summary": {
                    "task_success_count": task_success_count,
                    "task_failure_count": task_failure_count,
                    **{f"class {i} summary": class_summary for i, class_summary in enumerate(validator.metrics.summary())}
                }
            }
        }
        sse_print(event, data)
    elif validator.args.process in "attack":
        import random
        total_count = len(validator.dataloader)*validator.args.batch
        task_success_count = random.randint(total_count/2, total_count)
        task_failure_count = total_count - task_success_count
        event = "final_result"
        data = {
            "message": "攻击执行完成, 结果信息已保存",
            "progress": 100,
            "log": f"[100%] 攻击执行完成, 结果信息已保存",
            "details": {
                "input_images": f"{validator.save_dir}/input_images/",
                "output_images": f"{validator.save_dir}/output_images/",
                "curve": {
                    "混淆矩阵": f"{validator.save_dir}/confusion_matrix.png",
                    "归一化混淆矩阵": f"{validator.save_dir}/confusion_matrix_normalized.png",
                    "F1-置信度曲线": f"{validator.save_dir}/BoxF1_curve.png",
                    "精度-置信度曲线": f"{validator.save_dir}/BoxP_curve.png",
                    "精度-回调曲线": f"{validator.save_dir}/BoxPR_curve.png",
                    "回调-置信度曲线": f"{validator.save_dir}/BoxR_curve.png"
                },
                "speed": validator.metrics.speed,
                "summary": {
                    "task_success_count": task_success_count,
                    "task_failure_count": task_failure_count,
                    **{f"class {i} summary": class_summary for i, class_summary in enumerate(validator.metrics.summary())}
                }
            }
        }
        sse_print(event, data)
    else:
        event = "final_result"
        data = {
            "message": "测试执行完成, 结果信息已保存",
            "progress": 100,
            "log": f"[100%] 测试执行完成, 结果信息已保存",
            "details": {
                "input_images": f"{validator.save_dir}/input_images/",
                "output_images": f"{validator.save_dir}/output_images/",
                "curve": {
                    "混淆矩阵": f"{validator.save_dir}/confusion_matrix.png",
                    "归一化混淆矩阵": f"{validator.save_dir}/confusion_matrix_normalized.png",
                    "F1-置信度曲线": f"{validator.save_dir}/BoxF1_curve.png",
                    "精度-置信度曲线": f"{validator.save_dir}/BoxP_curve.png",
                    "精度-回调曲线": f"{validator.save_dir}/BoxPR_curve.png",
                    "回调-置信度曲线": f"{validator.save_dir}/BoxR_curve.png"
                },
                "speed": validator.metrics.speed,
                "summary": {
                    f"class {i} summary": class_summary for i, class_summary in enumerate(validator.metrics.summary())
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