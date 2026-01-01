import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from typing import Optional, Dict, Any, List
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
import json
import torch

'''
sse_dict = {
    "event": "task_failed",
    "data": {
        "status": "failure",
        "message": "Task failed due to an error.",
    "error_details": exception
    }
}
print(json.dumps(sse_dict, indent=4))
print('\n\n')
        
try:
    state_dict = torch.load(self.hparams.checkpoint)['state_dict']
    self.model.load_state_dict(state_dict)
    sse_dict = {
        "event": "model_loaded",
        "data": {
            "status": "success",
            "message": "Model loaded successfully.",
            "model_name": self.hparams.model,
            "model_path": self.hparams.checkpoint
        }
    }
    print(json.dumps(sse_dict, indent=4))
    print('\n\n')
except Exception as e:
    sse_dict = {
        "event": "model_loaded",
        "data": {
            "status": "failure",
            "message": "Failed to load model.",
            "error_details": "File not found."
        }
    }
    print(json.dumps(sse_dict, indent=4))
    print('\n\n')
            
sse_dict = {
    "event": "source_unzip_completed",
    "data": {
        "status": "success",
        "message": "Source zip file extracted successfully.",
        "image_count": 100,
        "output_directory": "/tmp/task_12345/images"
    }
}

sse_dict = {
    "event": "adv_zip_created",
    "data": {
        "status": "success",
        "message": "Adversarial samples zipped successfully.",
        "zip_file_path": "/tmp/task_12345/output/adv_samples.zip"
    }
}


sse_dict = {
    "event": "adv_sample_generation_start",
    "data": {
        "message": "Starting adversarial sample generation for image.",
        "image_name": "image_001.jpg"
    }
}

sse_dict = {
    "event": "adv_sample_generated",
    "data": {
        "status": "success",
        "message": "Adversarial sample generated successfully.",
        "image_name": "image_001.jpg",
        "metrics": {
        "mae": 0.123,
        "psnr": 30.5,
        "ssim": 0.95
    }
}



sse_dict = {
    "event": "global_metrics_calculated",
    "data": {
        "status": "success",
        "message": "Global metrics calculated.",
        "metrics": {
            "average_mae": 0.15,
            "std_dev_mae": 0.02,
            "average_psnr": 29.8,
            "std_dev_psnr": 0.5
        }
    }
}



sse_dict = {
    "event": "final_result",
    "data": {
        "status": "success",
        "message": "Final result summary.",
        "result": {
            "output_file": "/tmp/task_12345/output/adv_samples.zip",
            "metrics": {
                "average_mae": 0.15,
                "std_dev_mae": 0.02
            }
        }
    }
}

'''
    

class Callback(plc.Callback):
    # https://lightning.ai/docs/pytorch/1.4.0/extensions/generated/pytorch_lightning.callbacks.Callback.html#pytorch_lightning.callbacks.Callback
    
    def on_configure_sharded_model(self, pl_module: 'pl.LightningModule') -> None:
        """Called before configure sharded model"""

    def on_before_accelerator_backend_setup(self, pl_module: 'pl.LightningModule') -> None:
        """Called before accelerator is being setup"""
        pass
    
    def setup(self, pl_module: 'pl.LightningModule', stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune begins"""
        sse_dict = {
            "event": "task_initialized",
            "data": {
                "status": "success",
                "message": "Task initialized successfully.",
                "parameters": pl_module.hparams
            }
        }
        print(json.dumps(sse_dict, indent=4))
        print('\n\n')

    def teardown(self, pl_module: 'pl.LightningModule', stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune ends"""
        sse_dict = {
            "event": "task_completed",
            "data": {
                "status": "success",
                "message": "Task completed successfully.",
                "summary": "xxxxxxx"
            }
        }
        print(json.dumps(sse_dict, indent=4))
        print('\n\n')
    
    def on_init_start(self) -> None:
        """Called when the trainer initialization begins, model has not yet been set."""
        pass

    def on_init_end(self) -> None:
        """Called when the trainer initialization ends, model has not yet been set."""
        pass

    def on_fit_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when fit begins"""
        pass

    def on_fit_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when fit ends"""
        pass
    
    
    def on_train_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the train begins."""
        pass

    def on_train_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the train ends."""
        sse_dict = {
            "event": "global_metrics_calculated",
            "data": {
                "status": "success",
                "message": "Global metrics calculated.",
                # "metrics": pl_module.metrics
            }
        }
        print(json.dumps(sse_dict, indent=4))
        print('\n\n')
    
    def on_validation_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the test begins."""
        pass

    def on_test_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the test ends."""
        pass

    def on_predict_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the predict begins."""
        pass

    def on_predict_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when predict ends."""
        pass
    
    def on_batch_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the training batch begins."""
        pass

    def on_train_batch_start(
        self,
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch begins."""
        pass

    def on_train_batch_end(
        self,
        pl_module: 'pl.LightningModule',
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        pass
    
    def on_validation_batch_start(
        self,
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch begins."""
        pass

    def on_validation_batch_end(
        self,
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""
        pass

    def on_test_batch_start(
        self,
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch begins."""
        pass

    def on_test_batch_end(
        self,
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""
        pass

    def on_predict_batch_start(
        self,
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch begins."""
        pass

    def on_predict_batch_end(
        self,
        pl_module: 'pl.LightningModule',
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends."""
        pass

    def on_batch_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the training batch ends."""
        pass
    
    def on_pretrain_routine_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the pretrain routine begins."""
        pass

    def on_pretrain_routine_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the pretrain routine ends."""
        pass
    
    def on_sanity_check_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the validation sanity check starts."""
        pass

    def on_sanity_check_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the validation sanity check ends."""
        pass
    
    def on_keyboard_interrupt(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the training is interrupted by ``KeyboardInterrupt``."""
        pass

    def on_train_epoch_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(self, pl_module: 'pl.LightningModule', unused: Optional = None) -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """
        pass
    
    def on_validation_epoch_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the val epoch begins."""
        pass

    def on_validation_epoch_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the val epoch ends."""
        pass

    def on_test_epoch_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the test epoch begins."""
        pass

    def on_test_epoch_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the test epoch ends."""
        pass

    def on_predict_epoch_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when the predict epoch begins."""
        pass

    def on_predict_epoch_end(self, pl_module: 'pl.LightningModule', outputs: List[Any]) -> None:
        """Called when the predict epoch ends."""
        pass
    
    def on_epoch_start(self, pl_module: 'pl.LightningModule') -> None:
        """Called when either of train/val/test epoch begins."""
        pass

    def on_epoch_end(self, pl_module: 'pl.LightningModule') -> None:
        """Called when either of train/val/test epoch ends."""
        pass
    
    def on_load_checkpoint(
        self, pl_module: 'pl.LightningModule', callback_state: Dict[str, Any]
    ) -> None:
        """Called when loading a model checkpoint, use to reload state.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            pl_module: the current :class:`~pytorch_lightning.core.lightning.LightningModule` instance.
            callback_state: the callback state returned by ``on_save_checkpoint``.

        Note:
            The ``on_load_checkpoint`` won't be called with an undefined state.
            If your ``on_load_checkpoint`` hook behavior doesn't rely on a state,
            you will still need to override ``on_save_checkpoint`` to return a ``dummy state``.
        """
        pass
    
    def on_save_checkpoint(
        self, pl_module: 'pl.LightningModule', checkpoint: Dict[str, Any]
    ) -> dict:
        """
        Called when saving a model checkpoint, use to persist state.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            pl_module: the current :class:`~pytorch_lightning.core.lightning.LightningModule` instance.
            checkpoint: the checkpoint dictionary that will be saved.

        Returns:
            The callback state.
        """
        pass
    

    def on_after_backward(self, pl_module: 'pl.LightningModule') -> None:
        """Called after ``loss.backward()`` and before optimizers do anything."""
        pass

    def on_before_zero_grad(self, pl_module: 'pl.LightningModule', optimizer: Optimizer) -> None:
        """Called after ``optimizer.step()`` and before ``optimizer.zero_grad()``."""
        pass    
    