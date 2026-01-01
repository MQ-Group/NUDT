import os
from typing import Optional

import pytorch_lightning as pl
from torchvision.utils import make_grid, save_image


class AndbImageGridSaver(pl.Callback):
    def __init__(self, save_dir_key: str = 'default_root_dir', subdir: str = 'andb', every_n_epochs: int = 1, max_batches: int = 1):
        super().__init__()
        self.save_dir_key = save_dir_key
        self.subdir = subdir
        self.every_n_epochs = every_n_epochs
        self.max_batches = max_batches

    def _get_save_dir(self, trainer: 'pl.Trainer') -> Optional[str]:
        base = getattr(trainer, self.save_dir_key, None)
        if base is None:
            base = trainer.default_root_dir
        if base is None:
            return None
        path = os.path.join(base, self.subdir)
        os.makedirs(path, exist_ok=True)
        return path

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        epoch = trainer.current_epoch
        if self.every_n_epochs and (epoch + 1) % self.every_n_epochs != 0:
            return
        save_dir = self._get_save_dir(trainer)
        if save_dir is None:
            return
        dataloader = trainer.datamodule.val_dataloader()
        batches_saved = 0
        for batch in dataloader:
            imgs, labels = batch
            grid = make_grid(imgs[:64], nrow=8, normalize=True, scale_each=True)
            save_path = os.path.join(save_dir, f'val_epoch{epoch+1}_batch{batches_saved+1}.png')
            save_image(grid, save_path)
            batches_saved += 1
            if batches_saved >= self.max_batches:
                break


