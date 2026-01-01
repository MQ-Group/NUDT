# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc

# from sklearn.metrics import accuracy_score

from . import get_model, gen_loss_fn

class MInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        self.model = get_model(self.hparams)
        self.loss_function = gen_loss_fn(self.hparams.loss_fun)
        
    def forward(self, img):
        return self.model(img)

    
    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.5, 0.999), eps=1e-06)
        else:
            raise ValueError('Invalid optimizer type!')

        
        if getattr(self.hparams, 'lr_scheduler', None) is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler.lower() == 'steplr':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_decay_step, gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler.lower() == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.lr_decay_step, eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]
    
    def configure_callbacks(self):
        # When the model gets attached, e.g., when ``.fit()`` or ``.test()`` gets called,
        # Return: A list of callbacks which will extend the list of callbacks in the Trainer.
        # https://lightning.ai/docs/pytorch/1.4.0/extensions/callbacks.html
        early_stop = plc.EarlyStopping(
            monitor='val_acc',
            mode='max',
            patience=10,
            min_delta=0.001
        )

        checkpoint = plc.ModelCheckpoint(
            monitor='val_acc',
            filename='best-{epoch:02d}-{val_acc:.3f}',
            save_top_k=1,
            mode='max',
            save_last=True
        )

        return [early_stop, checkpoint]
            
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        img, labels = batch
        out = self(img)
        loss = self.loss_function(out, labels)
        
        # Logs the loss per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Return tensor to call ".backward" on
        return loss

    def training_step_end(self, training_step_outputs):
        '''
        当gpus=0 or 1时，这里的batch_parts即为traing_step的返回值（已验证）
        当gpus>1时，这里的batch_parts为list，list中每个为training_step返回值，list[i]为i号gpu的返回值（这里未验证）
        '''
        # print('training_step_end')
        pass

    def training_epoch_end(self, outputs):
        '''
        当gpu=0 or 1时，training_step_outputs为list，长度为steps的数量（不包括validation的步数，当你训练时，你会发现返回list<训练时的steps数，这是因为训练时显示的steps数据还包括了validation的，若将limit_val_batches=0.，即关闭validation，则显示的steps会与training_step_outputs的长度相同）。list中的每个值为字典类型，字典中会存有`training_step_end()`返回的键值，键名为`training_step()`函数返回的变量名，另外还有该值是在哪台设备上(哪张GPU上)，例如{device='cuda:0'}
        '''
        # print('training_epoch_end')
        pass
    

    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        img, labels = batch
        out = self(img)
        loss = self.loss_function(out, labels)
        # labels_predict = torch.argmax(out, dim=1)
        # acc = accuracy_score(labels, labels_predict)
        acc = (out.argmax(dim=-1) == labels).float().mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
        # self.log_dict(values, on_step=False, on_epoch=True, prog_bar=True)


    def validation_step_end(self, validation_step_outputs):
        # print('validation_step_end')
        pass

    def validation_epoch_end(self, outputs):
        # print('validation_epoch_end')
        pass
        
    
    def test_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        img, labels = batch
        out = self(img)
        # labels_predict = torch.argmax(out, dim=1)
        # acc = accuracy_score(labels, labels_predict)
        acc = (out.argmax(dim=-1) == labels).float().mean()

        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step_end(self, test_step_outputs):
        # print('test_step_end')
        pass
    
    def test_epoch_end(self, outputs):
        # print('test_epoch_end')
        pass
        
        
    def predict_step(self, batch, batch_idx, dataloader_idx):
        img, labels = batch
        out = self(img)
        
        # self.write_prediction(name='pred', value=out, filename='predictions.pt')
        
        # pred_dict = {'pred1': torch.tensor(...), 'pred2': torch.tensor(...)}
        # self.write_prediction_dict(pred_dict)
        return out

