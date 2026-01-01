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


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from . import download_dataset, get_train_dataset, get_validate_dataset, get_test_dataset

class DInterface(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.args = args

    def prepare_data(self):
        # print('prepare_data')
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        download_dataset(self.args)
    
    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        if stage == 'fit':
            pass

        if stage == 'validate':
            pass
            
        if stage == 'test':
            pass

        if stage == 'predict':
            pass
        
        if stage == 'tune':
            pass
    
    def setup(self, stage):
        # print('setup')
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if stage == 'fit':
            self.trainset = get_train_dataset(self.args)
            self.valset = get_validate_dataset(self.args)

        if stage == 'validate':
            self.testset = get_validate_dataset(self.args)
            
        if stage == 'test':
            self.testset = get_test_dataset(self.args)

        if stage == 'predict':
            self.predictset = get_test_dataset(self.args)
            
        if stage == 'tune':
            self.trainset = get_train_dataset(self.args)
            self.valset = get_validate_dataset(self.args)
            
        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.
    
        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)

    
    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predictset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    

