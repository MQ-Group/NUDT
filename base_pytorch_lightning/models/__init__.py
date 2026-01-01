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

from torch.nn import functional as F

from .backbone.badnet import BadNet
from .backbone.arcface import ArcFaceNet

def get_model(arg):
    name = arg.model
    if name == 'BadNet':
        model = BadNet(input_channels=arg.input_channels, output_num=arg.output_classes)
    elif name == 'ArcFace':
        # Use standard linear classifier; ArcMargin could be swapped later if needed
        model = ArcFaceNet(input_channels=arg.input_channels, output_num=arg.output_classes)
    
    return model


def gen_loss_fn(name):
    loss_fn = name.lower()
    if loss_fn == 'cross_entropy':
        loss_function = F.cross_entropy
    elif loss_fn == 'mse':
        loss_function = F.mse_loss
    elif loss_fn == 'l1':
        loss_function = F.l1_loss
    elif loss_fn == 'binary_cross_entropy':
        loss_function = F.binary_cross_entropy
    else:
        raise ValueError("Invalid Loss Type!")
    return loss_function
        