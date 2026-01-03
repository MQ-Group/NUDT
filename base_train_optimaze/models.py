import os
import torch
import torch.nn as nn
import torchvision.models as models


class LeNet5(nn.Module):
    """
    LeNet-5 网络结构
    原始论文: Gradient-Based Learning Applied to Document Recognition
    输入: 32x32 灰度图像
    输出: 10个类别 (0-9)
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # 特征提取部分
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 输入:1x32x32, 输出:6x28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 输入:6x14x14, 输出:16x10x10
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 展平后: 16x5x5 = 400
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 池化层
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 激活函数 (原始论文使用tanh，但ReLU更常用)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # 第一层: 卷积 -> 激活 -> 池化
        x = self.pool(self.activation(self.conv1(x)))  # 输出: 6x14x14
        
        # 第二层: 卷积 -> 激活 -> 池化
        x = self.pool(self.activation(self.conv2(x)))  # 输出: 16x5x5
        
        # 展平
        x = x.view(-1, 16 * 5 * 5)
        
        # 全连接层
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # 不需要激活函数，因为使用CrossEntropyLoss
        
        return x


def get_model(model_name, num_classes=10):
    if model_name == 'vgg':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'inception':
        model = models.inception_v3(weights=None, aux_logits=True)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'lenet':
        model = LeNet5()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model



def load_weight(model, pretrained_path):
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            model_dict = model.state_dict()
            # 只有当形状完全一致时才加载，防止分类头维度冲突
            pretrained_dict = {k: v for k, v in state_dict.items() 
                              if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        except Exception:
            pass

    return model