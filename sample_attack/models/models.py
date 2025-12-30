import os
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes=10, pretrained_path=None):
    if model_name.lower() in ['vgg16', 'vgg']:
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name.lower() in ['resnet50', 'resnet']:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() in ['inception_v3', 'inception']:
        model = models.inception_v3(weights=None, aux_logits=True)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if pretrained_path and os.path.exists(pretrained_path):
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
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


