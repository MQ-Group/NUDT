
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder
import os


def get_dataset(data_name, data_path, is_train):
    if data_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            ])
        dataset = datasets.MNIST(data_path, train=is_train, download=False, transform=transform)
        num_classes = 10
    elif data_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
            ])
        dataset = datasets.CIFAR10(data_path, train=is_train, download=False, transform=transform)
        num_classes = 10
    elif data_name == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
            ])
        dataset = datasets.CIFAR100(data_path, train=is_train, transform=transform)
        num_classes = 100
    elif data_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        root = os.path.join(data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        num_classes = 1000

    return dataset, num_classes


# 对于CIFAR等小尺寸数据集，需要resize到224x224（大多数预训练模型需要）
# if inputs.shape[-1] != 224:
#     import torch.nn.functional as F
#     inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)