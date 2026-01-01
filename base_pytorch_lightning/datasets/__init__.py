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


from .badnet.poisoned_dataset import CIFAR10Poison, MNISTPoison
from torchvision import datasets, transforms
import torch
from .casia_web_face.parquet_dataset import CasiaWebFaceParquet


def download_dataset(args):
    dataset = args.dataset.lower()
    if dataset == 'mnist':
        train_data = datasets.MNIST(root=args.data_path, train=True, download=args.download)
        test_data  = datasets.MNIST(root=args.data_path, train=False, download=args.download)
    elif dataset == 'cifar10':
        train_data = datasets.CIFAR10(root=args.data_path, train=True, download=args.download)
        test_data  = datasets.CIFAR10(root=args.data_path, train=False, download=args.download)
    elif dataset == 'imagenet':
        pass
    


def get_train_dataset(args):
    dataset = args.dataset.lower()
    if dataset == 'mnist':
        transform, detransform = build_transform('mnist')
        data = datasets.MNIST(args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'cifar10':
        transform, detransform = build_transform('cifar10')
        data = datasets.CIFAR10(args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'mnist_poison':
        transform, detransform = build_transform('mnist')
        data = MNISTPoison(args, args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'cifar10_poison':
        transform, detransform = build_transform('cifar10')
        data = CIFAR10Poison(args, args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'imagenet':
        pass
    elif dataset == 'casia_web_face':
        data = CasiaWebFaceParquet(
            root_dir=args.data_path,
            parquet_glob=getattr(args, 'parquet_glob', 'train-*.parquet'),
            image_field=getattr(args, 'image_field', 'image'),
            label_field=getattr(args, 'label_field', 'label'),
            path_field=getattr(args, 'path_field', 'path'),
            train=args.train,
            input_shape=(args.input_channels, args.input_height, args.input_width)
        )
    
    return data

def get_validate_dataset(args):
    dataset = args.dataset.lower()
    if dataset == 'mnist':
        transform, detransform = build_transform('mnist')
        data = datasets.MNIST(args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'cifar10':
        transform, detransform = build_transform('cifar10')
        data = datasets.CIFAR10(args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'mnist_poison':
        transform, detransform = build_transform('mnist')
        data = MNISTPoison(args, args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'cifar10_poison':
        transform, detransform = build_transform('cifar10')
        data = CIFAR10Poison(args, args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'imagenet':
        pass
    elif dataset == 'casia_web_face':
        data = CasiaWebFaceParquet(
            root_dir=args.data_path,
            parquet_glob=getattr(args, 'parquet_glob', 'train-*.parquet'),
            image_field=getattr(args, 'image_field', 'image'),
            label_field=getattr(args, 'label_field', 'label'),
            path_field=getattr(args, 'path_field', 'path'),
            train=args.train,
            input_shape=(args.input_channels, args.input_height, args.input_width)
        )
    
    return data

def get_test_dataset(args):
    dataset = args.dataset.lower()
    if dataset == 'mnist':
        transform, detransform = build_transform('mnist')
        data = datasets.MNIST(args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'cifar10':
        transform, detransform = build_transform('cifar10')
        data = datasets.CIFAR10(args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'mnist_poison':
        transform, detransform = build_transform('mnist')
        data = MNISTPoison(args, args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'cifar10_poison':
        transform, detransform = build_transform('cifar10')
        data = CIFAR10Poison(args, args.data_path, train=args.train, download=False, transform=transform)
    elif dataset == 'imagenet':
        pass
    elif dataset == 'casia_web_face':
        data = CasiaWebFaceParquet(
            root_dir=args.data_path,
            parquet_glob=getattr(args, 'parquet_glob', 'train-*.parquet'),
            image_field=getattr(args, 'image_field', 'image'),
            label_field=getattr(args, 'label_field', 'label'),
            path_field=getattr(args, 'path_field', 'path'),
            train=args.train,
            input_shape=(args.input_channels, args.input_height, args.input_width)
        )
    elif dataset == 'yolov5':
        from utils.yolov5.general import colorstr
        data = LoadImagesAndLabels(
            args.test_path,
            args.imgsz,
            args.batch_size,
            hyp=args.hyp,
            prefix=colorstr("val: "),
        )
    return data



def build_transform(dataset):
    if dataset == "cifar10":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "mnist":
        mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    return transform, detransform