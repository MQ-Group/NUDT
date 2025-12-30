
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader


def build_dataset(is_train, args):
    
    if args.data_set == 'mnist':
        dataset = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.data_set == 'cifar10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.data_set == 'cifar100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'imagenet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    return dataset, nb_classes
