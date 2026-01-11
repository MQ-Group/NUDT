
import torch
from torchvision import transforms
from torchvision.models._meta import _VOC_CATEGORIES

class VOCDatasetAdapter:
    """
    适配 VOCDetection 数据集，使其输出符合 SSD 模型训练要求的格式。
    目标：将 VOC 的 XML 注释转换为 (boxes, labels) 张量。
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # VOC 的 20 个类别 + 背景 (0为背景)
        self.classes = _VOC_CATEGORIES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 加载原始图像和注释
        img, target = self.dataset[idx]
        # 确保图像是 RGB
        img = img.convert("RGB")

        # 解析注释：获取所有对象的边界框和标签
        boxes = []
        labels = []
        objects = target['annotation']['object']
        # 处理单个对象的情况（VOC 注释中可能将单个对象不放在列表里）
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            # 获取类别名
            class_name = obj['name']
            if class_name not in self.class_to_idx:
                continue  # 跳过不在类别列表中的对象（理论上不会发生）
            label = self.class_to_idx[class_name]

            # 获取边界框坐标 (xmin, ymin, xmax, ymax)
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin'])
            ymin = float(bndbox['ymin'])
            xmax = float(bndbox['xmax'])
            ymax = float(bndbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        if len(boxes) == 0:
            # 如果没有有效目标，提供一个虚拟框和标签 (避免运行时错误)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # 将 PIL 图像转换为 Tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = transform(img)

        # SSD 模型要求目标是一个字典，包含 'boxes' 和 'labels'
        target = {
            "boxes": boxes,
            "labels": labels
        }
        return img, target