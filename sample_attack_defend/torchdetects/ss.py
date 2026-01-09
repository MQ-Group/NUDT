import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks.attack import Attack

# Spatial Smoothing
class SS(Attack):
    def __init__(self, model, kernel_size=3, detection_threshold=0.5):
        super().__init__("YOPO", model)
        self.kernel_size = kernel_size
        self.detection_threshold = detection_threshold

    def _median_smoothing(self, x, kernel_size=3):
        """
        中值滤波平滑：有效去除椒盐噪声和对抗扰动
        :param x: 输入张量 (batch, C, H, W)
        :param kernel_size: 滤波核大小
        :return: 平滑后的张量
        """
        # print(x.shape) # torch.Size([1, 3, 32, 32])
        if kernel_size <= 1:
            return x
        # 中值滤波实现（PyTorch 无原生中值滤波，需手动展开）
        padding = (kernel_size - 1) // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        # print(x_padded.shape) # torch.Size([1, 3, 34, 34])
        # 展开滑动窗口
        x_unfold = F.unfold(x_padded, kernel_size=kernel_size, stride=1)
        # print(x_unfold.shape) # torch.Size([1, 27, 1024])
        x_unfold = x_unfold.view(x.shape[0], x.shape[1], kernel_size*kernel_size, x.shape[2], x.shape[3])
        # 计算中值
        x_median = torch.median(x_unfold, dim=2)[0]
        return x_median

    def smooth(self, x):
        """
        Spatial Smoothing 核心操作：应用中值滤波进行空间平滑
        :param x: 输入张量 (batch, 1, 28, 28)
        :return: smoothed 后的张量
        """
        return self._median_smoothing(x, kernel_size=self.kernel_size)
    
    def detect_adversarial_sample(self, x):
        """
        对抗样本检测核心逻辑：
        1. 计算原始样本和 smoothed 样本的预测差异
        2. 超过阈值则判定为对抗样本
        :param x: 输入样本
        :return: 检测结果（bool 数组）、预测差异值
        """
        # print(x.shape) # torch.Size([1, 3, 32, 32])
        # 原始样本预测
        pred_original = self.get_logits(x)
        
        # Smoothed 样本预测
        x_smoothed = self.smooth(x)
        pred_smoothed = self.get_logits(x_smoothed)
        
        # 预测最大概率和类别
        max_values_original, max_indices_original = torch.max(pred_original, dim=1)
        max_values_smoothed, max_indices_smoothed = torch.max(pred_smoothed, dim=1)
        
        # 计算预测差异（L1 距离）
        pred_diff = torch.abs(max_values_original - max_values_smoothed)
        
        # 判定是否为对抗样本
        is_adversarial = (max_indices_original != max_indices_smoothed) or (pred_diff > self.detection_threshold and max_indices_original == max_indices_smoothed)
        
        # 类别不同时，差异为最大值
        pred_diff[max_indices_original != max_indices_smoothed] = 1.0
        
        return is_adversarial, pred_diff
    