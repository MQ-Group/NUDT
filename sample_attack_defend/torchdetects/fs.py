import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks.attack import Attack

# Feature Squeezing
class FS(Attack):
    def __init__(self, model, kernel_size=3, bit_depth=4, detection_threshold=0.5):
        super().__init__("YOPO", model)
        self.kernel_size = kernel_size
        self.bit_depth = bit_depth
        self.detection_threshold = detection_threshold

    def _median_smoothing(self, x, kernel_size=3):
        """
        中值滤波平滑：有效去除椒盐噪声和对抗扰动
        :param x: 输入张量 (batch, C, H, W)
        :param kernel_size: 滤波核大小
        :return: 平滑后的张量
        """
        if kernel_size <= 1:
            return x
        # 中值滤波实现（PyTorch 无原生中值滤波，需手动展开）
        padding = (kernel_size - 1) // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        # 展开滑动窗口
        x_unfold = F.unfold(x_padded, kernel_size=kernel_size, stride=1)
        x_unfold = x_unfold.view(x.shape[0], x.shape[1], kernel_size*kernel_size, x.shape[2], x.shape[3])
        # 计算中值
        x_median = torch.median(x_unfold, dim=2)[0]
        return x_median

    def _bit_depth_reduction(self, x, bits):
        """
        位深度压缩：将像素值从 8bit (0-255) 压缩到指定 bit 深度
        :param x: 输入张量（范围 0-1）
        :param bits: 目标位深度（如 4）
        :return: 压缩后的张量
        """
        max_val = 2 ** bits - 1
        x = x * 255  # 还原到 0-255 范围
        x = torch.round(x / (255 / max_val))  # 压缩到指定 bit 深度
        x = x / 255  # 还原到 0-1 范围
        return x.clamp(0, 1)
    
    def squeeze(self, x):
        """
        Feature Squeezing 核心操作：组合位深度压缩 + 空间平滑
        :param x: 输入张量 (batch, 1, 28, 28)
        :return: squeezed 后的张量
        """
        # 1. 位深度压缩（核心操作）
        x_bit = self._bit_depth_reduction(x, self.bit_depth)
        
        # 2. 空间平滑，中值滤波
        x_squeezed = self._median_smoothing(x_bit, kernel_size=self.kernel_size)
        
        return x_squeezed
    
    def detect_adversarial_sample(self, x):
        """
        对抗样本检测核心逻辑：
        1. 计算原始样本和 squeezed 样本的预测差异
        2. 超过阈值则判定为对抗样本
        :param x: 输入样本
        :return: 检测结果（bool 数组）、预测差异值
        """
        # 原始样本预测
        pred_original = self.get_logits(x)
        
        # Squeezed 样本预测
        x_squeezed = self.squeeze(x)
        pred_squeezed = self.get_logits(x_squeezed)
        
        # 预测最大概率和类别
        max_values_original, max_indices_original = torch.max(pred_original, dim=1)
        max_values_squeezed, max_indices_squeezed = torch.max(pred_squeezed, dim=1)
        
        # 计算预测差异（L1 距离）
        pred_diff = torch.abs(max_values_original - max_values_squeezed)
        
        # 判定是否为对抗样本
        is_adversarial = (max_indices_original != max_indices_squeezed) or (pred_diff > self.detection_threshold and max_indices_original == max_indices_squeezed)
        
        # 类别不同时，差异为最大值
        pred_diff[max_indices_original != max_indices_squeezed] = 1.0
        
        return is_adversarial, pred_diff
    