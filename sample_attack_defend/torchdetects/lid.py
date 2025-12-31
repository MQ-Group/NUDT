import torch
import torch.nn as nn

from torchattacks.attack import Attack

# Local Intrinsic Dimensionality
class LID(Attack):
    def __init__(self, model, k_nearest=20, detection_threshold=0.5):
        super().__init__("YOPO", model)
        self.k_nearest = k_nearest
        self.detection_threshold = detection_threshold

    def estimate_LID(self, x, references, k):
        """
        估计局部内在维度(LID)
        :param x: 查询点 (batch_size, ...)
        :param references: 参考点集合 (reference_size, ...)
        :param k: 近邻数量
        :return: LID 估计值 (batch_size,)
        """
        # 计算查询点与所有参考点的距离
        x = x.view(x.size(0), -1)  # 展平为向量
        references = references.view(references.size(0), -1)
        
        # 计算欧氏距离矩阵 (batch_size, reference_size)
        dist_matrix = torch.cdist(x, references, p=2)
        # 获取每个查询点的第k近邻距离
        dist_sorted, _ = torch.sort(dist_matrix, dim=1)
        kth_dist = dist_sorted[:, k]  # (batch_size,)
        
        # 避免除零错误
        kth_dist = torch.clamp(kth_dist, min=1e-8)
        
        # 计算到所有近邻的距离比例
        ratios = dist_sorted[:, :k] / kth_dist.unsqueeze(1)  # (batch_size, k)
        
        # 避免log(0)
        ratios = torch.clamp(ratios, min=1e-8)
        
        # LID公式: LID = -k / Σ(log(r_i/r_k))
        lid = -k / torch.sum(torch.log(ratios), dim=1)
        
        return lid

    def detect_adversarial_sample(self, x, reference_clean, reference_adv):
        """
        基于LID的对抗样本检测
        :param x: 待检测输入样本
        :param reference_clean: 干净样本参考集
        :param reference_adv: 对抗样本参考集
        :return: 检测结果（bool数组）、LID差异值
        """
        # 提取特征
        x_features = self.get_logits(x)
        clean_features = self.get_logits(reference_clean)
        adv_features = self.get_logits(reference_adv)
        
        
        # 分别计算相对于干净样本和对抗样本的LID
        lid_clean = self.estimate_LID(x_features, clean_features, self.k_nearest)
        lid_adv = self.estimate_LID(x_features, adv_features, self.k_nearest)
        
        # 计算LID差异作为检测依据
        lid_diff = torch.abs(lid_clean - lid_adv)
        
        # 判定是否为对抗样本
        is_adversarial = lid_diff > self.detection_threshold
            
        return is_adversarial, lid_diff