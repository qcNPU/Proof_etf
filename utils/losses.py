import torch
import torch.nn.functional as F
from .toolkit import *

def separation_loss_cosine(proto_features):
    """
    计算每个类别的3个prototype之间的分离损失，使用余弦相似度。
    proto_features: [num_classes, 3, feature_dim]，每个类别有3个prototype。
    """
    num_classes, num_prototypes, feature_dim = proto_features.shape
    loss = 0.0

    for i in range(num_classes):
        # 取出每个类别的3个prototype
        prototypes = proto_features[i]  # shape: [3, feature_dim]

        # 计算pairwise余弦相似度
        sim_matrix = F.cosine_similarity(prototypes.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)  # shape: [3, 3]

        # 创建一个对角线为0的上三角矩阵，用于屏蔽自己与自己的相似度
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()

        # 只取上三角部分的相似度
        sim_matrix = sim_matrix[mask]

        # 为了使原型间尽量不同，我们需要最小化相似度的总和
        # 注意：这里我们取负值是因为优化器默认是求最小值，而我们希望相似度尽可能小
        loss += -sim_matrix.mean()

    # 平均每个类别的损失
    loss /= num_classes

    return loss




class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, etf_targets, labels):
        """
        Args:
            features: hidden vector of shape [bsz, dim], image features
            labels: ground truth labels of shape [bsz]
            etf_targets: target vectors of shape [num_classes, dim]
        Returns:
            A loss scalar.
        """
        device = features.device
        batch_size = features.shape[0]
        num_classes = etf_targets.shape[0]

        # Normalize features and targets (assuming already normalized)
        # features = F.normalize(features, p=2, dim=1)  # Normalize features
        # etf_targets = F.normalize(etf_targets, p=2, dim=1)  # Normalize ETF targets

        # Compute cosine similarity between features and targets: [batch_size, num_classes]
        similarity_matrix = torch.matmul(features, etf_targets.T) / self.temperature  # [bsz, num_classes]

        # Create a mask based on the labels, indicating the target index for each sample
        mask = torch.zeros(batch_size, num_classes, device=device)
        mask[torch.arange(batch_size), labels] = 1.0  # Only keep positive pairs for the correct class

        # For numerical stability, subtract the max value for each row (logits normalization)
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()   #去掉剪max

        log_prob = F.log_softmax(logits, dim=1)

        # Compute loss for each image (maximize similarity with the correct target, minimize with others)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss




def separation_loss(proto_features):
    """
    计算每个类别的3个prototype之间的分离损失。
    proto_features: [num_classes, 3, feature_dim]，每个类别有3个prototype。
    """
    num_classes, num_prototypes, feature_dim = proto_features.shape
    loss = 0.0

    for i in range(num_classes):
        # 取出每个类别的3个prototype
        prototypes = proto_features[i]  # shape: [3, feature_dim]

        # 计算pairwise欧氏距离
        # 计算每对prototype之间的距离
        dist_matrix = torch.cdist(prototypes.unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)  # shape: [3, 3]

        # 只取上三角部分的距离（不计算对角线，即自己与自己的距离）
        dist_matrix = dist_matrix[torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()]

        # 计算距离的平均值作为损失
        loss += -dist_matrix.mean()

    loss /= num_classes

    return loss