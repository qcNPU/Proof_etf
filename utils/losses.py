import torch
import torch.nn.functional as F
from .toolkit import *
import numpy as np
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


def multiproto_max(image_features,proto_features,targets):
    img_expanded = image_features.unsqueeze(1).expand(-1, proto_features.shape[1], -1).unsqueeze(1).expand(-1, proto_features.shape[0], -1, -1)  # 扩展 A 为 (64, 3, 512)
    proto_expanded = proto_features.unsqueeze(0).expand(image_features.shape[0], -1, -1, -1)  # 扩展为 (64, 10, 3, 512)
    # 计算余弦相似度：A_expanded 和 B 的形状是 (64, 3, 512)，我们可以计算它们的点积
    cos_sim = F.cosine_similarity(img_expanded, proto_expanded, dim=-1)  # 输出的形状是 (64,10, 3)

    max_sim_values, final_sim_indices = cos_sim.max(dim=2)  # 64，,10
    proto_features = proto_features[range(proto_features.shape[0]), final_sim_indices]  # 选择每个样本最相似的向量 64，10，,512
    protoloss = F.cross_entropy((image_features.unsqueeze(1) * proto_features).sum(-1), targets)
    return protoloss


# 计算原型之间的余弦相似度矩阵
def compute_similarity_matrix(prototypes):
    """
    计算所有原型之间的余弦相似度矩阵。
    prototypes: (10, 512) 维的原型张量
    返回一个 (10, 10) 维的相似度矩阵，其中 [i, j] 是原型 i 与原型 j 之间的余弦相似度。
    """
    prototypes = F.normalize(prototypes, p=2, dim=-1)  # 对原型进行L2归一化
    similarity_matrix = F.cosine_similarity(prototypes.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)  # shape: [3, 3]
    # similarity_matrix = torch.matmul(prototypes, prototypes.T)  # 计算余弦相似度矩阵
    return similarity_matrix




def aug_prototype(proto_idx, prototypes, similarity_matrix):
    """
    对一个原型进行增强，基于与最相似的原型。
    proto_idx: 当前目标原型的索引
    prototypes: (10, 512) 维的原型张量
    similarity_matrix: 预计算的原型相似度矩阵
    gamma: 加权因子，用于控制目标原型和相似原型的比例
    """
    proto = prototypes[proto_idx]

    # 获取相似度矩阵中的该原型与所有其他原型的相似度
    similarities = similarity_matrix[proto_idx]

    # 排除自己，选择最相似的原型 B（相似度最高的一个）
    similarities[proto_idx] = -float('inf')  # 将自己与自己的相似度设为负无穷
    top_sim_idx = similarities.argmax()  # 找到相似度最大的原型索引

    proto_B = prototypes[top_sim_idx]

    # 使用加权组合生成增强原型
    gamma = np.random.uniform(0.5, 1, 1)[0]  # 为增强原型设置一个随机权重
    proto_aug = proto * gamma + proto_B * (1 - gamma)

    return proto_aug


# 计算损失，首先增强原型，然后使用增强的原型计算CE损失
def compute_aug_proto1(feature, prototypes):
    """
    计算损失，首先增强原型，然后使用增强的原型计算CE损失。
    """
    similarity_matrix = compute_similarity_matrix(feature)
    # 对于每个类别，增强其原型
    augmented_prototypes = []
    for proto_idx in range(prototypes.shape[0]):  # 假设有10个类别
        proto_aug = aug_prototype(proto_idx, prototypes, similarity_matrix)
        augmented_prototypes.append(proto_aug)

    # 使用增强的原型进行CE损失计算
    augmented_prototypes = torch.stack(augmented_prototypes)  # 将增强的原型合并成一个Tensor

    return augmented_prototypes


def compute_aug_proto(features, prototypes, gamma=None):
    """
    计算损失，首先增强原型，然后返回增强的原型。
    features: (batch_size, 512) 维的文本特征张量
    prototypes: (10, 512) 维的原型张量
    gamma: (batch_size, 1) 或单一的加权因子
    返回增强后的原型 (10, 512) 张量
    """
    # 计算文本特征和原型之间的相似度矩阵
    similarity_matrix = compute_similarity_matrix(features)  # (batch_size, 10)

    # 如果没有提供 gamma，则随机生成一个 gamma 值
    if gamma is None:
        gamma = np.random.uniform(0.5, 1, prototypes.size(0))  # 生成 gamma
        gamma = torch.tensor(gamma, dtype=torch.float32).to(features.device)
        # gamma = torch.rand(features.size(0), 1).to(features.device)  # 从 [0, 1] 范围内生成 gamma

    # 计算最相似的原型索引
    _, top_sim_idx = similarity_matrix.max(dim=1)  # 获取每个文本特征最相似的原型索引

    # 扩展 gamma 以进行加权操作
    gamma = gamma.unsqueeze(-1)  # 变为 (batch_size, 1)，用于加权
    gamma_complement = 1 - gamma  # 计算 (1 - gamma) 作为加权补偿

    # 从原型矩阵中选择最相似的原型
    proto_B = prototypes[top_sim_idx]

    # 使用加权公式生成增强的原型
    proto_aug = prototypes * gamma + proto_B * gamma_complement   # 对每个原型进行加权

    return proto_aug


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