import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from mmcv.runner import get_dist_info
from utils.logger import get_root_logger

from .cls_head import ClsHead



def generate_random_orthogonal_matrix(feat_in, num_classes):
    """
    生成一个随机的正交矩阵，列向量 两两正交且模长为 1
    Args:
        feat_in:
        num_classes:

    Returns:

    """
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)  # QR 分解，得到一个正交矩阵和一个上三角矩阵
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


class DRLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            target,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight


class ETFHead(ClsHead):
    """Classification head for Baseline.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes: int, in_channels: int, *args, **kwargs) -> None:
        if kwargs.get('eval_classes', None):
            self.eval_classes = kwargs.pop('eval_classes')
        else:
            self.eval_classes = num_classes

        super().__init__(*args, **kwargs)
        self.compute_loss = DRLoss()
        assert num_classes > 0, f'num_classes={num_classes} must be a positive integer'

        self.num_classes = num_classes
        self.in_channels = in_channels
        logger = get_root_logger()
        logger.info("ETF head : evaluating {} out of {} classes.".format(self.eval_classes, self.num_classes))
        # logger.info("ETF head : with_len : {}".format(self.with_len))

        orth_vec = generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        self.register_buffer('etf_vec', etf_vec)

        etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
        self.etf_rect = etf_rect

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def forward_train(self, x: torch.Tensor, gt_label: torch.Tensor, **kwargs) -> Dict:
        """Forward training data."""
        x = self.pre_logits(x)
        target = self.etf_vec[:, gt_label].t()  # 直接把图片特征往设定好的 ETF 向量上 pull
        losses = self.loss(x, target)
        if self.cal_acc:
            with torch.no_grad():
                cls_score = x @ self.etf_vec
                acc = self.compute_accuracy(cls_score[:, :self.eval_classes], gt_label)
                assert len(acc) == len(self.topk)
                losses['accuracy'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, acc)
                }
        return losses

    def get_eft_logits(self,x,num_cls):
        x = self.pre_logits(x)
        cls_score = (x @ self.etf_vec)[:, :num_cls]
        return cls_score


    def loss(self, feat, target, **kwargs):
        losses = dict()
        # compute loss
        loss = self.compute_loss(feat, target)
        losses['loss'] = loss
        return losses

    def simple_test(self, x, softmax=False, post_process=False):
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        cls_score = cls_score[:, :self.eval_classes]
        assert not softmax
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score

    @staticmethod
    def produce_training_rect(label: torch.Tensor, num_classes: int):
        rank, world_size = get_dist_info()
        if world_size > 0:
            recv_list = [None for _ in range(world_size)]
            dist.all_gather_object(recv_list, label.cpu())
            new_label = torch.cat(recv_list).to(device=label.device)
            label = new_label
        uni_label, count = torch.unique(label, return_counts=True)
        batch_size = label.size(0)
        uni_label_num = uni_label.size(0)
        assert batch_size == torch.sum(count)
        gamma = torch.tensor(batch_size / uni_label_num, device=label.device, dtype=torch.float32)
        rect = torch.ones(1, num_classes).to(device=label.device, dtype=torch.float32)
        rect[0, uni_label] = torch.sqrt(gamma / count)
        return rect
