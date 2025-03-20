from copy import deepcopy
from typing import Dict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from utils.losses import SupConLoss
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

    def __init__(self, num_classes: int, in_channels: int,device,target_choose,target_match, common_match, *args, **kwargs) -> None:
        if kwargs.get('eval_classes', None):
            self.eval_classes = kwargs.pop('eval_classes')
        else:
            self.eval_classes = num_classes

        super().__init__(*args, **kwargs)
        self.target_choose =target_choose
        self.target_match =target_match
        self.common_match = common_match
        if self.target_match == "random":
            self.target_choose = "fix"
        print(f"etfhead target_choose:{self.target_choose},target_match:{ self.target_match}")
        self.losses = "dr"
        # self.losses = "supcontra"
        if self.losses == "dr":
            self.compute_loss = DRLoss()
        else:
            self.compute_loss = SupConLoss()
        assert num_classes > 0, f'num_classes={num_classes} must be a positive integer'

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.device = device
        # logger = get_root_logger()
        # logger.info("ETF head : evaluating {} out of {} classes.".format(self.eval_classes, self.num_classes))

        orth_vec = generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        self.register_buffer('etf_vec', etf_vec.T)  #最终放进去的：(100,512)
        self.assignInfo = {}
        self.assignIndex = {}
        if self.common_match != 'common':
            self.register_buffer('etf_vec_text', deepcopy(etf_vec.T))  #最终放进去的：(100,512)
            self.assignInfo_text = {}
            self.assignIndex_text = {}



        # self.projector = self.select_projector(512,2048,512)
        # self.classifiers = nn.Sequential()

    def norm(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x


    def forward_train_v1(self, match_vec, gt_label,modal, **kwargs) -> Dict:

        x = self.norm(match_vec)
        target = self.assign_target(x, gt_label,modal)
        if self.losses == "dr":
            losses = self.loss(x, target)
        else:
            losses = self.loss(x, target,gt_label)
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


    def forward_train(self, match_vec,optimize_vec, gt_label, **kwargs) -> Dict:
        """Forward training data."""
        x = self.norm(match_vec)
        target = self.assign_target(x, gt_label)
        opti = self.norm(optimize_vec)
        losses = self.loss(opti, target)
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

    def assign_target(self, source, source_labels, modal):
        """
        为新类别分配target，最后返回所有类别的target
        Args:
            source:
            source_labels:

        Returns:

        """
        # new_lb = [i for i in source_labels if self.assignInfo.get(i.item()) is None]
        use_text = self.common_match != 'common' and modal == 'text'  #各自匹配
        assignIndex =  self.assignIndex_text if use_text else self.assignIndex
        assignInfo = self.assignInfo_text if use_text else self.assignInfo
        etf_vec = self.etf_vec_text if use_text else self.etf_vec
        new_lb = {ind: lb for ind, lb in enumerate(source_labels) if assignInfo.get(lb) is None}  # 字典去重

        if len(new_lb) > 0:
            # Normalise incoming prototypes
            # base_prototypes = normalize(source[-len(new_lb):,:])
            if "random" in self.target_match:
                num_new = len(new_lb)
                # 获取所有可用的ETF向量索引（总数量需>=新类别数量）
                num_etf = etf_vec.size(0)  # 假设etf_vec形状为 [num_etf, feature_dim]
                assert num_etf >= num_new, "ETF向量数量不足"
                # 生成随机无放回索引 (PyTorch设备兼容)
                rand_indices = torch.randperm(num_etf, device=self.device)[:num_new]
                col_ind = rand_indices.cpu().numpy()  # 转换为numpy数组保持接口兼容
                row_id = np.arange(num_new)  # 按顺序分配新类别
                # 定义keys（新增关键修复）
                keys = {ind: lb for ind, lb in enumerate(list(new_lb.values()))}
            else:
                keys = {ind:lb for ind,lb in enumerate(list(new_lb.values()))}
                base_prototypes = source[list(new_lb.keys())]  # 按照索引顺序取出源向量
                # 根据与class prototype 的相似度从初始化向量池中选择最相似的
                cost = cosine_similarity(base_prototypes.detach().cpu(), etf_vec.cpu())
                # labels 只用来记录哪个类别选了哪个向量
                row_id, col_ind = linear_sum_assignment(cost, maximize=True)

            # new_fc_tensor = self.etf_vec[col_ind]
            # Creating and appending a new classifier from the given reserved vectors
            # new_fc = nn.Linear(new_fc_tensor.shape[1], new_fc_tensor.shape[0], bias=False)
            # new_fc.weight.data.copy_(new_fc_tensor)
            # self.classifiers.append(new_fc)
            for i, label in keys.items():
                assignInfo[label] = etf_vec[col_ind[i]].view(-1, self.in_channels)  # 把 value 直接变成向量
                assignIndex[label] = col_ind[i]  # 先存着后面用

            if self.target_choose == "fix":
                # Remove from the final rv ，将已分配的向量从池子中去掉
                all_idx = np.arange(etf_vec.shape[0])
                etf_vec1 = etf_vec[all_idx[~np.isin(all_idx, col_ind)]]
                if use_text:
                    del self.etf_vec_text
                    self.register_buffer('etf_vec_text', etf_vec1)
                else:
                    del self.etf_vec
                    self.register_buffer('etf_vec', etf_vec1)
            print(f"modal:{modal},assignIndex: {assignIndex}")
        # 将类别对应的 target 向量返回
        assign_target = torch.cat([assignInfo[label] for label in source_labels], dim=0)
        return assign_target

    def clear_assignment(self, class_num):
        self.assignInfo = {}
        self.assignIndex = {}
        # self.classifiers = nn.Sequential()

    def get_assignment(self, cost, maximize=True):
        """Tak array with cosine scores and return the output col ind """
        _, col_ind = linear_sum_assignment(cost, maximize=True)
        return col_ind

    def get_eft_logits(self, x, total_class):
        # x = self.norm(x)
        #有缓存， 直接获取的是前面分配的target
        assign_target = self.assign_target(x, total_class)
        cls_score = (x @ assign_target.T)  # text feature 和 etf 对齐之后，etf 就不在分类中起作用了，不然没法 test
        return cls_score


    def get_classifier_logits(self, feat):
        output = []
        self.classifiers = self.classifiers.to(feat.device)
        for i, cls in enumerate(self.classifiers.children()):
            out = F.linear(F.normalize(feat, p=2, dim=-1), F.normalize(cls.weight, p=2, dim=-1))
            # out = out / self.temperature
            output.append(out)
        output = torch.cat(output, axis = 1)
        return output


    def loss(self, feat, target,gt_label=None):
        losses = dict()
        # compute loss
        loss = self.compute_loss(feat, target,gt_label)
        losses['loss'] = loss
        return losses

    def simple_test(self, x, softmax=False, post_process=False):
        x = self.norm(x)
        cls_score = x @ self.etf_vec
        cls_score = cls_score[:, :self.eval_classes]
        assert not softmax
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score

    # @staticmethod
    # def produce_training_rect(label: torch.Tensor, num_classes: int):
    #     rank, world_size = get_dist_info()
    #     if world_size > 0:
    #         recv_list = [None for _ in range(world_size)]
    #         dist.all_gather_object(recv_list, label.cpu())
    #         new_label = torch.cat(recv_list).to(device=label.device)
    #         label = new_label
    #     uni_label, count = torch.unique(label, return_counts=True)
    #     batch_size = label.size(0)
    #     uni_label_num = uni_label.size(0)
    #     assert batch_size == torch.sum(count)
    #     gamma = torch.tensor(batch_size / uni_label_num, device=label.device, dtype=torch.float32)
    #     rect = torch.ones(1, num_classes).to(device=label.device, dtype=torch.float32)
    #     rect[0, uni_label] = torch.sqrt(gamma / count)
    #     return rect
