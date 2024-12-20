# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

from .accuracy import Accuracy


class ClsHead(torch.nn.Module):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
    """

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 cal_acc=False,
                 init_cfg=None):
        super(ClsHead, self).__init__()

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = None
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.cal_acc = cal_acc

    def loss(self, cls_score, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses

    def forward_train(self, cls_score, gt_label, **kwargs):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def norm(self, x):
        if isinstance(x, tuple):
            x = x[-1]

        warnings.warn(
            'The input of ClsHead should be already logits. '
            'Please modify the backbone if you want to get pre-logits feature.'
        )
        return x