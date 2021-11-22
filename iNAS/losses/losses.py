import torch
from torch import nn as nn
from torch.nn import functional as F

from iNAS.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


class BCELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        self.reduction = reduction

    def forward(self, pred, target):
        target = target.float()
        return F.binary_cross_entropy(pred, target)


@LOSS_REGISTRY.register()
class DSBCELoss(nn.Module):
    """[summary]

    Args:
        aux_weight (float, optional): [description]. Defaults to 0.4.
        reduction (str, optional): [description]. Defaults to 'mean'.

    Raises:
        ValueError: [description]
    """

    def __init__(self, aux_weight=0.4, reduction='mean'):
        super(DSBCELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        self.reduction = reduction
        self.aux_weight = aux_weight

    def forward(self, pred, target):
        """[summary]

        Args:
            pred (dict): Should contain keys like {'level_0': tensor, 'level_j':tensor}.
                We assume level_0 is the final prediction.
            target ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert isinstance(pred, dict)
        assert 'level_0' in pred.keys()
        target = target.float()
        loss = 0.0
        for k in pred.keys():
            if k == 'level_0':
                loss += F.binary_cross_entropy(pred[k], target)
            else:
                loss += self.aux_weight * F.binary_cross_entropy(pred[k], target)
        return loss


@LOSS_REGISTRY.register()
class DSMSELoss(nn.Module):
    """[summary]

    Args:
        aux_weight (float, optional): [description]. Defaults to 0.4.
        reduction (str, optional): [description]. Defaults to 'mean'.

    Raises:
        ValueError: [description]
    """

    def __init__(self, aux_weight=0.4, reduction='mean'):
        super(DSMSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        self.reduction = reduction
        self.aux_weight = aux_weight

    def forward(self, pred_S, pred_T):
        """[summary]

        Args:
            pred ([type]): Should contain keys like {'level_0': tensor, 'level_j':tensor}.
                We assume level_0 is the final prediction.
            soft_target ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert isinstance(pred_S, dict)
        assert 'level_0' in pred_S.keys()
        soft_target = pred_T['level_0'].detach()
        loss = 0.0
        for k in pred_S.keys():
            if k == 'level_0':
                loss += F.mse_loss(pred_S[k], soft_target)
            else:
                loss += self.aux_weight * F.mse_loss(pred_S[k], soft_target)
        return loss


@LOSS_REGISTRY.register()
class DSCrossEntropyLoss(nn.Module):

    def __init__(self, weight, aux_weight=0.4, reduction='mean'):
        super(DSCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.aux_weight = aux_weight
        self.reduction = reduction

    def forward(self, pred, target):
        assert 'level_0' in pred and 'level_2' in pred, 'level_0 and level_2 are needed to compute loss'
        loss = F.cross_entropy(pred['level_0'], target, reduction=self.reduction, ignore_index=255)
        loss += self.aux_weight * F.cross_entropy(pred['level_2'], target, reduction=self.reduction, ignore_index=255)
        return self.weight * loss


@LOSS_REGISTRY.register()
class DSPixelWiseKLLoss(nn.Module):

    def __init__(self, weight, ignore_index=255, aux_weight=0.4, reduction='mean'):
        super(DSPixelWiseKLLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, pred, soft_target):
        pred_level_0 = pred['level_0']
        soft_target_level_0 = soft_target['level_0'].detach()

        assert pred_level_0.shape == soft_target_level_0.shape, 'the output dim of teacher and student differ'
        N, C, W, H = pred_level_0.shape
        softmax_pred_T = F.softmax(soft_target_level_0.permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum(
            -softmax_pred_T * logsoftmax(pred_level_0.permute(0, 2, 3, 1).contiguous().view(-1, C)))) / W / H

        # side output
        if self.aux_weight > 0.0:
            pred_level_2 = pred['level_2']
            soft_target_level_2 = soft_target['level_2'].detach()
            N, C, W, H = pred_level_2.shape
            softmax_pred_T2 = F.softmax(soft_target_level_2.permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
            logsoftmax = nn.LogSoftmax(dim=1)
            loss_aux = (torch.sum(
                -softmax_pred_T2 * logsoftmax(pred_level_2.permute(0, 2, 3, 1).contiguous().view(-1, C)))) / W / H
            loss += self.aux_weight * loss_aux

        return self.weight * loss
