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

    def forward(self, pred, soft_target):
        """[summary]

        Args:
            pred ([type]): Should contain keys like {'level_0': tensor, 'level_j':tensor}.
                We assume level_0 is the final prediction.
            soft_target ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert isinstance(pred, dict)
        assert 'level_0' in pred.keys()
        loss = 0.0
        for k in pred.keys():
            if k == 'level_0':
                loss += F.mse_loss(pred[k], soft_target)
            else:
                loss += self.aux_weight * F.mse_loss(pred[k], soft_target)
        return loss


@LOSS_REGISTRY.register()
class DSCrossEntropyLoss(nn.Module):

    def __init__(self, aux_weight=0.4, reduction='mean'):
        super(DSCrossEntropyLoss, self).__init__()
        self.aux_weight = aux_weight
        self.reduction = reduction

    def forward(self, pred, target):
        assert 'level_0' in pred and 'level_2' in pred, 'level_0 and level_2 are needed to compute loss'
        loss = F.cross_entropy(pred['level_0'], target, reduction=self.reduction, ignore_index=255)
        loss += self.aux_weight * F.cross_entropy(pred['level_2'], target, reduction=self.reduction, ignore_index=255)
        return loss
