from copy import deepcopy

from iNAS.utils import get_root_logger
from iNAS.utils.registry import LOSS_REGISTRY
from .losses import DSBCELoss, DSMSELoss

__all__ = ['DSMSELoss', 'DSBCELoss']


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must constain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
