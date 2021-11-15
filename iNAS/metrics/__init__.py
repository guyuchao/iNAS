import importlib
from os import path as osp

from iNAS.utils import get_root_logger, scandir
from iNAS.utils.registry import METRIC_REGISTRY

__all__ = ['build_metric']

# automatically scan and import metric modules for registry
# scan all the files under the 'metrics' folder and collect files ending with
# '_metric.py'
metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder) if v.endswith('_metric.py')]
# import all the metric modules
_metric_modules = [importlib.import_module(f'iNAS.metrics.{file_name}') for file_name in metric_filenames]


def build_metric(metric_type, verbose=True):
    """Build metric from options.

    Args:
        metric_type (str): Metric type.
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    metric = METRIC_REGISTRY.get(metric_type)()
    logger = get_root_logger()
    if verbose:
        logger.info(f'Metric [{metric.__class__.__name__}] is created.')
    return metric
