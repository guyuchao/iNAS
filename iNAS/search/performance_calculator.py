import torch
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm

from iNAS.archs import build_network
from iNAS.data import build_dataloader, build_dataset
from iNAS.data.data_sampler import EnlargedSampler
from iNAS.metrics import build_metric
from iNAS.utils import get_root_logger, tensor2img_batch
from iNAS.utils.bn_utils import update_bn_stats
from iNAS.utils.registry import PERFORMANCE_REGISTRY


def build_performance(opt):
    """Build performance evaluator from options.

    Args:
        opt (dict): Configuration.
    """
    opt = deepcopy(opt)
    model = PERFORMANCE_REGISTRY.get(opt['search']['performance_metrics']['type'])(opt)
    logger = get_root_logger()
    logger.info(f'Search [{model.__class__.__name__}] is created.')
    return model


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_bn_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = build_dataset(dataset_opt)
            # set finetune bn loader
            train_bn_loader = build_dataloader(
                train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])

            logger.info('Training statistics:' f'\n\tNumber of train images: {len(train_set)}')
        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_sampler = EnlargedSampler(val_set, opt['world_size'], opt['rank'], 1)
            val_loader = build_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=val_sampler,
                seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: ' f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_bn_loader, val_loader


@PERFORMANCE_REGISTRY.register()
class PerformanceCalculator():

    def __init__(self, opt):
        self.opt = opt
        self.logger = get_root_logger()
        self.train_bn_loader, self.val_loader = create_train_val_dataloader(opt, self.logger)
        self.supernet = build_network(opt['supernet'])

        # load pretrained models
        load_path = self.opt['path'].get('supernet_path', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key', 'params')
            self.load_network(self.supernet, load_path, self.opt['path'].get('strict_load', True), param_key)

        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.supernet = self.model_to_device(self.supernet)

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)

        if param_key is not None:
            load_net = load_net[param_key]

        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def validation(self, sample_arch, metric='Fmeasure', progress=False):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            raise NotImplementedError
        else:
            return self.nondist_validation(sample_arch, metric, progress)

    @torch.no_grad()
    def nondist_validation(self, sample_arch, metric, progress):
        metric_evaluator = build_metric(metric, verbose=False)
        val_iterator = tqdm(self.val_loader, desc='evaluate') if progress is True else self.val_loader
        for val_data in val_iterator:
            imgs = val_data['image']
            imgs = imgs.cuda()
            logits = self.supernet({'image': imgs, 'cfg': sample_arch})
            if isinstance(logits, dict):
                logits = logits['level_0']
            logits = tensor2img_batch(logits, val_data['height'], val_data['width'])
            labels = val_data['label']
            metric_evaluator.add_batch(logits, labels)
        return metric_evaluator.get_metric()

    def benchmark_performance(self, arch_Cfg):
        self.supernet.train()

        update_bn_stats(
            self.supernet,
            self.train_bn_loader,
            self.opt['search']['performance_metrics']['finetune_bn_iters'],
            arch_Cfg,
            progress=False)

        self.supernet.eval()
        result = self.validation(arch_Cfg, self.opt['search']['performance_metrics']['metric'], progress=False)
        return result

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        return net
