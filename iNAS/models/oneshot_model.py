import os.path as osp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm

from iNAS.archs import build_network
from iNAS.losses import build_loss
from iNAS.metrics import build_metric
from iNAS.models.base_model import BaseModel
from iNAS.utils import batch_resize, get_root_logger, imwrite, tensor2img_batch
from iNAS.utils.bn_utils import update_bn_stats
from iNAS.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class OneShotModel(BaseModel):
    """[summary]

    Args:
        opt ([type]): [description]
    """

    def __init__(self, opt):
        super(OneShotModel, self).__init__(opt)

        # define network
        self.supernet = build_network(opt['supernet'])
        if self.opt['train'].get('sync_bn', False):
            self.supernet = nn.SyncBatchNorm.convert_sync_batchnorm(self.supernet)
        self.supernet = self.model_to_device(self.supernet)
        self.print_network(self.supernet)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key', 'params')
            self.load_network(self.supernet, load_path, self.opt['path'].get('strict_load', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.supernet.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('main_opt'):
            self.main_opt = build_loss(train_opt['main_opt']).to(self.device)
        else:
            self.main_opt = None

        if train_opt.get('distill_opt'):
            self.distill_opt = build_loss(train_opt['distill_opt']).to(self.device)
        else:
            self.distill_opt = None

        if self.main_opt is None:
            raise ValueError('main_opt is None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        backbone_params_list, head_params_list = [], []

        for k, v in self.supernet.named_parameters():
            if v.requires_grad:
                if 'backbone' in k:
                    backbone_params_list.append(v)
                else:
                    head_params_list.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim'].pop('type')
        head_lr_mul = train_opt['optim'].pop('head_lr_mul')

        params_list = [
            {
                'params': backbone_params_list,
            },
            {
                'params': head_params_list,
                'lr': train_opt['optim']['lr'] * head_lr_mul
            },
        ]

        self.optimizer_g = self.get_optimizer(optim_type, params_list, **train_opt['optim'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.image = data['image'].to(self.device)
        if 'label' in data:
            self.label = data['label'].to(self.device)

    def multiscale_transform(self, img, label, sizes):
        size = random.choice(sizes)
        img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        label = F.interpolate(label.unsqueeze(1).float(), size=size, mode='nearest')
        return img, label

    def optimize_parameters(self, current_iter, arch_dict):
        self.optimizer_g.zero_grad()
        loss_dict = OrderedDict()
        teacher_pred = None

        if self.opt['train'].get('multi_scale_training', False):
            self.image, self.label = self.multiscale_transform(
                self.image, self.label, sizes=[9 * 32, 10 * 32, 11 * 32, 12 * 32])

        for idx, (sample_rate, sample_arch) in enumerate(arch_dict.items()):
            inp_dict = {'image': self.image, 'cfg': sample_arch}
            results = self.supernet(inp_dict)

            if teacher_pred is None:
                loss = self.main_opt(results, self.label)
            else:
                loss = self.distill_opt(results, teacher_pred)

            loss_dict['l_%s' % sample_rate] = loss.item()

            if sample_rate == 'max' and self.distill_opt is not None:
                teacher_pred = results  # not detach

            if idx == len(arch_dict) - 1:
                retain_graph = False
            else:
                retain_graph = True

            loss.backward(retain_graph=retain_graph)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def validation(self, dataloader, train_bn_loader, current_iter, tb_logger, arch_dict, save_img=False):
        """[summary]

        Args:
            dataloader ([type]): [description]
            train_bn_loader ([type]): [description]
            current_iter ([type]): [description]
            tb_logger ([type]): [description]
            arch_dict ([type]): [description]
            save_img (bool, optional): [description]. Defaults to False.
        """
        logger = get_root_logger()

        state_dict = self.supernet.module.state_dict()

        for sample_rate, sample_arch in arch_dict.items():
            logger.info('evaluate net_%s' % sample_rate)

            if self.opt['dist']:
                self.dist_validation(dataloader, train_bn_loader, current_iter, tb_logger, sample_arch, save_img)
            else:
                self.nondist_validation(dataloader, train_bn_loader, current_iter, tb_logger, sample_arch, save_img)

        self.supernet.module.load_state_dict(state_dict)

    def dist_validation(self, dataloader, train_bn_loader, current_iter, tb_logger, sample_arch, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, train_bn_loader, current_iter, tb_logger, sample_arch, save_img)

    @torch.no_grad()
    def nondist_validation(self, dataloader, train_bn_loader, current_iter, tb_logger, sample_arch, save_img):
        with_metrics = self.opt['val'].get('metrics') is not None
        dataset_name = self.opt['datasets']['val']['name']

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self.metric_evaluator = {metric: build_metric(metric) for metric in self.opt['val']['metrics'].keys()}

        # step1: precise bn
        update_bn_stats(self.supernet, train_bn_loader, self.opt['val']['finetune_bn_iters'], sample_arch)

        self.supernet.eval()
        for _, val_data in enumerate(tqdm(dataloader)):
            img_names = [osp.splitext(osp.basename(name))[0] for name in val_data['name']]
            imgs = val_data['image']
            imgs = imgs.cuda()
            logits = self.supernet({'image': imgs, 'cfg': sample_arch})
            if self.opt['val']['output_transform'] == 'sod':
                logits = tensor2img_batch(logits, val_data['height'], val_data['width'])
            elif self.opt['val']['output_transform'] == 'semseg':
                logits = batch_resize(logits, val_data['height'], val_data['width'])
            else:
                raise NotImplementedError

            if save_img:
                if self.opt['val']['output_transform'] == 'semseg':
                    raise NotImplementedError
                for logit, img_name in zip(logits, img_names):
                    save_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                    imwrite(logit, save_path)
            for metric in self.metric_evaluator.keys():
                self.metric_evaluator[metric].add_batch(logits, val_data['label'])
        self.supernet.train()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] = self.metric_evaluator[metric].get_metric()
            self._log_validation_metric_values(current_iter, self.opt['datasets']['val']['name'], tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        self.save_network(self.supernet, 'supernet', current_iter)
        self.save_training_state(epoch, current_iter)
