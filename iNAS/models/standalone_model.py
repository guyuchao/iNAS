import os.path as osp
import torch
from tqdm import tqdm

from iNAS.archs import build_network
from iNAS.metrics import build_metric
from iNAS.models.base_model import BaseModel
from iNAS.utils import get_root_logger, imwrite, tensor2img_batch
from iNAS.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class StandAloneModel(BaseModel):
    """[summary]

    Args:
        opt ([type]): [description]
    """

    def __init__(self, opt):
        super(StandAloneModel, self).__init__(opt)

        # define network
        self.standalone = build_network(opt['standalone'])
        self.standalone.reparams()
        self.standalone = self.model_to_device(self.standalone)
        self.print_network(self.standalone)

        # load pretrained models
        load_path = self.opt['path'].get('standalone_path', None)

        if load_path is not None:
            param_key = self.opt['path'].get('param_key', None)
            self.load_network(self.standalone, load_path, self.opt['path'].get('strict_load', True), param_key)

    @torch.no_grad()
    def validation(self, dataloader, current_iter, dst_opt, tb_logger, save_img=False):
        """[summary]

        Args:
            dataloader ([type]): [description]
            current_iter ([type]): [description]
            dst_opt ([type]): [description]
            tb_logger ([type]): [description]
            save_img (bool, optional): [description]. Defaults to False.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, dst_opt, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, dst_opt, tb_logger, save_img)

    @torch.no_grad()
    def dist_validation(self, dataloader, current_iter, dst_opt, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, dst_opt, tb_logger, save_img)

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, dst_opt, tb_logger, save_img):
        with_metrics = self.opt['val'].get('metrics') is not None
        dataset_name = dataloader.dataset.opt['name']

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self.metric_evaluator = {metric: build_metric(metric) for metric in self.opt['val']['metrics'].keys()}

        self.standalone.eval()
        for idx, val_data in enumerate(tqdm(dataloader)):
            img_names = [osp.splitext(osp.basename(name))[0] for name in val_data['name']]
            imgs = val_data['image']
            imgs = imgs.cuda()
            logits = self.standalone({'image': imgs})
            logits = tensor2img_batch(logits, val_data['height'], val_data['width'])
            if save_img:
                for logit, img_name in zip(logits, img_names):
                    save_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                    imwrite(logit, save_path)

            for metric in self.metric_evaluator.keys():
                self.metric_evaluator[metric].add_batch(logits, val_data['label'])

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] = self.metric_evaluator[metric].get_metric()
            self._log_validation_metric_values(current_iter, dst_opt['name'], tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
