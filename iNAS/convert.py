import logging
import numpy as np
import random
import torch
import torch.nn as nn
from os import path as osp
from tqdm import tqdm

from iNAS.archs import build_network
from iNAS.data import build_dataloader, build_dataset
from iNAS.metrics import build_metric
from iNAS.utils import (check_resume, get_env_info, get_root_logger, get_time_str, init_tb_logger, init_wandb_logger,
                        make_exp_dirs, mkdir_and_rename, scandir, tensor2img_batch)
from iNAS.utils.bn_utils import update_bn_stats
from iNAS.utils.options import dict2str, parse_options
from iNAS.utils.resource_benchmark_util import get_net_Gflops_Mparams


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_bn_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = build_dataset(dataset_opt)

            train_bn_loader = build_dataloader(
                train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])

        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: ' f'{len(val_set)}')

    return train_bn_loader, val_loader


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def set_random_seed(opt):
    torch.manual_seed(opt['manual_seed'])
    torch.cuda.manual_seed(opt['manual_seed'])
    np.random.seed(opt['manual_seed'])
    random.seed(opt['manual_seed'])


@torch.no_grad()
def evaluate(opt, network, sample_arch, dataloader):
    metric_results = {metric: 0 for metric in opt['val']['metrics'].keys()}
    metric_evaluator = {metric: build_metric(metric) for metric in opt['val']['metrics'].keys()}

    for idx, val_data in enumerate(tqdm(dataloader)):
        imgs = val_data['image']
        imgs = imgs.cuda()
        logits = network({'image': imgs, 'cfg': sample_arch})
        if isinstance(logits, dict):
            logits = logits['level_0']
        logits = tensor2img_batch(logits, val_data['height'], val_data['width'])
        for metric in metric_evaluator.keys():
            metric_evaluator[metric].add_batch(logits, val_data['label'])
    for metric in metric_results.keys():
        metric_results[metric] = metric_evaluator[metric].get_metric()
    return metric_results


def convert_deployment(root_path):
    """[summary]

    Args:
        root_path ([type]): [description]
    """
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    set_random_seed(opt)

    # load resume states if necessary
    resume_state = load_resume_state(opt)

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")

    logger = get_root_logger(logger_name='iNAS', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_bn_loader, val_loader = result

    supernet = build_network(opt['supernet']).cuda()
    standalone = build_network(opt['standalone']).cuda()

    logger.info('Step 0: load superent!')
    supernet.load_state_dict(
        torch.load(opt['path']['supernet_path'])['params'], strict=opt['path'].get('strict_load', True))

    logger.info('Step 1: finetune bn in superent!')
    update_bn_stats(supernet, train_bn_loader, opt['val']['finetune_bn_iters'], standalone.cfg)

    logger.info('Step 2: Eval without reparams')
    supernet.train()
    for name, module in supernet.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    result = evaluate(opt, supernet, standalone.cfg, val_loader)
    logger.info('Performance: %s' % str(result))

    # step3
    logger.info('Step 3: Eval with reparams')
    supernet.eval()
    result = evaluate(opt, supernet, standalone.cfg, val_loader)
    logger.info('Performance: %s' % str(result))

    # step4
    logger.info('Step 4: load statedict into standalone model')
    standalone.load_supernet_weight(supernet.state_dict())
    standalone.eval()
    result = evaluate(opt, standalone, standalone.cfg, val_loader)
    logger.info('Performance: %s' % str(result))
    # step4
    logger.info('Step 5: save stand_alone model dict')
    torch.save(standalone.state_dict(), opt['standalone']['cfg'].replace('.json', '.pth'))

    logger.info('Step 6: benchmark flops and params')
    resource_result = get_net_Gflops_Mparams(standalone, 224)
    logger.info(resource_result)


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    convert_deployment(root_path)
