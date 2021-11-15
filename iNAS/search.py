import json
import logging
import numpy as np
import random
import torch
from os import path as osp

from iNAS.search import build_search
from iNAS.utils import (get_env_info, get_root_logger, get_time_str, init_tb_logger, init_wandb_logger, make_exp_dirs,
                        mkdir_and_rename, scandir)
from iNAS.utils.options import dict2str, parse_options


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


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'models')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='', recursive=False, full_path=False))
            if len(states) != 0:
                states = [int(v.split('.json')[0].split('_')[-1]) for v in states]
                resume_state_path = osp.join(state_path, f'population_iter_{max(states)}.json')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        with open(resume_state_path, 'r') as fr:
            json_dict = json.load(fr)
        iter = int(resume_state_path.split('.json')[0].split('_')[-1])
        resume_state = {'json_dict': json_dict, 'iter': iter}
    return resume_state


def set_random_seed(opt):
    torch.manual_seed(opt['manual_seed'])
    torch.cuda.manual_seed(opt['manual_seed'])
    np.random.seed(opt['manual_seed'])
    random.seed(opt['manual_seed'])


def search_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    set_random_seed(opt)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

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

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create model
    if resume_state:  # resume training
        search = build_search(opt)
        logger.info(f'Resuming training from ' f"iter: {resume_state['iter']}.")
        population = resume_state['json_dict']
        current_iter = int(resume_state['iter']) + 1
    else:
        search = build_search(opt)
        current_iter = 0

    total_iter = opt['search']['total_iter']

    # training
    logger.info('Start searching:')

    for iter in range(current_iter, total_iter):

        if iter == 0:
            population = search.initialization()
        else:
            pareto_population = search.select_pareto_frontier(population)  # select pareto
            pareto_population = search.crossover_mutation(pareto_population)
            pareto_population = search.add_information(pareto_population)
            population = search.merge_dict(population, pareto_population)

        if iter % opt['logger']['print_freq'] == 0:
            logger.info('Finish searching iter_%d.' % iter)

        # save models and training states
        if iter % opt['logger']['save_checkpoint_freq'] == 0:
            logger.info('Saving searching states.')
            search.save_dict(iter, population)

    logger.info('Deriving_searched_results.')
    search.get_final_results(population)

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    search_pipeline(root_path)
