import json
import numpy as np
import os
import random
from copy import deepcopy
from operator import itemgetter
from tqdm import tqdm

from iNAS.archs import build_network
from iNAS.search.lgs_sampler import build_sampler
from iNAS.search.performance_calculator import build_performance
from iNAS.search.resource_calculator import build_resource
from iNAS.utils import get_net_Gflops_Mparams
from iNAS.utils.registry import SEARCH_REGISTRY


@SEARCH_REGISTRY.register()
class EvolutionSearch():
    """[summary]

    Args:
        opt ([type]): [description]
    """

    def __init__(self, opt):
        self.opt = opt
        self.performance_evaluator = build_performance(opt)
        self.resource_calculator = build_resource(opt['search']['resource_metrics'])
        self.sampler = build_sampler(opt['search']['sampler'])

    def benchmark_flops_parmas(self, arch_Cfg):
        standalone_cfg = arch_Cfg['standalone']
        standalone_cfg['cfg'] = arch_Cfg
        standalone = build_network(standalone_cfg)
        return get_net_Gflops_Mparams(standalone, 224)

    def _add_information(self, arch_dict):
        for _, arch_dict in tqdm(arch_dict.items()):
            if self.opt['search']['resource_metrics']['device'] not in arch_dict:
                latency = '%.4f' % self.resource_calculator.cal_latency(224, eval(
                    arch_dict['arch_cfg']))['total_latency']
                arch_dict[self.opt['search']['resource_metrics']['device']] = latency

            if self.opt['search']['performance_metrics']['metric'] not in arch_dict:
                performance = '%.4f' % self.performance_evaluator.benchmark_performance(eval(arch_dict['arch_cfg']))
                arch_dict[self.opt['search']['performance_metrics']['metric']] = performance

    def add_information(self, arch_dict):
        self._add_information(arch_dict)
        return arch_dict

    def initialization(self):
        arch_dict = self._initialization()
        arch_dict = self.add_information(arch_dict)
        return arch_dict

    def _initialization(self):
        arch_dict = {}
        groups = self.sampler.latency_groups
        init_populations = self.opt['search']['evolution_cfg']['init_populations']
        populations_per_group = init_populations // groups
        for group in range(0, self.sampler.latency_groups):
            for _ in tqdm(range(populations_per_group), desc='sampling_group_%d' % group):
                arch_cfg = self.sampler.sample(group)
                arch_dict['arch_%d' % len(arch_dict)] = {'arch_cfg': str(arch_cfg)}
        return arch_dict

    def save_dict(self, iter, arch_dict):
        with open(os.path.join(self.opt['path']['models'], 'population_iter_%d.json' % iter), 'w') as fw:
            json.dump(arch_dict, fw)

    def load_dict(self, filename):
        with open(os.path.join(self.opt['path']['models'], filename), 'r') as fr:
            arch_dict = json.load(fr)
        return arch_dict

    def merge_dict(self, first_dict, second_dict):
        new_dict = deepcopy(first_dict)
        for k, v in second_dict.items():
            new_dict['arch_%d' % len(new_dict)] = v
        return new_dict

    def ParetoTwoDimensions(self, data):
        sorted_data = sorted(data, key=itemgetter(0, 1), reverse=False)
        assert data == sorted_data
        pareto_idx = list()
        pareto_idx.append(0)
        cut_off = sorted_data[0][1]
        for i in range(1, len(sorted_data)):
            if sorted_data[i][1] > cut_off:
                pareto_idx.append(i)
                cut_off = sorted_data[i][1]
        return pareto_idx

    def get_final_results(self, sample_dict):
        sample_idx = []
        sample_latency = []
        sample_accuracy = []
        device = self.opt['search']['resource_metrics']['device']
        metric = self.opt['search']['performance_metrics']['metric']
        for k, cfg_di in sample_dict.items():
            sample_idx.append(int(k.split('_')[1]))
            sample_latency.append(float(cfg_di[device]))
            sample_accuracy.append(float(cfg_di[metric]))
        xy = list(zip(sample_latency, sample_accuracy, sample_idx))
        sorted_xy = sorted(xy, key=itemgetter(0, 1), reverse=False)
        pareto_idx = self.ParetoTwoDimensions(sorted_xy)
        arr = np.array(sorted_xy)
        arr_pareto = arr[pareto_idx]
        final_result_dir = os.path.join(self.opt['path']['models'], f'{device}_search')
        os.makedirs(final_result_dir, exist_ok=True)
        for pareto_point in arr_pareto:
            _, _, idx = pareto_point
            arch_name = 'arch_%d' % (int(idx))
            arch_cfg = sample_dict[arch_name]['arch_cfg']
            arch_lat = float(sample_dict[arch_name][device])
            arch_performance = float(sample_dict[arch_name][metric])
            arch_save_name = '%s_lat@%.2fms_%s@%.4f.json' % (device, arch_lat, metric, arch_performance)
            with open(os.path.join(final_result_dir, arch_save_name), 'w') as fw:
                json.dump(arch_cfg, fw)

    def select_pareto_frontier(self, sample_dict):
        group_cnt = np.ones(self.sampler.latency_groups) * (
            self.opt['search']['evolution_cfg']['selection_num'] // self.sampler.latency_groups)

        sample_dict = deepcopy(sample_dict)
        remain_dict = {}

        min_latency = self.sampler.get_min_latency()
        max_latency = self.sampler.get_max_latency()
        divide_part = (max_latency - min_latency) / self.sampler.latency_groups

        while len(remain_dict) < self.opt['search']['evolution_cfg']['selection_num']:
            sample_idx = []
            sample_latency = []
            sample_accuracy = []
            for k, cfg_di in sample_dict.items():
                sample_idx.append(int(k.split('_')[1]))
                sample_latency.append(float(cfg_di[self.opt['search']['resource_metrics']['device']]))
                sample_accuracy.append(float(cfg_di[self.opt['search']['performance_metrics']['metric']]))
            xy = list(zip(sample_latency, sample_accuracy, sample_idx))
            sorted_xy = sorted(xy, key=itemgetter(0, 1), reverse=False)
            pareto_idx = self.ParetoTwoDimensions(sorted_xy)
            arr = np.array(sorted_xy)
            arr_pareto = arr[pareto_idx]

            for pareto_point in arr_pareto:
                _, _, idx = pareto_point
                arch_name = 'arch_%d' % (int(idx))
                cfg_group = int(
                    np.floor(
                        (eval(sample_dict[arch_name][self.opt['search']['resource_metrics']['device']]) - min_latency) /
                        divide_part))
                if cfg_group > self.sampler.latency_groups - 1:
                    cfg_group = self.sampler.latency_groups - 1
                if group_cnt[cfg_group] > 0:
                    group_cnt[cfg_group] -= 1
                    remain_dict['arch_%d' % len(remain_dict)] = sample_dict[arch_name]
                del sample_dict[arch_name]

                if len(remain_dict) >= self.opt['search']['evolution_cfg']['selection_num']:
                    break
        return remain_dict

    def crossover_mutation(self, sample_dict):
        new_sample_dict = {}
        keys = list(sample_dict.keys())

        for arch_name, arch_cfg in sample_dict.items():
            arch_k = random.choice(keys)
            while arch_k == arch_name:
                arch_k = random.choice(keys)
            cfg_this = eval(arch_cfg['arch_cfg'])
            cfg_swap = eval(sample_dict[arch_k]['arch_cfg'])
            cfg_crossover = self.swap_stage(cfg_this, cfg_swap)
            cfg_mutation = self.Mutation(cfg_crossover)
            new_sample_dict['arch_%d' % len(new_sample_dict)] = {'arch_cfg': str(cfg_mutation)}
        return new_sample_dict

    def swap_stage(self, cfg_this, cfg_swap):
        new_cfg = {'backbone_cfg': {}, 'transport_cfg': {}, 'decoder_cfg': {}}
        while True:
            for stage_name, stage_cfg in cfg_this['backbone_cfg'].items():
                if random.random() < self.opt['search']['evolution_cfg']['crossover_prob']:
                    new_cfg['backbone_cfg'][stage_name] = cfg_swap['backbone_cfg'][stage_name]
                else:
                    new_cfg['backbone_cfg'][stage_name] = stage_cfg

            for level_name, level_cfg in cfg_this['transport_cfg'].items():
                if random.random() < self.opt['search']['evolution_cfg']['crossover_prob']:
                    new_cfg['transport_cfg'][level_name] = cfg_swap['transport_cfg'][level_name]
                else:
                    new_cfg['transport_cfg'][level_name] = level_cfg

            for level_name, level_cfg in cfg_this['decoder_cfg'].items():
                if random.random() < self.opt['search']['evolution_cfg']['crossover_prob']:
                    new_cfg['decoder_cfg'][level_name] = cfg_swap['decoder_cfg'][level_name]
                else:
                    new_cfg['decoder_cfg'][level_name] = level_cfg
            if str(new_cfg) != str(cfg_this):
                break
        return new_cfg

    def Mutation(self, cfg):
        new_cfg = {'backbone_cfg': {}, 'transport_cfg': {}, 'decoder_cfg': {}}
        supernet_cfg = self.sampler.supernet_cfg

        for stage_name, stage_cfg in cfg['backbone_cfg'].items():
            search_space = supernet_cfg['backbone_cfg'][stage_name]
            kernel_array = []
            filter_array = []
            for kernel, filter in zip(stage_cfg['kernel'], stage_cfg['filter']):
                if random.random() < self.opt['search']['evolution_cfg']['mutation_prob']:
                    fusion_number = random.choice(list(range(1, len(search_space['kernel']) + 1)))
                    kernel_comb = sorted(random.sample(search_space['kernel'], k=fusion_number))
                    kernel_array.append(kernel_comb)
                else:
                    kernel_array.append(kernel)

                if random.random() < self.opt['search']['evolution_cfg']['mutation_prob']:
                    filter_array.append(random.choice(search_space['filter']))
                else:
                    filter_array.append(filter)
            stage_cfg['kernel'] = kernel_array
            stage_cfg['filter'] = filter_array
            new_cfg['backbone_cfg'][stage_name] = stage_cfg

        for level_name, level_cfg in cfg['transport_cfg'].items():
            search_space = supernet_cfg['transport_cfg'][level_name]
            level_idx = int(level_name.split('_')[1])

            if random.random() < self.opt['search']['evolution_cfg']['mutation_prob']:
                choose_idx_list = random.sample(
                    search_space['connection'], k=random.randint(0, len(search_space['connection'])))
                if level_idx not in choose_idx_list:
                    choose_idx_list.append(level_idx)
                new_cfg['transport_cfg'][level_name] = {'connection': choose_idx_list}
            else:
                new_cfg['transport_cfg'][level_name] = {'connection': level_cfg['connection']}

            if random.random() < self.opt['search']['evolution_cfg']['mutation_prob']:
                fusion_number = random.choice(list(range(1, len(search_space['kernel']) + 1)))
                kernel_comb = sorted(random.sample(search_space['kernel'], k=fusion_number))
                new_cfg['transport_cfg'][level_name]['kernel'] = kernel_comb
            else:
                new_cfg['transport_cfg'][level_name]['kernel'] = level_cfg['kernel']

        for level_name, level_cfg in cfg['decoder_cfg'].items():
            search_space = supernet_cfg['decoder_cfg'][level_name]
            level_idx = int(level_name.split('_')[1])

            if random.random() < self.opt['search']['evolution_cfg']['mutation_prob']:
                choose_idx_list = random.sample(
                    search_space['connection'], k=random.randint(0, len(search_space['connection'])))

                if level_idx + 1 not in choose_idx_list and level_name != 'level_4':
                    choose_idx_list.append(level_idx + 1)

                if level_idx not in choose_idx_list:
                    choose_idx_list.append(level_idx)
                new_cfg['decoder_cfg'][level_name] = {'connection': choose_idx_list}
            else:
                new_cfg['decoder_cfg'][level_name] = {'connection': level_cfg['connection']}

            if random.random() < self.opt['search']['evolution_cfg']['mutation_prob']:
                fusion_number = random.choice(list(range(1, len(search_space['kernel']) + 1)))
                kernel_comb = sorted(random.sample(search_space['kernel'], k=fusion_number))
                new_cfg['decoder_cfg'][level_name]['kernel'] = kernel_comb
            else:
                new_cfg['decoder_cfg'][level_name]['kernel'] = level_cfg['kernel']

        return new_cfg
