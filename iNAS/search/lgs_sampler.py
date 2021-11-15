import numpy as np
import random
from copy import deepcopy

from iNAS.archs.supernet_cfg import SupernetCfg
from iNAS.search.resource_calculator import LatencyCalculator
from iNAS.utils import get_root_logger
from iNAS.utils.registry import SAMPLER_REGISTRY


def build_sampler(opt):
    """Build sampler from options.

    Args:
        opt (dict): Configuration. It must constain:
            type (str): Sampler type.
    """
    opt = deepcopy(opt)
    sampler_type = opt.pop('type')
    model = SAMPLER_REGISTRY.get(sampler_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Sampler [{model.__class__.__name__}] is created.')
    return model


@SAMPLER_REGISTRY.register()
class LGS():
    """[summary]

    Args:
        groups ([type]): [description]
        resource_table ([type]): [description]
        device ([type]): [description]
    """

    def __init__(self, groups, resource_table, device):
        self.supernet_cfg = SupernetCfg
        self.latency_groups = groups
        self.latency_calculator = LatencyCalculator(resource_table, device)

    def sampling_backbone_stage(self, sample_rate, stage_name):
        stage_cfg = self.supernet_cfg['backbone_cfg'][stage_name]

        sample_stage_cfg = {}
        if sample_rate == 'min':
            sample_stage_cfg['depth'] = min(stage_cfg['depth'])
        elif sample_rate == 'max':
            sample_stage_cfg['depth'] = max(stage_cfg['depth'])
        else:
            sample_stage_cfg['depth'] = random.choice(stage_cfg['depth'])

        sample_stage_cfg['ratio'] = stage_cfg['ratio']
        sample_stage_cfg['stride'] = stage_cfg['stride']

        filter_array = []
        kernel_size_array = []
        for idx in range(sample_stage_cfg['depth']):
            if sample_rate == 'min':
                filter_array.append(min(stage_cfg['filter']))
                kernel_size_array.append([min(stage_cfg['kernel'])])
            elif sample_rate == 'max':
                filter_array.append(max(stage_cfg['filter']))
                kernel_size_array.append(stage_cfg['kernel'])
            else:
                filter_array.append(random.choice(stage_cfg['filter']))
                fusion_number = random.choice(list(range(1, len(stage_cfg['kernel']) + 1)))
                kernel_size_array.append(sorted(random.sample(stage_cfg['kernel'], k=fusion_number)))

        sample_stage_cfg['kernel'] = kernel_size_array
        sample_stage_cfg['filter'] = filter_array
        return sample_stage_cfg

    def judge_in_group(self, arch_latency, lower_bound, upper_bound, latency_group, percent):
        divide_part = (upper_bound - lower_bound) / latency_group
        arch_group = int(np.floor((arch_latency - lower_bound) / divide_part))
        arch_group = arch_group - 1 if arch_group == latency_group else arch_group

        if arch_group == int(np.floor(latency_group * percent)):
            return True
        else:
            return False

    def get_min_latency(self):
        min_arch = {
            'backbone_cfg': self.uniform_sampling_backbone('min'),
            'transport_cfg': self.uniform_sampling_transport('min'),
            'decoder_cfg': self.uniform_sampling_decoder('min'),
        }
        min_latency = self.latency_calculator.cal_latency(224, min_arch)
        return min_latency['total_latency']

    def get_max_latency(self):
        max_arch = {
            'backbone_cfg': self.uniform_sampling_backbone('max'),
            'transport_cfg': self.uniform_sampling_transport('max'),
            'decoder_cfg': self.uniform_sampling_decoder('max'),
        }
        max_latency = self.latency_calculator.cal_latency(224, max_arch)
        return max_latency['total_latency']

    def sampling_backbone_groups(self, group):
        assert group >= 0 and group < self.latency_groups
        sample_backbone_cfg = {}
        inp_res = 224
        inc = 3
        for stage_name, stage_cfg in self.supernet_cfg['backbone_cfg'].items():
            min_arch = self.sampling_backbone_stage('min', stage_name)
            min_latency = self.latency_calculator.cal_stage_latency(stage_name, min_arch, inc, inp_res)
            max_arch = self.sampling_backbone_stage('max', stage_name)
            max_latency = self.latency_calculator.cal_stage_latency(stage_name, max_arch, inc, inp_res)
            idx = 1
            while True:
                sample_stage = self.sampling_backbone_stage('middle', stage_name)
                arch_latency = self.latency_calculator.cal_stage_latency(stage_name, sample_stage, inc, inp_res)
                if self.judge_in_group(arch_latency, min_latency, max_latency,
                                       self.latency_groups // int(np.ceil(idx / 100)), group / self.latency_groups):
                    break
                else:
                    idx += 1
            if sample_stage['stride'] == 2:
                inp_res = inp_res // 2

            inc = sample_stage['filter'][-1]
            sample_backbone_cfg[stage_name] = sample_stage
        level_channels = {
            'level_0': sample_backbone_cfg['stage_1']['filter'][-1],
            'level_1': sample_backbone_cfg['stage_2']['filter'][-1],
            'level_2': sample_backbone_cfg['stage_3']['filter'][-1],
            'level_3': sample_backbone_cfg['stage_5']['filter'][-1],
            'level_4': sample_backbone_cfg['stage_7']['filter'][-1]
        }

        return sample_backbone_cfg, level_channels

    def uniform_sampling_backbone(self, sample_rate):
        supernet_backbone_cfg = self.supernet_cfg['backbone_cfg']
        sample_backbone_cfg = {}
        for stage_n, stage_cfg in supernet_backbone_cfg.items():
            sample_backbone_cfg[stage_n] = {}
            if sample_rate == 'min':
                sample_backbone_cfg[stage_n]['depth'] = min(stage_cfg['depth'])
            elif sample_rate == 'max':
                sample_backbone_cfg[stage_n]['depth'] = max(stage_cfg['depth'])
            else:
                sample_backbone_cfg[stage_n]['depth'] = random.choice(stage_cfg['depth'])

            sample_backbone_cfg[stage_n]['ratio'] = stage_cfg['ratio']
            sample_backbone_cfg[stage_n]['stride'] = stage_cfg['stride']

            filter_array = []
            kernel_size_array = []
            for idx in range(sample_backbone_cfg[stage_n]['depth']):
                if sample_rate == 'min':
                    filter_array.append(min(stage_cfg['filter']))
                    kernel_size_array.append([min(stage_cfg['kernel'])])
                elif sample_rate == 'max':
                    filter_array.append(max(stage_cfg['filter']))
                    kernel_size_array.append(stage_cfg['kernel'])
                else:
                    filter_array.append(random.choice(stage_cfg['filter']))
                    fusion_number = random.choice(list(range(1, len(stage_cfg['kernel']) + 1)))
                    kernel_size_array.append(sorted(random.sample(stage_cfg['kernel'], k=fusion_number)))

            sample_backbone_cfg[stage_n]['kernel'] = kernel_size_array
            sample_backbone_cfg[stage_n]['filter'] = filter_array
        return sample_backbone_cfg

    def uniform_sampling_transport(self, sample_rate):
        supernet_transport_cfg = self.supernet_cfg['transport_cfg']
        sample_transport_cfg = {}
        for level, level_cfg in supernet_transport_cfg.items():
            sample_transport_cfg[level] = {}
            level_idx = int(level.split('_')[1])
            if sample_rate == 'min':
                choose_idx_list = []
                kernel = [min(level_cfg['kernel'])]
            elif sample_rate == 'max':
                choose_idx_list = level_cfg['connection']
                kernel = level_cfg['kernel']
            else:
                choose_idx_list = random.sample(
                    level_cfg['connection'], k=random.randint(0, len(level_cfg['connection'])))
                fusion_number = random.choice(list(range(1, len(level_cfg['kernel']) + 1)))
                kernel = sorted(random.sample(level_cfg['kernel'], k=fusion_number))

            if level_idx not in choose_idx_list:
                choose_idx_list.append(level_idx)
            sample_transport_cfg[level]['connection'] = choose_idx_list
            sample_transport_cfg[level]['kernel'] = kernel
        return sample_transport_cfg

    def sampling_transport_group(self, group, level_channels_dict):
        min_arch, max_arch = self.uniform_sampling_transport('min'), self.uniform_sampling_transport('max')
        min_latency, max_latency = self.latency_calculator.cal_latency_transport(
            224, min_arch,
            level_channels_dict), self.latency_calculator.cal_latency_transport(224, max_arch, level_channels_dict)

        while True:
            sample_transport = self.uniform_sampling_transport('middle')
            arch_latency = self.latency_calculator.cal_latency_transport(224, sample_transport, level_channels_dict)
            if self.judge_in_group(arch_latency, min_latency, max_latency, self.latency_groups,
                                   group / self.latency_groups):
                break
        return sample_transport

    def uniform_sampling_decoder(self, sample_rate):
        supernet_decoder_cfg = self.supernet_cfg['decoder_cfg']
        sample_decoder_cfg = {}
        for level, level_cfg in supernet_decoder_cfg.items():
            sample_decoder_cfg[level] = {}
            level_idx = int(level.split('_')[1])
            if sample_rate == 'min':
                choose_idx_list = []
                kernel = [min(level_cfg['kernel'])]
            elif sample_rate == 'max':
                choose_idx_list = level_cfg['connection']
                kernel = level_cfg['kernel']
            else:
                choose_idx_list = random.sample(
                    level_cfg['connection'], k=random.randint(0, len(level_cfg['connection'])))
                fusion_number = random.choice(list(range(1, len(level_cfg['kernel']) + 1)))
                kernel = sorted(random.sample(level_cfg['kernel'], k=fusion_number))

            if level_idx + 1 not in choose_idx_list and level != 'level_4':
                choose_idx_list.append(level_idx + 1)

            if level_idx not in choose_idx_list:
                choose_idx_list.append(level_idx)

            sample_decoder_cfg[level]['connection'] = choose_idx_list
            sample_decoder_cfg[level]['kernel'] = kernel
        return sample_decoder_cfg

    def sampling_decoder_group(self, group, level_channels_dict):
        min_arch, max_arch = self.uniform_sampling_decoder('min'), self.uniform_sampling_decoder('max')
        min_latency = self.latency_calculator.cal_latency_decoder(224, min_arch, level_channels_dict)
        max_latency = self.latency_calculator.cal_latency_decoder(224, max_arch, level_channels_dict)
        while True:
            sample_decoder = self.uniform_sampling_decoder('middle')
            arch_latency = self.latency_calculator.cal_latency_decoder(224, sample_decoder, level_channels_dict)
            if self.judge_in_group(arch_latency, min_latency, max_latency, self.latency_groups,
                                   group / self.latency_groups):
                break
        return sample_decoder

    def sample(self, group):
        backbone_cfg, level_channels_dict = self.sampling_backbone_groups(group)
        transport_cfg = self.sampling_transport_group(group, level_channels_dict)
        decoder_cfg = self.sampling_decoder_group(group, level_channels_dict)
        return {'backbone_cfg': backbone_cfg, 'transport_cfg': transport_cfg, 'decoder_cfg': decoder_cfg}
