import json
import random

from iNAS.archs.supernet_cfg import SupernetCfg


class Uniform_Architecture_Sampler:
    """[summary]
    """

    def __init__(self):
        self.supernet_cfg = SupernetCfg

    def sampling_backbone(self, sample_rate):
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
            for _ in range(sample_backbone_cfg[stage_n]['depth']):
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

    def sampling_transport(self, sample_rate):
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

    def sampling_decoder(self, sample_rate):
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

    def sampling(self, sample_rate='max'):
        assert sample_rate in ['min', 'max'] or 'middle' in sample_rate

        backbone_cfg = self.sampling_backbone(sample_rate)
        transport_cfg = self.sampling_transport(sample_rate)
        decoder_cfg = self.sampling_decoder(sample_rate)
        return {'backbone_cfg': backbone_cfg, 'transport_cfg': transport_cfg, 'decoder_cfg': decoder_cfg}

    def save_arch(self, arch_cfg, path):
        with open(path, 'w') as fw:
            json.dump(str(arch_cfg), fw)

    def load_arch(self, path):
        with open(path, 'r') as fr:
            arch_cfg = json.load(fr)
            arch_cfg = eval(arch_cfg)
        return arch_cfg
