import torch.nn as nn

from iNAS.archs.iNAS_supernet.iNAS_util import (DynamicTransportDownsample, DynamicTransportFusion,
                                                DynamicTransportUpsample, Identity)


class DynamicTransport(nn.Module):
    """[summary]

    Args:
        cfg ([type]): [description]
        level_channel_list ([type]): [description]
    """

    def __init__(self, cfg, level_channel_list):
        super(DynamicTransport, self).__init__()
        self.cfg = cfg
        self.level_channels_list = level_channel_list
        self.layers = len(cfg)
        self.mainops = nn.ModuleDict()

        for level_name, level_cfg in cfg.items():
            channels = self.level_channels_list[level_name]
            max_kernel_size_array = level_cfg['kernel']
            self.mainops[level_name] =\
                DynamicTransportFusion(max_in_channels=channels, max_kernel_size_array=max_kernel_size_array)

        self.branches = nn.ModuleDict()
        for level_name, level_cfg in cfg.items():
            connect_list = level_cfg['connection']
            level_idx = int(level_name.split('_')[1])
            self.branches[level_name] = nn.ModuleDict()
            for connect_idx in connect_list:
                if level_idx == connect_idx:
                    self.branches[level_name][str(connect_idx)] = Identity()
                elif level_idx < connect_idx:
                    # upsample
                    self.branches[level_name][str(connect_idx)] = DynamicTransportUpsample(
                        connect_idx - level_idx,
                        max_in_channels=self.level_channels_list['level_%d' % connect_idx],
                        max_out_channels=self.level_channels_list['level_%d' % level_idx])
                else:
                    self.branches[level_name][str(connect_idx)] = DynamicTransportDownsample(
                        level_idx - connect_idx,
                        max_in_channels=self.level_channels_list['level_%d' % connect_idx],
                        max_out_channels=self.level_channels_list['level_%d' % level_idx])

    def forward(self, x, cfg=None):
        if cfg is None:
            cfg = self.cfg

        layer_oup = {'level_0': None, 'level_1': None, 'level_2': None, 'level_3': None, 'level_4': None}

        level_channels_list = {}
        for level_name in x.keys():
            level_channels_list[level_name] = x[level_name].shape[1]

        for level_name, level_cfg in cfg.items():
            connect_list = level_cfg['connection']

            level_idx = int(level_name.split('_')[1])
            cell_features = []

            for connect_idx in connect_list:
                branchop = self.branches[level_name][str(connect_idx)]
                feature = branchop(x['level_%d' % connect_idx], level_channels_list[level_name])
                cell_features.append(feature)

            level_features = self.mainops[level_name](cell_features, cfg[level_name]['kernel'])

            layer_oup['level_%d' % level_idx] = level_features

        return layer_oup
