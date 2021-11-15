import torch.nn as nn
from collections import OrderedDict

from iNAS.archs.iNAS_standalone.iNAS_util import Conv2d, DecoderFusion, DecoderUpsample, Identity


class Decoder(nn.Module):
    """Stand-alone decoder.

    Args:
        cfg (dict): Defines the decoder configuration.
        level_channel_list ([type]): [description]
    """

    def __init__(self, cfg, level_channel_dict, num_classes, activation, output_level_ids):
        super(Decoder, self).__init__()
        self.level_channel_dict = level_channel_dict
        self.cfg = cfg

        self.num_classes = num_classes

        assert activation in ['sigmoid', 'none', 'softmax'], 'activation is not support in this version'
        if activation == 'sigmoid':
            act_class = nn.Sigmoid()
        elif activation == 'softmax':
            act_class = nn.Softmax(dim=1)
        else:
            act_class = nn.Identity()

        self.output_level_ids = eval(output_level_ids)

        self.mainops = nn.ModuleDict()
        for level_name, level_cfg in cfg.items():
            channels = self.level_channel_dict[level_name]
            kernel_size_array = level_cfg['kernel']
            self.mainops[level_name] = DecoderFusion(channels=channels, kernel_size_array=kernel_size_array)

        self.branches = nn.ModuleDict()
        for level_name, level_cfg in cfg.items():
            connect_list = level_cfg['connection']
            level_idx = int(level_name.split('_')[1])
            self.branches[level_name] = nn.ModuleDict()
            for connect_idx in connect_list:
                if level_idx == connect_idx:
                    self.branches[level_name][str(connect_idx)] = Identity()
                elif level_idx < connect_idx:
                    self.branches[level_name][str(connect_idx)] = DecoderUpsample(
                        connect_idx - level_idx,
                        in_channels=self.level_channel_dict['level_%d' % connect_idx],
                        out_channels=self.level_channel_dict['level_%d' % level_idx])

        self.decoder_dict = nn.ModuleDict()
        for level_name, channels in level_channel_dict.items():
            level_idx = int(level_name.split('_')[1])
            if level_idx in self.output_level_ids:
                self.decoder_dict[level_name] = nn.Sequential(
                    OrderedDict([
                        ('conv', Conv2d(channels, num_classes, kernel_size=1)),
                        ('act', act_class),
                        ('upsample', nn.UpsamplingBilinear2d(scale_factor=2**(level_idx + 1))),
                    ]))

    def forward(self, x):
        cfg = self.cfg

        decoder_order = ['level_4', 'level_3', 'level_2', 'level_1', 'level_0']

        for level_name in decoder_order:
            cell_features = []

            connection_list = cfg[level_name]['connection']
            for connect_idx in connection_list:
                up_feature = self.branches[level_name][str(connect_idx)](x['level_%d' % connect_idx])
                cell_features.append(up_feature)

            cell_op = self.mainops[level_name]

            cell_features = cell_op(cell_features)
            x[level_name] = cell_features

        if self.training is True:
            for level_name, decode_func in self.decoder_dict.items():
                x[level_name] = decode_func(x[level_name])
            return x
        else:
            f = self.decoder_dict['level_0'](x['level_0'])
            return f
