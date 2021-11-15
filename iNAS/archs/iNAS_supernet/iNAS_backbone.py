import torch.nn as nn
from collections import OrderedDict

from iNAS.archs.iNAS_supernet.iNAS_util import SMSU, DynamicBatchNorm2d, DynamicConv2d


class DynamicSE(nn.Module):
    """[summary]

    Args:
        max_channels ([type]): [description]
    """
    REDUCTION = 4

    def __init__(self, max_channels):
        super(DynamicSE, self).__init__()

        self.max_channels = max_channels
        self.reduction = DynamicSE.REDUCTION

        num_mid = self.max_channels // self.reduction

        self.fc = nn.Sequential(
            OrderedDict([
                ('reduce', DynamicConv2d(self.max_channels, num_mid, 1, 1, 0, bias=True)),
                ('relu', nn.ReLU(inplace=True)),
                ('expand', DynamicConv2d(num_mid, self.max_channels, 1, 1, 0, bias=True)),
                ('h_sigmoid', nn.Hardsigmoid(inplace=True)),
            ]))

    def forward(self, x):
        in_channel = x.size(1)
        num_mid = in_channel // self.reduction

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        self.fc.reduce.active_out_channel = num_mid
        self.fc.expand.active_out_channel = in_channel

        y = self.fc(y)
        return x * y


class DynamicMBConvLayer(nn.Module):
    """[summary]

    Args:
        max_in_channels ([type]): [description]
        max_out_channels ([type]): [description]
        stride ([type]): [description]
        expand_ratio ([type]): [description]
        max_kernel_size_array ([type]): [description]
    """

    def __init__(self, max_in_channels, max_out_channels, stride, expand_ratio, max_kernel_size_array):
        super(DynamicMBConvLayer, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.expand_ratio = expand_ratio
        max_hidden_dim = int(max_in_channels * expand_ratio)

        self.inverted_bottleneck = nn.Sequential(
            OrderedDict([
                ('conv', DynamicConv2d(max_in_channels, max_hidden_dim)),
                ('bn', DynamicBatchNorm2d(max_hidden_dim)),
                ('act', nn.SiLU(inplace=True)),
            ]))

        self.depth_conv = nn.Sequential(
            OrderedDict([('convbn',
                          SMSU(max_hidden_dim, max_kernel_size_array=max_kernel_size_array, stride=self.stride)),
                         ('act', nn.SiLU(inplace=True)), ('se', DynamicSE(max_hidden_dim))]))

        self.point_linear = nn.Sequential(
            OrderedDict([
                ('conv', DynamicConv2d(max_hidden_dim, max_out_channels)),
                ('bn', DynamicBatchNorm2d(max_out_channels, zero_init=True)),
            ]))

        if self.stride == 2:
            self.identity_downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.identity_conv = DynamicConv2d(max_in_channels, max_out_channels, kernel_size=1)

    def forward(self, x, out_channels, kernel_size_array):
        in_channels = x.shape[1]
        hidden_dim = int(self.expand_ratio * in_channels)

        self.inverted_bottleneck.conv.active_out_channel = hidden_dim
        self.depth_conv.convbn.active_kernel_size_array = kernel_size_array
        self.point_linear.conv.active_out_channel = out_channels

        residual = x
        x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)

        if x.shape == residual.shape:
            x += residual
        else:
            if residual.shape[2] != x.shape[2]:
                residual = self.identity_downsample(residual)
            if residual.shape[1] != x.shape[1]:
                self.identity_conv.active_out_channel = out_channels
                residual = self.identity_conv(residual)
            x += residual
        return x


class DynamicBackbone(nn.Module):
    """[summary]

    Args:
        cfg ([type]): [description]
    """

    def __init__(self, cfg):
        super(DynamicBackbone, self).__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            OrderedDict([('stemconv', DynamicConv2d(3, cfg['stem']['filter'][0], kernel_size=3, stride=2, padding=1)),
                         ('stemnorm', DynamicBatchNorm2d(cfg['stem']['filter'][0])),
                         ('stemact', nn.SiLU(inplace=True))]))
        self.stage_1 = self.build_stage(cfg['stage_1'], cfg['stem']['filter'][-1])
        self.stage_2 = self.build_stage(cfg['stage_2'], cfg['stage_1']['filter'][-1])
        self.stage_3 = self.build_stage(cfg['stage_3'], cfg['stage_2']['filter'][-1])
        self.stage_4 = self.build_stage(cfg['stage_4'], cfg['stage_3']['filter'][-1])
        self.stage_5 = self.build_stage(cfg['stage_5'], cfg['stage_4']['filter'][-1])
        self.stage_6 = self.build_stage(cfg['stage_6'], cfg['stage_5']['filter'][-1])
        self.stage_7 = self.build_stage(cfg['stage_7'], cfg['stage_6']['filter'][-1])

    def build_stage(self, stage_cfg, previous_filters):
        module = nn.ModuleList()
        max_filter_array = stage_cfg['filter']
        max_depth = stage_cfg['depth']
        max_kernel_array = stage_cfg['kernel']
        ratio = stage_cfg['ratio']
        stride = stage_cfg['stride']

        module.append(DynamicMBConvLayer(previous_filters, max_filter_array[0], stride, ratio, max_kernel_array[0]))
        for d in range(1, max_depth):
            module.append(
                DynamicMBConvLayer(max_filter_array[d - 1], max_filter_array[d], 1, ratio, max_kernel_array[d]))
        return module

    def forward(self, x, cfg=None):
        if cfg is None:
            cfg = self.cfg

        out_c = cfg['stem']['filter'][0]
        self.stem.stemconv.active_out_channel = out_c
        x = self.stem(x)

        depth = cfg['stage_1']['depth']
        kernel_array = cfg['stage_1']['kernel']
        filter_array = cfg['stage_1']['filter']

        for d in range(depth):
            x = self.stage_1[d](x, filter_array[d], kernel_array[d])
        x_2x = x

        depth = cfg['stage_2']['depth']
        kernel_array = cfg['stage_2']['kernel']
        filter_array = cfg['stage_2']['filter']
        for d in range(depth):
            x = self.stage_2[d](x, filter_array[d], kernel_array[d])
        x_4x = x

        depth = cfg['stage_3']['depth']
        kernel_array = cfg['stage_3']['kernel']
        filter_array = cfg['stage_3']['filter']
        for d in range(depth):
            x = self.stage_3[d](x, filter_array[d], kernel_array[d])

        x_8x = x

        depth = cfg['stage_4']['depth']
        kernel_array = cfg['stage_4']['kernel']
        filter_array = cfg['stage_4']['filter']
        for d in range(depth):
            x = self.stage_4[d](x, filter_array[d], kernel_array[d])

        depth = cfg['stage_5']['depth']
        kernel_array = cfg['stage_5']['kernel']
        filter_array = cfg['stage_5']['filter']

        for d in range(depth):
            x = self.stage_5[d](x, filter_array[d], kernel_array[d])

        x_16x = x

        depth = cfg['stage_6']['depth']
        kernel_array = cfg['stage_6']['kernel']
        filter_array = cfg['stage_6']['filter']
        for d in range(depth):
            x = self.stage_6[d](x, filter_array[d], kernel_array[d])

        depth = cfg['stage_7']['depth']
        kernel_array = cfg['stage_7']['kernel']
        filter_array = cfg['stage_7']['filter']
        for d in range(depth):
            x = self.stage_7[d](x, filter_array[d], kernel_array[d])
        x_32x = x

        return {'level_0': x_2x, 'level_1': x_4x, 'level_2': x_8x, 'level_3': x_16x, 'level_4': x_32x}
