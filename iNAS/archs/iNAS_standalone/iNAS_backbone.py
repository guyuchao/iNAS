import torch.nn as nn
from collections import OrderedDict

from iNAS.archs.iNAS_standalone.iNAS_util import SMSU, BatchNorm2d, Conv2d


class SE(nn.Module):
    """Squeeze and excitation block.

    Args:
        in_channels (int): Numbers of channels in the input feature map.
    """

    REDUCTION = 4

    def __init__(self, in_channels):
        super(SE, self).__init__()

        self.in_channels = in_channels
        self.reduction = SE.REDUCTION

        mid_channels = self.in_channels // self.reduction

        self.fc = nn.Sequential(
            OrderedDict([
                ('reduce', Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True)),
                ('relu', nn.ReLU(inplace=True)),
                ('expand', Conv2d(mid_channels, in_channels, 1, 1, 0, bias=True)),
                ('h_sigmoid', nn.Hardsigmoid(inplace=True)),
            ]))

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y


class MBConvLayer(nn.Module):
    """Mobilenet-V2 convolution layer with searchable multi-scale unit(SMSU).

    Args:
        in_channels (int): [description]
        out_channels (int): [description]
        stride (int): [description]
        expand_ratio (int): [description]
        kernel_size_array (list): Kernel combinations in SMSU.
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio, kernel_size_array):
        super(MBConvLayer, self).__init__()

        self.stride = stride
        assert stride in [1, 2]
        self.expand_ratio = expand_ratio
        mid_channels = int(in_channels * expand_ratio)
        self.inverted_bottleneck = nn.Sequential(
            OrderedDict([
                ('conv', Conv2d(in_channels, mid_channels)),
                ('bn', BatchNorm2d(mid_channels)),
                ('act', nn.SiLU(inplace=True)),
            ]))

        self.depth_conv = nn.Sequential(
            OrderedDict([('convbn', SMSU(mid_channels, kernel_size_array=kernel_size_array, stride=self.stride)),
                         ('act', nn.SiLU(inplace=True)), ('se', SE(mid_channels))]))

        self.point_linear = nn.Sequential(
            OrderedDict([
                ('conv', Conv2d(mid_channels, out_channels)),
                ('bn', BatchNorm2d(out_channels, zero_init=True)),
            ]))

        if self.stride == 2:
            self.identity_downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.identity_conv = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
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
                residual = self.identity_conv(residual)
            x += residual
        return x


class Backbone(nn.Module):
    """Stand-alone backbone.

    Args:
        cfg (dict): Defines the backbone configuration.
    """

    def __init__(self, cfg):
        super(Backbone, self).__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            OrderedDict([('stemconv', Conv2d(3, cfg['stem']['filter'][0], kernel_size=3, stride=2, padding=1)),
                         ('stemnorm', BatchNorm2d(cfg['stem']['filter'][0])), ('stemact', nn.SiLU(inplace=True))]))
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

        module.append(MBConvLayer(previous_filters, max_filter_array[0], stride, ratio, max_kernel_array[0]))
        for d in range(1, max_depth):
            module.append(MBConvLayer(max_filter_array[d - 1], max_filter_array[d], 1, ratio, max_kernel_array[d]))
        return module

    def forward(self, x):
        x = self.stem(x)
        for d in range(len(self.stage_1)):
            x = self.stage_1[d](x)
        x_2x = x

        for d in range(len(self.stage_2)):
            x = self.stage_2[d](x)
        x_4x = x

        for d in range(len(self.stage_3)):
            x = self.stage_3[d](x)
        x_8x = x

        for d in range(len(self.stage_4)):
            x = self.stage_4[d](x)

        for d in range(len(self.stage_5)):
            x = self.stage_5[d](x)
        x_16x = x

        for d in range(len(self.stage_6)):
            x = self.stage_6[d](x)

        for d in range(len(self.stage_7)):
            x = self.stage_7[d](x)
        x_32x = x

        return {'level_0': x_2x, 'level_1': x_4x, 'level_2': x_8x, 'level_3': x_16x, 'level_4': x_32x}
