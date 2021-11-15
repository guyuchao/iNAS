import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TransportFusion(nn.Module):
    """[summary]

    Args:
        channels ([type]): [description]
        kernel_size_array ([type]): [description]
    """

    def __init__(self, channels, kernel_size_array):

        super(TransportFusion, self).__init__()

        self.fusion = nn.Sequential(
            OrderedDict([
                ('depthwise1', SMSU(channels, kernel_size_array=kernel_size_array)),
                ('pointwise1', Conv2d(in_channels=channels, out_channels=channels)),
                ('bn1', BatchNorm2d(channels)),
                ('act1', nn.SiLU(inplace=True)),
                ('depthwise2', SMSU(channels, kernel_size_array=kernel_size_array)),
                ('pointwise2', Conv2d(in_channels=channels, out_channels=channels)),
                ('bn2', BatchNorm2d(channels)),
            ]))

        self.fusionact = nn.SiLU(inplace=True)

    def forward(self, features):
        if isinstance(features, list):
            x = sum(features) / len(features)
        else:
            f_list = [features[:, idx] for idx in range(features.shape[1])]
            x = sum(f_list) / len(f_list)

        residual = x
        x = self.fusion(x)

        x += residual
        x = self.fusionact(x)
        return x


class TransportUpsample(nn.Module):
    """[summary]

    Args:
        level_diff ([type]): [description]
        in_channels ([type]): [description]
        out_channels ([type]): [description]
    """

    def __init__(self, level_diff, in_channels, out_channels):

        super(TransportUpsample, self).__init__()

        self.upsample = nn.Sequential(
            OrderedDict([('conv', Conv2d(in_channels, out_channels)), ('bn', BatchNorm2d(out_channels)),
                         ('upsample', nn.UpsamplingBilinear2d(scale_factor=2**level_diff))]))

    def forward(self, x):
        x = self.upsample(x)
        return x


class TransportDownsample(nn.Module):
    """[summary]

    Args:
        level_diff ([type]): [description]
        in_channels ([type]): [description]
        out_channels ([type]): [description]
    """

    def __init__(self, level_diff, in_channels, out_channels):

        super(TransportDownsample, self).__init__()
        self.level_diff = level_diff

        self.downsample = nn.Sequential(
            OrderedDict([
                ('downsample', nn.MaxPool2d(kernel_size=2**level_diff, stride=2**level_diff)),
                ('conv', Conv2d(in_channels, out_channels)),
                ('bn', BatchNorm2d(out_channels, )),
            ]))

    def forward(self, x):
        x = self.downsample(x)
        return x


class DecoderUpsample(nn.Module):
    """[summary]

    Args:
        level_diff ([type]): [description]
        in_channels ([type]): [description]
        out_channels ([type]): [description]
    """

    def __init__(self, level_diff, in_channels, out_channels):

        super(DecoderUpsample, self).__init__()

        self.upsample = nn.Sequential(
            OrderedDict([('conv', Conv2d(in_channels, out_channels)), ('bn', BatchNorm2d(out_channels)),
                         ('act', nn.SiLU(inplace=True)),
                         ('upsample', nn.UpsamplingBilinear2d(scale_factor=2**level_diff))]))

    def forward(self, x):
        x = self.upsample(x)
        return x


class DecoderFusion(nn.Module):
    """[summary]

    Args:
        channels ([type]): [description]
        kernel_size_array ([type]): [description]
    """

    def __init__(self, channels, kernel_size_array):

        super(DecoderFusion, self).__init__()
        self.fusion = nn.Sequential(
            OrderedDict([
                ('depthwise1', SMSU(channels, kernel_size_array=kernel_size_array)),
                ('pointwise1', Conv2d(in_channels=channels, out_channels=channels)),
                ('bn1', BatchNorm2d(channels)),
                ('act1', nn.SiLU(inplace=True)),
            ]))

    def forward(self, features):
        if isinstance(features, list):
            x = sum(features) / len(features)
        else:
            f_list = [features[:, idx] for idx in range(features.shape[1])]
            x = sum(f_list) / len(f_list)
        x = self.fusion(x)
        return x


class SMSU(nn.Module):
    """[summary]

    Args:
        channels ([type]): [description]
        kernel_size_array ([type]): [description]
        stride (int, optional): [description]. Defaults to 1.
    """

    def __init__(self, channels, kernel_size_array, stride=1):

        super(SMSU, self).__init__()
        self.channels = channels
        self.kernel_size_array = kernel_size_array
        self.stride = stride

        max_kernel = max(self.kernel_size_array)
        padding = self.get_same_padding(224, max_kernel, self.stride, dilation=1)

        self.conv = nn.Conv2d(
            self.channels,
            self.channels,
            max_kernel,
            stride=self.stride,
            padding=padding,
            bias=False,
            groups=self.channels)
        self.bn_dict = nn.ModuleDict()
        scale_params = {}

        for ks in self.kernel_size_array:
            scale_params['ks_%d' % ks] = nn.Parameter(torch.eye(ks**2))
            self.bn_dict['ks_%d' % ks] = BatchNorm2d(self.channels)

        for name, param in scale_params.items():
            self.register_parameter(name, param)

    def get_substart_end(self, kernel_size, max_kernel_size):
        start = (max_kernel_size - kernel_size) // 2
        end = start + kernel_size
        return start, end

    def scale_filter(self, filter, ks):
        scale_param = self.__getattr__('ks_%d' % (ks))
        filter = filter.view(*filter.shape[:2], -1)
        filter = F.linear(filter, scale_param).view(*filter.shape[:2], ks, ks)
        return filter

    def reparams(self):
        new_filter = torch.zeros_like(self.conv.weight)
        new_bias = torch.zeros(self.conv.weight.shape[0], device=self.conv.weight.device)
        for ks in self.kernel_size_array:
            mu = self.bn_dict['ks_%d' % ks].bn.running_mean
            var = self.bn_dict['ks_%d' % ks].bn.running_var
            w = self.bn_dict['ks_%d' % ks].bn.weight
            b = self.bn_dict['ks_%d' % ks].bn.bias
            eps = self.bn_dict['ks_%d' % ks].bn.eps

            start, end = self.get_substart_end(ks, max(self.kernel_size_array))
            filter = self.conv.weight[:, :, start:end, start:end].contiguous()
            filter = self.scale_filter(filter, ks)
            std = (var + eps).sqrt()
            t = (w / std).reshape(-1, 1, 1, 1)
            filter = filter * t
            bias = -mu * w / std + b
            start, end = self.get_substart_end(ks, max(self.kernel_size_array))
            new_filter[:, :, start:end, start:end] += filter
            new_bias[:] += bias
        new_filter /= len(self.kernel_size_array)
        new_bias /= len(self.kernel_size_array)
        self.conv.weight.copy_(new_filter)
        self.conv.bias = torch.nn.Parameter(new_bias)
        del self.bn_dict

    def get_same_padding(self, x_size, kernel_size, stride, dilation):
        return math.ceil(
            max((math.ceil(x_size / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x_size, 0) / 2)

    def forward(self, x):
        y = self.conv(x)
        return y


class BatchNorm2d(nn.Module):
    """[summary]

    Args:
        num_features ([type]): [description]
        zero_init (bool, optional): [description]. Defaults to False.
    """

    def __init__(self, num_features, zero_init=False):

        super(BatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        if zero_init is True:
            nn.init.zeros_(self.bn.weight)

    def forward(self, x):
        y = self.bn(x)
        return y


class Conv2d(nn.Module):
    """[summary]

    Args:
        in_channels ([type]): [description]
        out_channels ([type]): [description]
        kernel_size (int, optional): [description]. Defaults to 1.
        stride (int, optional): [description]. Defaults to 1.
        padding (int, optional): [description]. Defaults to 0.
        dilation (int, optional): [description]. Defaults to 1.
        bias (bool, optional): [description]. Defaults to False.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):

        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        y = self.conv(x)
        return y
