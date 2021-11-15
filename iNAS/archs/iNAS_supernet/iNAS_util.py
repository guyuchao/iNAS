import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, out_channels=None):
        return x


class DynamicTransportFusion(nn.Module):
    """[summary]

    Args:
        max_in_channels ([type]): [description]
        max_kernel_size_array ([type]): [description]
    """

    def __init__(self, max_in_channels, max_kernel_size_array):
        super(DynamicTransportFusion, self).__init__()

        self.fusion = nn.Sequential(
            OrderedDict([
                ('depthwise1', SMSU(max_in_channels, max_kernel_size_array=max_kernel_size_array)),
                ('pointwise1', DynamicConv2d(max_in_channels=max_in_channels, max_out_channels=max_in_channels)),
                ('bn1', DynamicBatchNorm2d(max_in_channels)),
                ('act1', nn.SiLU(inplace=True)),
                ('depthwise2', SMSU(max_in_channels, max_kernel_size_array=max_kernel_size_array)),
                ('pointwise2', DynamicConv2d(max_in_channels=max_in_channels, max_out_channels=max_in_channels)),
                ('bn2', DynamicBatchNorm2d(max_in_channels)),
            ]))

        self.fusionact = nn.SiLU(inplace=True)

    def forward(self, features, kernel_size_array):
        # step1 fusion
        x = sum(features) / len(features)
        residual = x

        in_channels = x.shape[1]
        self.fusion.pointwise1.active_out_channel = in_channels
        self.fusion.pointwise2.active_out_channel = in_channels
        self.fusion.depthwise1.active_kernel_size_array = kernel_size_array
        self.fusion.depthwise2.active_kernel_size_array = kernel_size_array
        x = self.fusion(x)

        x += residual
        x = self.fusionact(x)

        return x


class DynamicTransportUpsample(nn.Module):
    """[summary]

    Args:
        level_diff ([type]): [description]
        max_in_channels ([type]): [description]
        max_out_channels ([type]): [description]
    """

    def __init__(self, level_diff, max_in_channels, max_out_channels):
        super(DynamicTransportUpsample, self).__init__()

        self.upsample = nn.Sequential(
            OrderedDict([('conv', DynamicConv2d(max_in_channels, max_out_channels)),
                         ('bn', DynamicBatchNorm2d(max_out_channels)),
                         ('upsample', nn.UpsamplingBilinear2d(scale_factor=2**level_diff))]))

    def forward(self, x, out_channels):
        self.upsample.conv.active_out_channel = out_channels
        x = self.upsample(x)
        return x


class DynamicTransportDownsample(nn.Module):
    """[summary]

    Args:
        level_diff ([type]): [description]
        max_in_channels ([type]): [description]
        max_out_channels ([type]): [description]
    """

    def __init__(self, level_diff, max_in_channels, max_out_channels):
        super(DynamicTransportDownsample, self).__init__()
        self.level_diff = level_diff

        self.downsample = nn.Sequential(
            OrderedDict([
                ('downsample', nn.MaxPool2d(kernel_size=2**level_diff, stride=2**level_diff)),
                ('conv', DynamicConv2d(max_in_channels, max_out_channels)),
                ('bn', DynamicBatchNorm2d(max_out_channels)),
            ]))

    def forward(self, x, out_channels):
        self.downsample.conv.active_out_channel = out_channels
        x = self.downsample(x)
        return x


class DynamicDecoderUpsample(nn.Module):
    """[summary]

    Args:
        level_diff ([type]): [description]
        max_in_channels ([type]): [description]
        max_out_channels ([type]): [description]
    """

    def __init__(self, level_diff, max_in_channels, max_out_channels):
        super(DynamicDecoderUpsample, self).__init__()

        self.upsample = nn.Sequential(
            OrderedDict([('conv', DynamicConv2d(max_in_channels, max_out_channels)),
                         ('bn', DynamicBatchNorm2d(max_out_channels)), ('act', nn.SiLU(inplace=True)),
                         ('upsample', nn.UpsamplingBilinear2d(scale_factor=2**level_diff))]))

    def forward(self, x, out_channels):
        self.upsample.conv.active_out_channel = out_channels
        x = self.upsample(x)
        return x


class DynamicDecoderFusion(nn.Module):
    """[summary]

    Args:
        max_in_channels ([type]): [description]
        max_kernel_size_array ([type]): [description]
    """

    def __init__(self, max_in_channels, max_kernel_size_array):
        super(DynamicDecoderFusion, self).__init__()
        self.fusion = nn.Sequential(
            OrderedDict([
                ('depthwise1', SMSU(max_in_channels, max_kernel_size_array=max_kernel_size_array)),
                ('pointwise1', DynamicConv2d(max_in_channels=max_in_channels, max_out_channels=max_in_channels)),
                ('bn1', DynamicBatchNorm2d(max_in_channels)),
                ('act1', nn.SiLU(inplace=True)),
            ]))

    def forward(self, features, kernel_size_array):
        # step1 fusion
        x = sum(features) / len(features)
        in_channels = x.shape[1]
        self.fusion.depthwise1.active_kernel_size_array = kernel_size_array
        self.fusion.pointwise1.active_out_channel = in_channels
        x = self.fusion(x)
        return x


class SMSU(nn.Module):
    """[summary]

    Args:
        channels ([type]): [description]
        max_kernel_size_array ([type]): [description]
        stride (int, optional): [description]. Defaults to 1.
    """

    def __init__(self, channels, max_kernel_size_array, stride=1):
        super(SMSU, self).__init__()
        self.channels = channels
        self.max_kernel_size_array = max_kernel_size_array
        self.stride = stride

        max_kernel = max(self.max_kernel_size_array)
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

        for ks in self.max_kernel_size_array:
            scale_params['ks_%d' % ks] = nn.Parameter(torch.eye(ks**2))
            self.bn_dict['ks_%d' % ks] = DynamicBatchNorm2d(self.channels)

        for name, param in scale_params.items():
            self.register_parameter(name, param)

        self.active_kernel_size_array = max_kernel_size_array

    def get_substart_end(self, kernel_size, max_kernel_size):
        start = (max_kernel_size - kernel_size) // 2
        end = start + kernel_size
        return start, end

    def get_same_padding(self, x_size, kernel_size, stride, dilation):
        return math.ceil(
            max((math.ceil(x_size / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x_size, 0) / 2)

    def scale_filter(self, filter, ks):
        scale_param = self.__getattr__('ks_%d' % (ks))
        filter = filter.view(*filter.shape[:2], -1)
        filter = F.linear(filter, scale_param).view(*filter.shape[:2], ks, ks)
        return filter

    def forward(self, x):
        in_channel = x.size(1)
        x_size = x.size(2)
        if self.training:
            feature_list = []
            for ks in self.active_kernel_size_array:
                start, end = self.get_substart_end(ks, max(self.max_kernel_size_array))
                filter = self.conv.weight[:in_channel, :, start:end, start:end].contiguous()
                filter = self.scale_filter(filter, ks)
                padding = self.get_same_padding(x_size, ks, self.stride, dilation=1)
                f = F.conv2d(x, filter, None, self.stride, padding, dilation=1, groups=in_channel)
                f = self.bn_dict['ks_%d' % ks](f)
                feature_list.append(f)
            y = sum(feature_list) / len(feature_list)
        else:
            new_filter = torch.zeros(
                (in_channel, 1, max(self.active_kernel_size_array), max(self.active_kernel_size_array)),
                device=self.conv.weight.device)
            new_bias = torch.zeros(in_channel, device=self.conv.weight.device)
            for ks in self.active_kernel_size_array:
                mu = self.bn_dict['ks_%d' % ks].bn.running_mean[:in_channel]
                var = self.bn_dict['ks_%d' % ks].bn.running_var[:in_channel]
                w = self.bn_dict['ks_%d' % ks].bn.weight[:in_channel]
                b = self.bn_dict['ks_%d' % ks].bn.bias[:in_channel]
                eps = self.bn_dict['ks_%d' % ks].bn.eps
                start, end = self.get_substart_end(ks, max(self.max_kernel_size_array))
                filter = self.conv.weight[:in_channel, :, start:end, start:end].contiguous()
                filter = self.scale_filter(filter, ks)
                std = (var + eps).sqrt()
                t = (w / std).reshape(-1, 1, 1, 1)
                filter = filter * t
                bias = -(mu * w / std) + b
                start, end = self.get_substart_end(ks, max(self.active_kernel_size_array))
                new_filter[:, :, start:end, start:end] += filter
                new_bias[:] += bias
            new_filter /= len(self.active_kernel_size_array)
            new_bias /= len(self.active_kernel_size_array)
            padding = self.get_same_padding(x_size, max(self.active_kernel_size_array), self.stride, dilation=1)
            y = F.conv2d(x, new_filter, new_bias, self.stride, padding, dilation=1, groups=in_channel)
        return y


class DynamicBatchNorm2d(nn.Module):
    """[summary]

    Args:
        num_features ([type]): [description]
        zero_init (bool, optional): [description]. Defaults to False.
    """

    def __init__(self, num_features, zero_init=False):
        super(DynamicBatchNorm2d, self).__init__()

        self.bn = nn.BatchNorm2d(num_features)
        if zero_init is True:
            nn.init.zeros_(self.bn.weight)

    def forward(self, x):
        weight = self.bn.weight
        bias = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        c = x.shape[1]
        y = nn.functional.batch_norm(x, running_mean[:c], running_var[:c], weight[:c], bias[:c], self.bn.training,
                                     self.bn.momentum, self.bn.eps)
        return y


class DynamicConv2d(nn.Module):
    """[summary]

    Args:
        max_in_channels ([type]): [description]
        max_out_channels ([type]): [description]
        kernel_size (int, optional): [description]. Defaults to 1.
        stride (int, optional): [description]. Defaults to 1.
        padding (int, optional): [description]. Defaults to 0.
        dilation (int, optional): [description]. Defaults to 1.
        bias (bool, optional): [description]. Defaults to False.
    """

    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=padding,
            bias=bias,
        )
        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x):
        out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()
        if self.conv.bias is None:
            bias = self.conv.bias
        else:
            bias = self.conv.bias[:out_channel]
        y = F.conv2d(x, filters, bias, self.stride, self.padding, self.dilation, 1)
        return y
