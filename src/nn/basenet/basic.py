from collections import OrderedDict
import torch.nn as nn
from .tf_like import MaxPool2dSame, Conv2dSame


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    if padding == 'same':
        if padding_mode != 'zeros':
            raise NotImplementedError(f"padding_mode {padding_mode} for same padding")
        return Conv2dSame(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride, dilation=dilation, groups=groups, bias=bias)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)


def MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    if padding == 'same':
        return MaxPool2dSame(kernel_size=kernel_size, stride=stride, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)


def conv(*args, **kwargs):
    return lambda last_layer: Conv2d(get_num_of_channels(last_layer), *args, **kwargs)


def get_num_of_channels(layers, channle_name='out_channels'):
    """access out_channels from last layer of nn.Sequential/list"""
    if hasattr(layers, channle_name):
        return getattr(layers, channle_name)
    elif isinstance(layers, int):
        return layers
    elif isinstance(layers, nn.BatchNorm2d):
        return layers.num_features
    else:
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            if hasattr(layer, channle_name):
                return getattr(layer, channle_name)
            elif isinstance(layer, (nn.Sequential, nn.BatchNorm2d)):
                return get_num_of_channels(layer, channle_name)
    raise RuntimeError("cant get_num_of_channels {} from {}".format(channle_name, layers))


def Sequential(*args):
    f = nn.Sequential(*args)
    f.in_channels = get_num_of_channels(f, 'in_channels')
    f.out_channels = get_num_of_channels(f)
    return f


def sequential(*args):
    def create_sequential(last_layer):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            layers = OrderedDict()
            for key, a in args[0].items():
                m = a(last_layer)
                layers[key] = m
                last_layer = m
            return Sequential(layers)
        else:
            layers = []
            for a in args:
                layers.append(a(last_layer))
                last_layer = layers[-1]
            return Sequential(*layers)

    return create_sequential


def ConvBn(*args, **kwargs):
    """drop in block for Conv2d with BatchNorm and ReLU"""
    c = Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels))


def conv_bn(*args, **kwargs):
    return lambda last_layer: ConvBn(get_num_of_channels(last_layer), *args, **kwargs)


def ConvBnRelu(*args, **kwargs):
    """drop in block for Conv2d with BatchNorm and ReLU"""
    c = Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels),
                      nn.ReLU(inplace=True))


def conv_bn_relu(*args, **kwargs):
    return lambda last_layer: ConvBnRelu(get_num_of_channels(last_layer), *args, **kwargs)


def ConvBnRelu6(*args, **kwargs):
    """drop in block for Conv2d with BatchNorm and ReLU6"""
    c = Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels),
                      nn.ReLU(inplace=True))


def conv_bn_relu6(*args, **kwargs):
    return lambda last_layer: ConvBnRelu6(get_num_of_channels(last_layer), *args, **kwargs)


def ConvRelu(*args, **kwargs):
    return Sequential(Conv2d(*args, **kwargs),
                      nn.ReLU(inplace=True))


def conv_relu(*args, **kwargs):
    return lambda last_layer: ConvRelu(get_num_of_channels(last_layer), *args, **kwargs)


def ConvRelu6(*args, **kwargs):
    return Sequential(Conv2d(*args, **kwargs),
                      nn.ReLU6(inplace=True))


def conv_relu6(*args, **kwargs):
    return lambda last_layer: ConvRelu6(get_num_of_channels(last_layer), *args, **kwargs)


def ReluConv(*args, **kwargs):
    return Sequential(nn.ReLU(inplace=True),
                      Conv2d(*args, **kwargs))


def relu_conv(*args, **kwargs):
    return lambda last_layer: ReluConv(get_num_of_channels(last_layer), *args, **kwargs)


def BnReluConv(*args, **kwargs):
    """drop in block for Conv2d with BatchNorm and ReLU"""
    c = Conv2d(*args, **kwargs)
    return Sequential(nn.BatchNorm2d(c.in_channels),
                      nn.ReLU(inplace=True),
                      c)


def bn_relu_conv(*args, **kwargs):
    return lambda last_layer: BnReluConv(get_num_of_channels(last_layer), *args, **kwargs)


def maxpool(*args, **kwargs):
    def max_pool_module(last_layer):
        m = MaxPool2d(*args, **kwargs)
        m.in_channels = m.out_channels = last_layer.out_channels
        return m

    return max_pool_module