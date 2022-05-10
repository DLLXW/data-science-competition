# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class FrozenBatchNorm1d(nn.Module):
    """
    BatchNorm1d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm1d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1)
        bias = bias.reshape(1, -1)
        return x * scale + bias


def replace_bn(f, src_bn, dst_bn):
    """
    Replace BatchNorm modules in input model and copy BatchNorm parameters.
    Current implementation only supports normal BN to FrozenBN
    :param f: nn model.
    :param src_bn: original batch norm module.
    :param dst_bn: target batch norm module.
    """
    for key, module in f._modules.items():
        if key.isdigit():
            key = int(key)
            if isinstance(f[key], src_bn):

                for p in ['weight', 'bias']:
                    f[key].__dict__['_buffers'][p] = f[key].__dict__['_parameters'][p].data
                    del f[key].__dict__['_parameters'][p]
                attrs = f[key].__dict__.copy()
                f[key] = dst_bn(f[key].num_features)

                for attr_key, attr_value in attrs.items():
                    if hasattr(f[key], attr_key):
                        setattr(f[key], attr_key, attr_value)
        else:
            if isinstance(f._modules[key], src_bn):

                for p in ['weight', 'bias']:
                    f._modules[key].__dict__['_buffers'][p] = f._modules[key].__dict__['_parameters'][p].data
                    del f._modules[key].__dict__['_parameters'][p]
                attrs = f._modules[key].__dict__.copy()
                f._modules[key] = dst_bn(f._modules[key].num_features)

                for attr_key, attr_value in attrs.items():
                    if hasattr(f._modules[key], attr_key):
                        setattr(f._modules[key], attr_key, attr_value)

        replace_bn(module, src_bn, dst_bn)


def freeze_bn(model):
    replace_bn(model, nn.BatchNorm2d, FrozenBatchNorm2d)
    replace_bn(model, nn.BatchNorm1d, FrozenBatchNorm1d)
