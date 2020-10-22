# Parts of this file are taken from https://github.com/google-research/google-research/blob/master/cold_posterior_bnn/models.py
# licensed under the Apache License, Version 2.0

import torch.nn as nn

from .conv_nets import Conv2dPrior
from .dense_nets import LinearPrior
from .base import ClassificationModel
from .. import prior

class BasicBlock(nn.Module):
    def __init__(self, in_filters, filters, stride, conv_kwargs, batchnorm):
        super().__init__()
        self.main = nn.Sequential(
            Conv2dPrior(in_filters, filters, kernel_size=3, padding=1, stride=stride, **conv_kwargs),
            batchnorm(filters),
            nn.ReLU(),
            Conv2dPrior(filters, filters, kernel_size=3, padding=1, stride=1, **conv_kwargs),
            batchnorm(filters))

        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                Conv2dPrior(in_filters, filters, kernel_size=1, padding=0, stride=stride, **conv_kwargs),
                batchnorm(filters),
            )

    def forward(self, x):
        y = self.main(x)
        z = self.shortcut(x)
        return nn.functional.relu(y + z)

def ResNet(softmax_temp=1., depth=20, num_classes=10,
           prior_w=prior.Normal, loc_w=0., std_w=2**.5,
           prior_b=prior.Normal, loc_b=0., std_b=1.,
           scaling_fn=None, bn=True, weight_prior_params={}, bias_prior_params={},
           conv_prior_w=prior.Normal):
    conv_kwargs = dict(
        prior_w=conv_prior_w, loc_w=loc_w, std_w=std_w,
        prior_b=None, scaling_fn=scaling_fn,
        weight_prior_params=weight_prior_params,
        bias_prior_params=bias_prior_params)
    batchnorm = (nn.BatchNorm2d if bn else nn.Identity)

    # Main network code
    num_res_blocks = (depth - 2) // 6
    filters = 16
    if (depth - 2) % 6 != 0:
        raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

    layers = [
        Conv2dPrior(3, filters, kernel_size=3, padding=1, stride=1, **conv_kwargs),
        batchnorm(filters),
        nn.ReLU()]

    for stack in range(3):
        stride = (1 if stack == 0 else 2)
        prev_filters = filters
        filters *= stride

        layers.append(BasicBlock(
            prev_filters, filters, stride, conv_kwargs, batchnorm))

        for _ in range(num_res_blocks-1):
            layers.append(BasicBlock(
                filters, filters, 1, conv_kwargs, batchnorm))

    layers += [
        nn.AvgPool2d(8),
        nn.Flatten(),
        LinearPrior(filters, num_classes,
                    prior_w=prior_w, loc_w=loc_w, std_w=std_w,
                    prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                    scaling_fn=scaling_fn,
                    weight_prior_params=weight_prior_params,
                    bias_prior_params=bias_prior_params)]
    return ClassificationModel(nn.Sequential(*layers), softmax_temp=softmax_temp)


def CorrelatedResNet(softmax_temp=1., depth=20, num_classes=10,
           prior_w=prior.Normal, loc_w=0., std_w=2**.5,
           prior_b=prior.Normal, loc_b=0., std_b=1.,
           scaling_fn=None, bn=True, weight_prior_params={}, bias_prior_params={},
           conv_prior_w=prior.ConvCorrelatedNormal):
    return ResNet(
        softmax_temp=softmax_temp, depth=depth, num_classes=num_classes,
        prior_w=prior_w, loc_w=loc_w, std_w=std_w,
        prior_b=prior_b, loc_b=loc_b, std_b=std_b,
        scaling_fn=scaling_fn, bn=bn, weight_prior_params=weight_prior_params,
        bias_prior_params=bias_prior_params,
        conv_prior_w=conv_prior_w)
