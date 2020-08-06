import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import Conv2d
from . import RegressionModel, LinearPrior
from .. import prior

__all__ = ('Conv2dPrior', 'PreActResNet18', 'PreActResNet34')

def Conv2dPrior(in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, groups=1, padding_mode='zeros',
            prior_w=prior.Normal, loc_w=0., std_w=1., prior_b=prior.Normal,
            loc_b=0., std_b=1., scaling_fn=None):
    if scaling_fn is None:
        def scaling_fn(std, dim):
            return std/dim**0.5
    kernel_size = nn.modules.utils._pair(kernel_size)
    bias_prior = prior_b((out_channels,), 0., std_b) if prior_b is not None else None
    return Conv2d(weight_prior=prior_w((out_channels, in_channels//groups, kernel_size[0], kernel_size[1]),
                                       loc_w, scaling_fn(std_w, in_channels)),
                  bias_prior=bias_prior,
                 stride=stride, padding=padding, dilation=dilation,
                  groups=groups, padding_mode=padding_mode)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True,
                 prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                 prior_b=prior.Normal, loc_b=0., std_b=1.,
                scaling_fn=None):
        super(PreActBlock, self).__init__()
        if bn:
            batchnorm = nn.BatchNorm2d
        else:
            batchnorm = nn.Identity
        self.bn1 = batchnorm(in_planes)
        self.conv1 = Conv2dPrior(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                 prior_w=prior_w, loc_w=loc_w, std_w=std_w,
                                 prior_b=None, scaling_fn=scaling_fn)
        self.bn2 = batchnorm(planes)
        self.conv2 = Conv2dPrior(planes, planes, kernel_size=3, stride=1, padding=1,
                                 prior_w=prior_w, loc_w=loc_w, std_w=std_w,
                                 prior_b=None, scaling_fn=scaling_fn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = x
        out = self.bn1(out)
        out = F.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(F.relu(out))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bn=True,
                 prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                 prior_b=prior.Normal, loc_b=0., std_b=1.,
                scaling_fn=None):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.bn = bn
        self.prior_w = prior_w
        self.loc_w = loc_w
        self.std_w = std_w
        self.prior_b = prior_b
        self.loc_b = loc_b
        self.std_b = std_b
        self.scaling_fn = scaling_fn

        self.conv1 = Conv2dPrior(3, 64, kernel_size=3, stride=1, padding=1, prior_b=None,
                           prior_w=self.prior_w, loc_w=self.loc_w, std_w=self.std_w,
                           scaling_fn=self.scaling_fn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = LinearPrior(512*block.expansion, num_classes,
                            prior_w=self.prior_w, loc_w=self.loc_w, std_w=self.std_w,
                            prior_b=self.prior_b, loc_b=self.loc_b, std_b=self.std_b,
                            scaling_fn=self.scaling_fn)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn=self.bn,
                                prior_w=self.prior_w, loc_w=self.loc_w, std_w=self.std_w,
                                prior_b=self.prior_b, loc_b=self.loc_b, std_b=self.std_b,
                                scaling_fn=self.scaling_fn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(noise_std=1.,
             prior_w=prior.Normal, loc_w=0., std_w=2**.5,
             prior_b=prior.Normal, loc_b=0., std_b=1.,
            scaling_fn=None, bn=True):
    return RegressionModel(PreActResNet(PreActBlock,
                                        [2,2,2,2], bn=bn,
                                        prior_w=prior_w,
                                       loc_w=loc_w,
                                       std_w=std_w,
                                       prior_b=prior_b,
                                       loc_b=loc_b,
                                       std_b=std_b,
                                       scaling_fn=scaling_fn,), noise_std)


def PreActResNet34(noise_std=1.,
             prior_w=prior.Normal, loc_w=0., std_w=2**.5,
             prior_b=prior.Normal, loc_b=0., std_b=1.,
            scaling_fn=None, bn=True):
    return RegressionModel(PreActResNet(PreActBlock,
                                        [3,4,6,3], bn=bn,
                                        prior_w=prior_w,
                                       loc_w=loc_w,
                                       std_w=std_w,
                                       prior_b=prior_b,
                                       loc_b=loc_b,
                                       std_b=std_b,
                                       scaling_fn=scaling_fn,), noise_std)