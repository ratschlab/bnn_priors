"""
Adapted from  https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pandas as pd
from numbers import Number

from .layers import Conv2d
from .base import RegressionModel, ClassificationModel
from .dense_nets import LinearPrior
from .. import prior

__all__ = ('Conv2dPrior', 'PreActResNet18', 'PreActResNet34', 'ClassificationConvNet', 'CorrelatedClassificationConvNet', 'ThinPreActResNet18', 'DataDrivenPreActResNet18')

def Conv2dPrior(in_channels, out_channels, kernel_size=3, stride=1,
            padding=0, dilation=1, groups=1, padding_mode='zeros',
            prior_w=prior.Normal, loc_w=0., std_w=1., prior_b=prior.Normal,
            loc_b=0., std_b=1., scaling_fn=None, weight_prior_params={}, bias_prior_params={}):
    if scaling_fn is None:
        def scaling_fn(std, dim):
            return std/dim**0.5

    in_dim = in_channels * kernel_size**2
    kernel_size = nn.modules.utils._pair(kernel_size)
    bias_prior = prior_b((out_channels,), 0., std_b, **bias_prior_params) if prior_b is not None else None
    return Conv2d(weight_prior=prior_w((out_channels, in_channels//groups, kernel_size[0], kernel_size[1]),
                                       loc_w, scaling_fn(std_w, in_channels),  # TODO: use `in_dim` here to prevent the variance from blowing up
                                       **weight_prior_params),
                  bias_prior=bias_prior,
                 stride=stride, padding=padding, dilation=dilation,
                  groups=groups, padding_mode=padding_mode)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def ClassificationConvNet(in_channels, img_height, out_features, width, depth=3, softmax_temp=1.,
                           prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                           prior_b=prior.Normal, loc_b=0., std_b=1.,
                           scaling_fn=None, weight_prior_params={}, bias_prior_params={}):
    assert depth >= 2, "We can't have less than two layers"
    layers = [Reshape(-1, in_channels, img_height, img_height),
              Conv2dPrior(in_channels, width, kernel_size=3, padding=1, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params),
            nn.ReLU(), nn.MaxPool2d(2)]
    for _ in range(depth-2):
        layers.append(Conv2dPrior(width, width, kernel_size=3, padding=1, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
    layers.append(nn.Flatten())
    reshaped_size = width*(img_height//2**(depth-1))**2
    layers.append(LinearPrior(reshaped_size, out_features, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params))
    return ClassificationModel(nn.Sequential(*layers), softmax_temp)


def CorrelatedClassificationConvNet(in_channels, img_height, out_features, width, depth=3, softmax_temp=1.,
                           prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                           prior_b=prior.Normal, loc_b=0., std_b=1.,
                           scaling_fn=None, weight_prior_params={}, bias_prior_params={}):
    # This is the same as `ClassificationConvNet`, but with the `ConvCorrelatedNormal` prior. The scaling is chosen
    # to be such that the same prior is obtained when no correlation is given.
    assert depth >= 2, "We can't have less than two layers"
    # TODO: ideally we should be able to specify different priors for conv weight and dense weight
    # But for now it probably suffices to be able to change the conv weight prior to try different ones
    conv_prior_w = prior_w
    prior_w = prior.Normal
    conv_weight_prior_params_1 = {}
    if "lengthscale_1" in weight_prior_params:
        conv_weight_prior_params_1["lengthscale"] = weight_prior_params["lengthscale_1"]

    conv_weight_prior_params_2 = {}
    if "lengthscale_2" in weight_prior_params:
        conv_weight_prior_params_2["lengthscale"] = weight_prior_params["lengthscale_2"]

    weight_prior_params = {k: v for k, v in weight_prior_params.items()
                           if k not in ["lengthscale_1", "lengthscale_2"]}
    layers = [Reshape(-1, in_channels, img_height, img_height),
              Conv2dPrior(in_channels, width, kernel_size=3, padding=1, prior_w=conv_prior_w, loc_w=loc_w,
                          std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                          scaling_fn=scaling_fn, weight_prior_params=conv_weight_prior_params_1,
                          bias_prior_params=bias_prior_params),
            nn.ReLU(), nn.MaxPool2d(2)]
    for _ in range(depth-2):
        layers.append(Conv2dPrior(width, width, kernel_size=3, padding=1, prior_w=conv_prior_w, loc_w=loc_w,
                      std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                      scaling_fn=scaling_fn, weight_prior_params=conv_weight_prior_params_2,
                      bias_prior_params=bias_prior_params))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
    layers.append(nn.Flatten())
    reshaped_size = width*(img_height//2**(depth-1))**2
    layers.append(LinearPrior(reshaped_size, out_features, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params))
    return ClassificationModel(nn.Sequential(*layers), softmax_temp)




class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True,
                 prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                 prior_b=prior.Normal, loc_b=0., std_b=1.,
                scaling_fn=None, weight_prior_params={}, bias_prior_params={}):
        super(PreActBlock, self).__init__()
        if bn:
            batchnorm = nn.BatchNorm2d
        else:
            batchnorm = nn.Identity
        self.bn1 = batchnorm(in_planes)
        self.conv1 = Conv2dPrior(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                 prior_w=prior_w, loc_w=loc_w, std_w=std_w,
                                 prior_b=None, scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                                bias_prior_params=bias_prior_params)
        self.bn2 = batchnorm(planes)
        self.conv2 = Conv2dPrior(planes, planes, kernel_size=3, stride=1, padding=1,
                                 prior_w=prior_w, loc_w=loc_w, std_w=std_w,
                                 prior_b=None, scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                                bias_prior_params=bias_prior_params)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dPrior(in_planes, self.expansion*planes, kernel_size=1, stride=stride,
                                 prior_w=prior_w, loc_w=loc_w, std_w=std_w,
                                 prior_b=None, scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                                bias_prior_params=bias_prior_params)
            )
        else:
            self.shortcut = (lambda x: x)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bn=True,
                 prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                 prior_b=prior.Normal, loc_b=0., std_b=1.,
                 in_planes=64, scaling_fn=None,
                 weight_prior_params={}, bias_prior_params={}):
        super(PreActResNet, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.prior_w = prior_w
        self.loc_w = loc_w
        self.std_w = std_w
        self.prior_b = prior_b
        self.loc_b = loc_b
        self.std_b = std_b
        self.scaling_fn = scaling_fn
        self.weight_prior_params = weight_prior_params
        self.bias_prior_params = bias_prior_params

        if prior_w in [prior.ConvCorrelatedNormal, prior.FixedCovNormal]:
            dense_prior_w = prior.Normal
        elif prior_w == prior.FixedCovGenNorm:
            dense_prior_w = prior.GenNorm
        else:
            dense_prior_w = prior_w

        # `self.in_planes` gets modified, so we use `in_planes`.
        self.conv1 = Conv2dPrior(3, in_planes, kernel_size=3, stride=1, padding=1, prior_b=None,
                           prior_w=self.prior_w, loc_w=self.loc_w, std_w=self.std_w,
                           scaling_fn=self.scaling_fn, weight_prior_params=self.weight_prior_params,
                            bias_prior_params=self.bias_prior_params)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * in_planes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * in_planes, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * in_planes, num_blocks[3], stride=2)
        self.linear = LinearPrior(8 * in_planes, num_classes,
                            prior_w=dense_prior_w, loc_w=self.loc_w, std_w=self.std_w,
                            prior_b=self.prior_b, loc_b=self.loc_b, std_b=self.std_b,
                            scaling_fn=self.scaling_fn, weight_prior_params=self.weight_prior_params,
                            bias_prior_params=self.bias_prior_params)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn=self.bn,
                                prior_w=self.prior_w, loc_w=self.loc_w, std_w=self.std_w,
                                prior_b=self.prior_b, loc_b=self.loc_b, std_b=self.std_b,
                                scaling_fn=self.scaling_fn, weight_prior_params=self.weight_prior_params,
                                bias_prior_params=self.bias_prior_params))
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


def PreActResNet18(softmax_temp=1., width=64,
             prior_w=prior.Normal, loc_w=0., std_w=2**.5,
             prior_b=prior.Normal, loc_b=0., std_b=1.,
            scaling_fn=None, bn=True, weight_prior_params={}, bias_prior_params={}):

    load_file_keys = ['lengthscale_dict_file']

    load_file = {k: v for k, v in weight_prior_params.items() if k in load_file_keys}
    weight_prior_params = {k: v for k, v in weight_prior_params.items() if k not in load_file_keys}

    model = ClassificationModel(PreActResNet(PreActBlock,
                                        [2,2,2,2], bn=bn,
                                        prior_w=prior_w,
                                       loc_w=loc_w,
                                       std_w=std_w,
                                       prior_b=prior_b,
                                       loc_b=loc_b,
                                       std_b=std_b,
                                       scaling_fn=scaling_fn, in_planes=width,
                                       weight_prior_params=weight_prior_params,
                                        bias_prior_params=bias_prior_params), softmax_temp)

    if 'lengthscale_dict_file' in load_file:
        lengthscale_dict = pd.read_pickle(load_file['lengthscale_dict_file'])
        sd = model.state_dict()
        for k, v in lengthscale_dict.items():
            assert k.startswith("net.module.") and k.endswith(".p")
            new_k = "net." + k[len("net.module."):-len(".p")] + ".lengthscale"
            sd[ new_k ][...] = v
        model.load_state_dict(sd)
    return model


def DataDrivenPreActResNet18(softmax_temp=1., width=64,
                             prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                             prior_b=prior.Normal, loc_b=0., std_b=1.,
                             scaling_fn=None, bn=True, weight_prior_params={}, bias_prior_params={}):
    assert scaling_fn is None
    scaling_fn = (lambda std, dim: std)

    load_file_keys = ['mean_covs_file', 'fits_dict_file']
    load_file = {k: v for k, v in weight_prior_params.items() if k in load_file_keys}
    weight_prior_params = {k: v for k, v in weight_prior_params.items() if k not in load_file_keys}

    model = ClassificationModel(PreActResNet(PreActBlock,
                                        [2,2,2,2], bn=bn,
                                        prior_w=prior_w,
                                       loc_w=loc_w,
                                       std_w=std_w,
                                       prior_b=prior_b,
                                       loc_b=loc_b,
                                       std_b=std_b,
                                       scaling_fn=scaling_fn, in_planes=width,
                                       weight_prior_params=weight_prior_params,
                                        bias_prior_params=bias_prior_params), softmax_temp)

    if 'mean_covs_file' in load_file:
        loaded_keys = set()
        mean_covs = pd.read_pickle(load_file['mean_covs_file'])
        prior_modules = {("net." + k[len("net.module."):-len(".p")]): k
                         for k in mean_covs.keys()}
        for name, mod in model.named_modules():
            if name not in prior_modules:
                continue
            key = prior_modules[name]
            mean, cov = mean_covs[key]
            loaded_keys.add(key)

            if isinstance(mean, Number):
                assert mod.loc.numel() == 1
                mod.loc[...] = float(mean)
            else:
                assert mod.loc.shape == mean.shape
                mod.loc[...] = torch.from_numpy(mean)

            if isinstance(cov, Number):
                assert mod.scale.numel() == 1
                mod.scale[...] = cov**.5
            else:
                assert mod.scale.shape == cov.shape
                mod.assign_cov(torch.from_numpy(cov))
        assert loaded_keys == set(mean_covs.keys())

    if 'fits_dict_file' in load_file:
        assert prior_w == prior.FixedCovGenNorm
        loaded_keys = set()
        _, fits_dict = pd.read_pickle(load_file['fits_dict_file'])
        prior_modules = {("net." + k[len("net.module."):-len(".p")]): k
                         for k in fits_dict.keys()}
        for name, mod in model.named_modules():
            if name not in prior_modules:
                continue
            key = prior_modules[name]
            loaded_keys.add(key)

            mod.beta[...] = fits_dict[key]["gennorm"][0]
            if isinstance(mod, prior.FixedCovGenNorm):
                mod.base_scale[...] = fits_dict[key]["gennorm"][2]
            else:
                assert isinstance(mod, prior.GenNorm)
                mod.loc[...] = fits_dict[key]["gennorm"][1]
                mod.scale[...] = fits_dict[key]["gennorm"][2]

        assert loaded_keys == set(fits_dict.keys())

    return model


def ThinPreActResNet18(softmax_temp=1.,
                       prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                       prior_b=prior.Normal, loc_b=0., std_b=1.,
                       scaling_fn=None, bn=True,
                       weight_prior_params={}, bias_prior_params={}):
    return ClassificationModel(PreActResNet(
        PreActBlock, [2,2,2,2], bn=bn,
        prior_w=prior_w, loc_w=loc_w, std_w=std_w,
        prior_b=prior_b, loc_b=loc_b, std_b=std_b,
        in_planes=16, scaling_fn=scaling_fn,
        weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params),
                               softmax_temp)


def PreActResNet34(softmax_temp=1.,
             prior_w=prior.Normal, loc_w=0., std_w=2**.5,
             prior_b=prior.Normal, loc_b=0., std_b=1.,
            scaling_fn=None, bn=True, weight_prior_params={}, bias_prior_params={}):
    return ClassificationModel(PreActResNet(PreActBlock,
                                        [3,4,6,3], bn=bn,
                                        prior_w=prior_w,
                                       loc_w=loc_w,
                                       std_w=std_w,
                                       prior_b=prior_b,
                                       loc_b=loc_b,
                                       std_b=std_b,
                                       scaling_fn=scaling_fn,
                                        weight_prior_params=weight_prior_params,
                                        bias_prior_params=bias_prior_params), softmax_temp)
