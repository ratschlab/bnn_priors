import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ClassificationModel
from .dense_nets import LinearPrior
from .conv_nets import Reshape, Conv2dPrior
from .. import prior

import pandas as pd
from pathlib import Path

__all__ = ('DataDrivenGaussianClassificationConvNet', 'DataDrivenDoubleGammaClassificationConvNet')

def DataDrivenGaussianClassificationConvNet(
        in_channels, img_height, out_features, width, depth=3, softmax_temp=1.,
        prior_w=prior.Normal, loc_w=0., std_w=2**.5,
        prior_b=prior.Normal, loc_b=0., std_b=1.,
        scaling_fn=None, weight_prior_params={}, bias_prior_params={}):
    def no_scaling(std, dim):
        return std
    assert depth == 3, "That's what we have data"

    mean_covs = pd.read_pickle(Path(__file__).parent/"mean_covs_mnist_classification.pkl.gz")
    # 2**(depth-1) is because of the MaxPool
    reshaped_size = width*(img_height//2**(depth-1))**2

    layers = [Reshape(-1, in_channels, img_height, img_height),
              Conv2dPrior(in_channels, width, kernel_size=3, padding=1,
                          prior_w=prior.FixedCovNormal,
                          loc_w=torch.from_numpy(mean_covs['net.module.1.weight_prior.p'][0]),
                          std_w=torch.from_numpy(mean_covs['net.module.1.weight_prior.p'][1]),
                          prior_b=prior.Normal,
                          loc_b=mean_covs['net.module.1.bias_prior.p'][0],
                          std_b=mean_covs['net.module.1.bias_prior.p'][1] ** .5,
                          scaling_fn=no_scaling),
              nn.ReLU(), nn.MaxPool2d(2),
              Conv2dPrior(width, width, kernel_size=3, padding=1,
                          prior_w=prior.FixedCovNormal,
                          loc_w=torch.from_numpy(mean_covs['net.module.4.weight_prior.p'][0]),
                          std_w=torch.from_numpy(mean_covs['net.module.4.weight_prior.p'][1]),
                          prior_b=prior.Normal,
                          loc_b=mean_covs['net.module.4.bias_prior.p'][0],
                          std_b=mean_covs['net.module.4.bias_prior.p'][1] ** .5,
                          scaling_fn=no_scaling),
              nn.ReLU(), nn.MaxPool2d(2),
              nn.Flatten(),
              LinearPrior(reshaped_size, out_features,
                          prior_w=prior.Normal,
                          loc_w=mean_covs['net.module.8.weight_prior.p'][0],
                          std_w=mean_covs['net.module.8.weight_prior.p'][1] ** .5,
                          prior_b=prior.Normal,
                          loc_b=mean_covs['net.module.8.bias_prior.p'][0],
                          std_b=mean_covs['net.module.8.bias_prior.p'][1] ** .5,
                          scaling_fn=no_scaling),
              ]
    return ClassificationModel(nn.Sequential(*layers), softmax_temp)


def DataDrivenDoubleGammaClassificationConvNet(
        in_channels, img_height, out_features, width, depth=3, softmax_temp=1.,
        prior_w=prior.Normal, loc_w=0., std_w=2**.5,
        prior_b=prior.Normal, loc_b=0., std_b=1.,
        scaling_fn=None, weight_prior_params={}, bias_prior_params={}):
    def no_scaling(std, dim):
        return std
    assert depth == 3, "That's what we have data"

    mean_covs = pd.read_pickle(Path(__file__).parent/"mean_covs_mnist_classification.pkl.gz")
    _, fits = pd.read_pickle(Path(__file__).parent/"fits_mnist_classification.pkl.gz")
    # 2**(depth-1) is because of the MaxPool
    reshaped_size = width*(img_height//2**(depth-1))**2

    layers = [Reshape(-1, in_channels, img_height, img_height),
              Conv2dPrior(in_channels, width, kernel_size=3, padding=1,
                          prior_w=prior.FixedCovLaplace,
                          loc_w=torch.from_numpy(mean_covs['net.module.1.weight_prior.p'][0]),
                          std_w=torch.from_numpy(mean_covs['net.module.1.weight_prior.p'][1]),
                          prior_b=prior.Normal,
                          loc_b=mean_covs['net.module.1.bias_prior.p'][0],
                          std_b=mean_covs['net.module.1.bias_prior.p'][1] ** .5,
                          scaling_fn=no_scaling),
              nn.ReLU(), nn.MaxPool2d(2),
              Conv2dPrior(width, width, kernel_size=3, padding=1,
                          prior_w=prior.FixedCovDoubleGamma,
                          loc_w=torch.from_numpy(mean_covs['net.module.4.weight_prior.p'][0]),
                          std_w=torch.from_numpy(mean_covs['net.module.4.weight_prior.p'][1]),
                          prior_b=prior.Normal,
                          loc_b=mean_covs['net.module.4.bias_prior.p'][0],
                          std_b=mean_covs['net.module.4.bias_prior.p'][1] ** .5,
                          scaling_fn=no_scaling,
                          weight_prior_params=dict(
                              concentration=fits["net.module.4.weight_prior.p"]["dgamma"][0],
                          )),
              nn.ReLU(), nn.MaxPool2d(2),
              nn.Flatten(),
              LinearPrior(reshaped_size, out_features,
                          prior_w=prior.DoubleGamma,
                          loc_w=fits['net.module.8.weight_prior.p']["dgamma"][1],
                          std_w=fits['net.module.8.weight_prior.p']["dgamma"][2],
                          prior_b=prior.Normal,
                          loc_b=mean_covs['net.module.8.bias_prior.p'][0],
                          std_b=mean_covs['net.module.8.bias_prior.p'][1] ** .5,
                          scaling_fn=no_scaling,
                          weight_prior_params=dict(
                              concentration=fits['net.module.8.weight_prior.p']["dgamma"][0],
                          )),
              ]
    return ClassificationModel(nn.Sequential(*layers), softmax_temp)
