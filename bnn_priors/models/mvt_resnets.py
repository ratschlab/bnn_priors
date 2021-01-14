import torch
import torch.nn as nn
import torch.nn.functional as F

from .google_resnet import ResNet
from .. import prior

import numpy as np
from pathlib import Path


def DataDrivenMVTGoogleResNet(softmax_temp=1., depth=20, num_classes=10,
                              bn=True):
    assert depth == 20, "We only have data for depth=20"

    mvt = torch.load(Path(__file__).parent/"cifar10_opt_mvt.pkl")
    net = ResNet(softmax_temp=softmax_temp, depth=depth,
                 num_classes=num_classes, bn=bn)
    named_modules = dict(net.named_modules())
    named_parameters = dict(net.named_parameters())

    for key, (_lik, params) in mvt.items():
        *parent, prior_mod_name, _p = key.split(".")
        parent = list(parent)
        parent = ".".join(parent)

        shape = named_parameters[key].shape
        is_conv = (len(shape) == 4)
        if is_conv and params["event_dim"] == "t":
            permute = (1, 0, 2, 3)
            event_dim = 3
        else:
            permute = None
            try:
                event_dim = int(params["event_dim"])
            except ValueError as e:
                raise ValueError(f"`event_dim` for key `{key}` raised {e}")
        df = F.softplus(params["df"]).item()

        # the network still knows how to calculate forward passes after
        # replacing its Prior modules, because Prior modules are usually leaves,
        # and only contain information on how to calculate a parameter, not the
        # forward pass itself.
        named_modules[parent].add_module(
            prior_mod_name,
            prior.MultivariateT(shape, params["loc"], params["scale_tril"],
                                df=df, event_dim=event_dim, permute=permute))

    return net

def DecreasingMVTGoogleResNet(softmax_temp=1., depth=20, num_classes=10,
           prior_w=prior.ConvCorrelatedNormal, loc_w=0., std_w=2**.5,
           prior_b=prior.Normal, loc_b=0., std_b=1.,
           scaling_fn=None, bn=True, weight_prior_params={}, bias_prior_params={},
           dense_prior_w=prior.Normal):
    assert depth == 20, "We only have data for depth=20"

    net = ResNet(
        softmax_temp=softmax_temp, depth=depth, num_classes=num_classes,
        prior_w=prior_w, loc_w=loc_w, std_w=std_w,
        prior_b=prior_b, loc_b=loc_b, std_b=std_b,
        scaling_fn=scaling_fn, bn=bn, weight_prior_params=weight_prior_params,
        bias_prior_params=bias_prior_params,
        conv_prior_w=prior_w)

    named_modules = dict(net.named_modules())

    for key, parameter in prior.named_params_with_prior(net):
        *parent, prior_mod_name, _p = key.split(".")
        parent_key = ".".join(parent)

        shape = parameter.shape
        is_conv = (len(shape) == 4)
        if is_conv:
            permute = (1, 0, 2, 3)
            event_dim = 3
        else:
            permute = None
            event_dim = len(shape)

        df = {"0": 3.55, "3": 3.0, "4": 5.5, "5": 20.0, "6": 32., "7": 50.,
              "8": 60., "9": 70., "10": 80., "11": 90., "14": 1000.}[parent[1]]
        if df > 32.:
            pass  # Leave the Normal as it is
        else:
            in_dim = torch.Size(shape)[1:].numel()
            scale_tril = std_w / in_dim**0.5
            loc = loc_w

            if is_conv and shape[-1] != 1 and prior_w == prior.ConvCorrelatedNormal:
                loc = torch.zeros([1]) + loc

                lengthscale = 1.
                p = np.mgrid[:shape[-2], :shape[-1]].reshape(2, -1).T
                # computes the matrix of Euclidean distances between all the points in p
                distance_matrix = np.sum((p[:, None, :] - p[None, :, :]) ** 2.0, 2) ** 0.5
                distance_matrix = torch.tensor(distance_matrix).to(loc)
                kernel = torch.exp(- distance_matrix / lengthscale)

                scale_tril = kernel.cholesky() * scale_tril

            mod = prior.MultivariateT(shape, loc, scale_tril, df=df,
                                      event_dim=event_dim, permute=permute)
            # the network still knows how to calculate forward passes after
            # replacing its Prior modules, because Prior modules are usually leaves,
            # and only contain information on how to calculate a parameter, not the
            # forward pass itself.
            named_modules[parent_key].add_module(prior_mod_name, mod)
    return net
