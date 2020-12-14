import torch
import torch.nn as nn
import torch.nn.functional as F

from .google_resnet import ResNet
from .. import prior

import pandas as pd
from pathlib import Path


def DataDrivenMVTGoogleResNet(softmax_temp=1., depth=20, num_classes=10,
                              bn=True):
    assert depth == 20, "That's what we have data"

    mvt = pd.read_pickle(Path(__file__).parent/"cifar10_opt_mvt.pkl.gz")
    net = ResNet(softmax_temp=softmax_temp, depth=depth,
                 num_classes=num_classes,
                 prior_w=prior_w, loc_w=loc_w, std_w=std_w,
                 prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                 scaling_fn=scaling_fn, bn=bn,
                 weight_prior_params=weight_prior_params,
                 bias_prior_params=bias_prior_params, conv_prior_w=prior_w)
    named_modules = dict(net.named_modules())
    named_parameters = dict(net.named_parameters())

    for key, (_lik, params) in mvt.items():
        *parent, prior_mod_name, _p = key.split(".")
        parent = ".".join(parent)

        shape = named_parameters[key].shape
        is_conv = (len(shape) == 4)
        if is_conv and params["event_dim"] == "t":
            permute = (1, 0, 2, 3)
            event_dim = 2
        else:
            permute = None
            try:
                event_dim = int(params["event_dim"])
            except ValueError as e:
                raise ValueError(f"`event_dim` for key `{key}` raised {e}")
        df = F.softplus(params["df"]).item()

        named_modules[parent].register_module(
            prior_mod_name,
            prior.MultivariateT(shape, params["loc"], params["scale_tril"],
                                df=df, event_dim=event_dim, permute=permute))

    return net
