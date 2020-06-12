from torch.distributions import Distribution
from pyro.distributions import Normal
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Any
import pyro

from pyro.nn.module import to_pyro_module_, PyroSample


class PriorMixin:
    def get_potential_fn(self, X, y):
        def potential_fn(params):
            for n, p in self.named_parameters():
                p.data = params[n]
                if n == 'lin3.bias':
                    print(n, p, params[n])
            for n, p in self.named_parameters():
                if n == 'lin3.bias':
                    print(n, p, params[n])

            # self.load_state_dict(params)  # Copy params
            loss = self(X).log_prob(y).sum()
            # loss = torch.zeros((), dtype=X.dtype, device=X.device)
            for _, prior, parameter in self.named_priors():
                loss.add_(prior.log_prob(parameter).sum())
                # loss.add_(prior.log_prob(params[prior_name[:-6]]).sum())
            return loss
        return potential_fn

    def path_getattr(self, name):
        path = name.split('.')
        out = getattr(self, path[0])
        if len(path) > 1:
            return PriorMixin.path_getattr(out, path[1:])
        return out

    def _sample_from_prior(self, prior, parameter):
        parameter.data.copy_(prior.sample())

    def sample_from_prior(self, prior_name):
        assert prior_name.endswith("_prior")
        name = prior_name[:-6]

        prior = self.path_getattr(prior_name)
        parameter = self.path_getattr(name)
        self._sample_from_prior(prior, parameter)

    def sample_all_priors(self):
        for _, prior, parameter in self.named_priors():
            self._sample_from_prior(prior, parameter)

    def register_prior(self, name: str, prior: Callable[[nn.Parameter], Distribution]):
        path = name.split('.')
        if len(path) > 1:
            return PriorMixin.register_prior(getattr(self, path[0]), ".".join(path[1:]), prior)

        parameter = getattr(self, name)
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(f"Cannot add prior {prior} to parameter {parameter}")
        prior_name = f"{name}_prior"
        prior_dist = prior(parameter)
        if not isinstance(prior_dist, Distribution):
            raise TypeError(f"prior {prior} returned not a Distribution, {prior_dist}")
        return setattr(self, prior_name, prior_dist)

    def named_priors(self, prefix="", recurse=True):
        for n, parameter in self.named_parameters(recurse=False):
            prior_name = f"{n}_prior"
            if hasattr(self, prior_name):
                out_name = ("." if prefix else "").join([prefix, prior_name])
                yield out_name, getattr(self, prior_name), parameter

        if recurse:
            for mname, module_ in self.named_children():
                submodule_prefix = prefix + ("." if prefix else "") + mname
                for it in PriorMixin.named_priors(module_, prefix=submodule_prefix, recurse=True):
                    yield it


class Linear(nn.Linear, PriorMixin):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        for n, _ in self.named_parameters(recurse=False):
            self.register_prior(n, lambda p: Normal(torch.zeros_like(p), torch.ones_like(p)))


class DenseNet(nn.Module, PriorMixin):
    def __init__(self, in_dim, out_dim, width):
        super().__init__()
        self.lin1 = Linear(in_dim, width, bias=True)
        # self.lin2 = Linear(width, width, bias=True)
        self.lin3 = Linear(width, out_dim, bias=True)

    def forward(self, x, y):
        x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        x = self.lin3(x)
        # return Normal(x, torch.ones_like(x))
        return pyro.sample('y', Normal(x, torch.ones_like(x)), obs=y)

    def forward_normal(self, x):
        x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        return self.lin3(x)



def make_dense(*args, **kwargs):
    model = DenseNet(*args, **kwargs)
    to_pyro_module_(model)

    for m in model.modules():
        for name, prior, value in list(m.named_priors(recurse=False)):
            setattr(m, name[:-6], PyroSample(prior=Normal(0, 1).expand(value.shape).to_event(value.dim())))
    return model
