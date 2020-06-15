import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import constraints, Distribution, Normal
from typing import Callable
import contextlib
import collections
from collections import OrderedDict


class PriorMixin:
    "Common to all of our BNN priors, which should also be `torch.nn.Module`s"

    def potential(self, x, y):
        "log p(y, params in the module | x)"
        lik = self(x).log_prob(y).sum()
        for _, prior, parameter in self.named_priors():
            lik.add_(prior.log_prob(parameter).sum())
        return -lik

    def get_potential(self, x, y):
        def potential_with_params(params):
            "log p(y, nested_params | x)"
            with self.using_params(params):
                return self.potential(x, y)
        return potential_with_params

    def _overwrite_parameters(self, params):
        # TODO: keep old parameters that don't have a prior
        self._old_parameters = self._parameters
        self._parameters = new_parameters = OrderedDict()

        nested_params = collections.defaultdict(lambda: OrderedDict(), {})
        for k, v in params.items():
            first, *rest = k.split(".")
            if len(rest) == 0:
                new_parameters[first] = v
            else:
                nested_params[first][".".join(rest)] = v

        for k, v in nested_params.items():
            PriorMixin._overwrite_parameters(self._modules[k], v)

    def _restore_parameters(self):
        if '_old_parameters' in self.__dict__:
            self._parameters = self._old_parameters
            del self._old_parameters
        for m in self.children():
            PriorMixin._restore_parameters(m)

    @contextlib.contextmanager
    def using_params(self, params):
        "Uses the parameters stored in `nested_params` instead of the ones in this Module"
        try:
            self._overwrite_parameters(params)
            yield
        finally:
            self._restore_parameters()

    def path_getattr(self, name):
        path = name.split('.')
        out = getattr(self, path[0])
        if len(path) > 1:
            return PriorMixin.path_getattr(out, path[1:])
        return out

    def _sample_from_prior(self, prior, parameter):
        parameter.data.copy_(prior.sample())

    def sample_from_prior(self, prior_name):
        """Samples from the prior `prior_name`, and sets the value of the corresponding
        parameter
        """
        assert prior_name.endswith("_prior")
        name = prior_name[:-6]

        prior = self.path_getattr(prior_name)
        parameter = self.path_getattr(name)
        self._sample_from_prior(prior, parameter)

    def sample_all_priors(self):
        "Samples from all the priors in this model, and sets the value of the corresponding parameters"
        for _, prior, parameter in self.named_priors():
            self._sample_from_prior(prior, parameter)

    def register_prior(self, name: str, prior_dist: Callable[[], Distribution]):
        path = name.split('.')
        if len(path) > 1:
            return PriorMixin.register_prior(getattr(self, path[0]), ".".join(path[1:]), prior)
        parameter = getattr(self, name)
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(
                f"Cannot add prior {prior} to parameter {parameter}: not a "
                "`torch.nn.Parameter`")
        prior_name = f"{name}_prior"
        prior_dist = prior()
        sample_shape = prior_dist.batch_shape + prior_dist.event_shape
        if sample_shape != parameter.size():
            raise TypeError(
                f"Prior's shape {sample_shape} different from parameter's shape "
                f"{parameter.size()}")
        return setattr(self, prior_name, prior)

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

    def params_with_prior(self, clone=True, recurse=True):
        "returns nested dict of the params with priors"
        for n, _, param in self.named_priors(recurse=recurse):
            p = param.data.detach()
            if clone:
                p = p.clone()
            yield n[:-6], p

    def params_with_prior_dict(self, clone=True, recurse=True):
        return OrderedDict(self.params_with_prior(clone, recurse))


class Linear(nn.Linear, PriorMixin):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        for n, p in list(self.named_parameters(recurse=False)):
            loc_name = n+"_prior_loc"
            scale_name = n+"_prior_scale"
            self.register_buffer(
                loc_name, torch.tensor(0.).expand(p.size()))
            self.register_buffer(
                scale_name, torch.tensor(1.).expand(p.size()))
            self.register_prior(
                n,
                lambda: Normal(getattr(self, loc_name), getattr(self, scale_name)))


class DenseNet(nn.Module, PriorMixin):
    def __init__(self, in_dim, out_dim, width):
        super().__init__()
        self.lin1 = Linear(in_dim, width, bias=True)
        self.lin2 = Linear(width, width, bias=True)
        self.lin3 = Linear(width, out_dim, bias=True)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return Normal(x, torch.ones_like(x))
