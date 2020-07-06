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

    def prior_logprob(self):
        "log p(self.params)"
        logprob = None
        for _, prior, parameter in self.named_priors():
            lp = prior(parameter).log_prob(parameter).sum()
            if logprob is None:
                logprob = lp
            else:
                logprob.add_(lp)
        return logprob

    def log_likelihood(self, x, y):
        "log p(y | self.params, x)"
        return self(x).log_prob(y).sum()

    def potential(self, x, y):
        "-log p(y, self.params | x)"
        return -self.log_likelihood(x, y) - self.prior_logprob()

    def get_potential(self, x, y):
        def potential_with_params(params):
            "-log p(y, params | x)"
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
        parameter.data.copy_(prior(parameter).sample())

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

    def register_prior(self, name: str, prior: Callable[[], Distribution]):
        path = name.split('.')
        if len(path) > 1:
            return PriorMixin.register_prior(getattr(self, path[0]), ".".join(path[1:]), prior)
        parameter = getattr(self, name)
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(
                f"Cannot add prior {prior} to parameter {parameter}: not a "
                "`torch.nn.Parameter`")
        prior_name = f"{name}_prior"
        prior_dist = prior(parameter)
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

    def params_with_prior(self, clone=False, recurse=True):
        "returns nested dict of the params with priors"
        for n, _, param in self.named_priors(recurse=recurse):
            if clone:
                param = param.data.detach().clone()
            yield n[:-6], param

    def params_with_prior_dict(self, clone=False, recurse=True):
        return OrderedDict(self.params_with_prior(clone, recurse))


class Linear(nn.Linear, PriorMixin):
    def __init__(self, in_features, out_features, bias=True, weight_prior=None, bias_prior=None):
        super().__init__(in_features, out_features, bias=bias)
        if weight_prior is None:
            def weight_prior(p): return Normal(
                    torch.zeros_like(p), torch.ones_like(p))
        if bias_prior is None:
            def bias_prior(p): return Normal(
                    torch.zeros_like(p), torch.ones_like(p))
        self.register_prior("weight", weight_prior)
        self.register_prior("bias", bias_prior)


class DenseNet(nn.Module, PriorMixin):
    def __init__(self, in_dim, out_dim, width, output_std=1.,
                 weight_prior=None, bias_prior=None):
        super().__init__()
        self.output_std = output_std
        self.lin1 = Linear(in_dim, width, bias=True,
                           weight_prior=weight_prior,
                           bias_prior=bias_prior)
        self.lin2 = Linear(width, width, bias=True,
                           weight_prior=weight_prior,
                           bias_prior=bias_prior)
        self.lin3 = Linear(width, out_dim, bias=True,
                           weight_prior=weight_prior,
                           bias_prior=bias_prior)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return Normal(x, torch.ones_like(x) * self.output_std)
