import torch
import torch.nn as nn
import itertools
import abc
from typing import Dict, Sequence
import contextlib

__all__ = ('value_or_call', 'Prior', 'named_priors', 'named_params_with_prior',
           'params_with_prior')


def value_or_call(vs):
    if callable(vs):
        return vs()
    return vs


class Prior(nn.Module, abc.ABC):
    """Prior distribution over a parameter of type Float/Double. class._dist
    should be overwritten with a `torch.distribution.Distribution` object.

    Arguments:
       shape: (torch.Size, tuple): shape of the parameter
       *args, **kwargs: arguments to construct the underlying class `Prior._dist`

    TODO: register all args as tensors if they're not Parameters or Modules.
    """
    def __init__(self, shape: torch.Size, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.p = nn.Parameter(self._sample_value(shape))

        for key, arg in itertools.chain(enumerate(args), kwargs.items()):
            assert str(key) != "p", "repeated name of parameter"
            if isinstance(arg, nn.Module):
                self.add_module(str(key), arg)
            elif isinstance(arg, nn.Parameter):
                self.register_parameter(str(key), arg)
            elif isinstance(arg, torch.Tensor):
                self.register_buffer(str(key), arg)
            else:
                setattr(self, str(key), arg)


    @staticmethod
    @abc.abstractmethod
    def _dist(*args, **kwargs):
        pass

    def dist(self):
        return self._dist(*map(value_or_call, self.args),
                          **{k: value_or_call(v) for k, v in self.kwargs.items()})

    def log_prob(self):
        return self.dist().log_prob(self.p).sum()

    def _sample_value(self, shape: torch.Size):
        dist = self.dist()
        dim = len(dist.event_shape)
        if dim != 0:
            shape = shape[:-dim]
        return dist.sample(sample_shape=shape)

    @torch.no_grad()
    def sample(self):
        self.p.data = self._sample_value(self.p.size()).to(self.p.data)
        self.p.grad = None

    def forward(self):
        """The value of parameter that has this prior. For reparameterised priors, this
        computes the forward function.
        """
        return self.p


def named_priors(mod: nn.Module):
    "iterate over all `Prior`s in `mod`"
    return filter(lambda kv: isinstance(kv[1], Prior), mod.named_modules())


def named_params_with_prior(mod: nn.Module):
    """iterate over all parameters that have a `Prior` specified, and their names
    """
    return ((k+("p" if k == "" else ".p"), v.p) for (k, v) in named_priors(mod))


def params_with_prior(mod: nn.Module):
    "iterate over all parameters that have a `Prior` specified"
    return (v.p for _, v in named_priors(mod))
