import torch
import torch.nn as nn
import abc
import numpy as np
from numbers import Number


__all__ = ('value_or_call', 'Prior', 'named_priors', 'named_params_with_prior')


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
    """
    def __init__(self, shape: torch.Size, **kwargs):
        super().__init__()
        self.kwargs_keys = list(kwargs.keys())
        for key, arg in kwargs.items():
            assert str(key) != "p", "repeated name of parameter"
            if isinstance(arg, Number):
                arg = torch.tensor(arg, dtype=torch.get_default_dtype())
            elif isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg).to(torch.get_default_dtype())

            if isinstance(arg, nn.Module):
                self.add_module(str(key), arg)
            elif isinstance(arg, nn.Parameter):
                self.register_parameter(str(key), arg)
            elif isinstance(arg, torch.Tensor):
                self.register_buffer(str(key), arg)
            else:
                setattr(self, str(key), arg)

        # `self._sample_value` uses kwargs
        self.p = nn.Parameter(self._sample_value(shape))


    @staticmethod
    @abc.abstractmethod
    def _dist(*args, **kwargs):
        pass

    def _dist_obj(self):
        return self._dist(**{k: value_or_call(getattr(self, k))
                             for k in self.kwargs_keys})

    def log_prob(self) -> torch.Tensor:
        return self._dist_obj().log_prob(self.p).sum()

    def _sample_value(self, shape: torch.Size):
        dist = self._dist_obj()
        dim = len(dist.event_shape) + len(dist.batch_shape)
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
