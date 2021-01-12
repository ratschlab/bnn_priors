import torch
import math
import torch.distributions as td

from . import distributions
from .base import Prior
from numbers import Number


__all__ = ('MultivariateT',)

class ReshapeTransform(td.Transform):
    domain = td.constraints.real
    codomain = td.constraints.real
    bijective = True

    def __init__(self, in_event_shape, out_event_shape, permute=None, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.event_dim = len(out_event_shape)
        self.out_event_shape = out_event_shape
        self.in_event_shape = in_event_shape
        self.permute = permute

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros((), device=x.device, dtype=x.dtype)

    def _call(self, x):
        batch_shape = x.size()[:-len(self.in_event_shape)]
        if self.permute is not None:
            x = x.permute(*self.permute)
        return x.view(batch_shape + self.out_event_shape)

    def _inverse(self, y):
        batch_shape = y.size()[:-len(self.out_event_shape)]
        y = y.view(batch_shape + self.in_event_shape)
        if self.permute is not None:
            y = y.permute(*self.permute)  # permutation-lists are their own inverse
        return y


class MultivariateT(Prior):
    def __init__(self, shape, loc, scale_tril, df=3, event_dim=None,
                 permute=None):
        if event_dim is None:
            event_dim = len(shape)
        assert event_dim >= 1
        out_event_shape = torch.Size(shape)[len(shape)-event_dim:]
        assert len(out_event_shape) == event_dim

        if isinstance(scale_tril, Number) or isinstance(loc, Number):
            scale_tril = torch.ones([1, 1]) * scale_tril
            loc = torch.zeros([1]) + loc

        correlation_size = scale_tril.size(-1)
        assert loc.size(-1) == correlation_size

        if correlation_size == 1:
            if out_event_shape[-1] == 1:
                event_shape = out_event_shape
            else:
                event_shape = torch.Size([*out_event_shape, 1])
        else:
            # put dimensions of out_event_shape together until they are as large
            # as the covariance matrix / mean.
            size = 1
            coincides = False
            # iterate starting from the end
            for i, size_i in reversed(list(enumerate(out_event_shape))):
                size *= size_i
                if size == correlation_size:
                    coincides = True
                    last_idx = i
                    break
            assert coincides
            event_shape = torch.Size([*out_event_shape[:last_idx], correlation_size])

        super().__init__(shape=shape, loc=loc, scale_tril=scale_tril, df=df,
                         event_shape=event_shape,
                         out_event_shape=out_event_shape, permute=permute)

    def _dist(self, loc, scale_tril, df, event_shape, out_event_shape, permute):
        return td.TransformedDistribution(
            distributions.MultivariateT(df=df, loc=loc, scale_tril=scale_tril,
                                        event_shape=event_shape),
            ReshapeTransform(event_shape, out_event_shape, permute=permute))
