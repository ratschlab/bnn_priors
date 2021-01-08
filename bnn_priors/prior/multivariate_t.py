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
        event_shape = torch.Size(shape)[len(shape)-event_dim:]
        assert len(event_shape) == event_dim

        if isinstance(scale_tril, Number) or isinstance(loc, Number):
            scale_tril = torch.ones([1, 1]) * scale_tril
            loc = torch.zeros([1]) + loc

        correlation_len = scale_tril.shape[-1]
        if event_shape[-1] != 1 and correlation_len == 1:
            correlation_shape = torch.Size([*event_shape, 1])
        else:
            assert event_shape.numel() == loc.size(-1)
            assert event_shape.numel() == scale_tril.size(-1)

        # if correlation_dim == 0:
        #     mvt_shape = torch.Size([1])
        # else:
        #     mvt_shape = torch.Size([event_shape[-correlation_dim:].numel()])

        super().__init__(shape=shape, loc=loc, scale_tril=scale_tril, df=df,
                         event_shape=event_shape, permute=permute)

    def _dist(self, loc, scale_tril, df, event_shape, permute):
        return td.TransformedDistribution(
            distributions.MultivariateT(df=df, loc=loc, scale_tril=scale_tril,
                                        event_dim=len(event_shape)),
            ReshapeTransform(torch.Size([event_shape.numel()]), event_shape,
                             permute=permute))
