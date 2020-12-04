import torch
import math
import torch.distributions as td

from . import distributions
from .base import Prior
from numbers import Number


__all__ = ('MultivariateT', 'ConvMultivariateT')

class ReshapeTransform(td.Transform):
    domain = td.constraints.real
    codomain = td.constraints.real
    bijective = True

    def __init__(self, in_event_shape, out_event_shape, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.event_dim = len(out_event_shape)
        self.out_event_shape = out_event_shape
        self.in_event_shape = in_event_shape

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros((), device=x.device, dtype=x.dtype)

    def _call(self, x):
        batch_shape = x.size()[:-len(self.in_event_shape)]
        return x.view(batch_shape + self.out_event_shape)

    def _inverse(self, y):
        batch_shape = y.size()[:-len(self.out_event_shape)]
        return y.view(batch_shape + self.in_event_shape)


class MultivariateT(Prior):
    def __init__(self, shape, loc, cov, df=3, event_dim=None):
        if event_dim is None:
            event_dim = len(shape)
        event_shape = torch.Size(shape)[len(shape)-event_dim:]
        assert len(event_shape) == event_dim

        if isinstance(cov, Number) or len(cov.shape) == 0:
            cov = torch.zeros([1, 1]) + cov ** 2  # it is std_dev
            loc = torch.zeros([1]) + loc
        super().__init__(shape=shape, loc=loc, cov=cov, df=df,
                         event_shape=event_shape)

    def _dist(self, loc, cov, df, event_shape):
        return td.TransformedDistribution(
            distributions.MultivariateT(df=df, loc=loc, covariance_matrix=cov,
                                        event_dim=len(event_shape)),
            ReshapeTransform(torch.Size([event_shape.numel()]), event_shape))

class ConvMultivariateT(Prior):
    pass
