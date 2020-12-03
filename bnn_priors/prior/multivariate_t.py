import torch
import math

from . import distributions
from .base import Prior
from numbers import Number


__all__ = ('MultivariateT', 'ConvMultivariateT')


class MultivariateT(Prior):
    def __init__(self, shape, loc, cov, df=3, event_dim=None):
        if isinstance(cov, Number) or len(cov.shape) == 0:
            cov = torch.eye(shape[-2]*shape[-1]) * cov**2  # it is std_dev
            loc = torch.zeros(shape[-2]*shape[-1]) + loc
        if event_dim is None:
            event_dim = len(shape)
        event_dim = (event_dim,)  # tuple to prevent conversion to Tensor
        super().__init__(shape=shape, loc=loc, cov=cov, df=df,
                         event_dim=event_dim)

    def _dist(self, loc, cov, df, event_dim):
        event_dim, = event_dim  # undo tuple
        return distributions.MultivariateT(
            df=df, loc=loc, covariance_matrix=cov, event_dim=event_dim)

class ConvMultivariateT(Prior):
    pass
