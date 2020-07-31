import torch.distributions as td
import torch
import math

from .base import Prior

__all__ = ('LocScale', 'Normal', 'Laplace', 'Cauchy', 'StudentT', 'Uniform', 'LogNormal')


class LocScale(Prior):
    """Prior with a `loc` and `scale` parameters, implemented as a reparameterised
    version of loc=0 and scale=1.

    Arguments:
       shape (torch.Size): shape of the parameter
       loc (float, torch.Tensor, prior.Prior): location
       scale (float, torch.Tensor, prior.Prior): scale
    """
    def __init__(self, shape, loc, scale):
        super().__init__(shape, loc, scale)
        self.loc = loc
        self.scale = scale

    def forward(self):
        return self.p


class Normal(LocScale):
    _dist = td.Normal

    
class LogNormal(LocScale):
    _dist = td.LogNormal


class Laplace(LocScale):
    _dist = td.Laplace


class Cauchy(LocScale):
    _dist = td.Cauchy


class StudentT(LocScale):
    _dist = td.StudentT
    def __init__(self, shape, loc, scale, df=1):
        Prior.__init__(self, shape, df=df, loc=loc, scale=scale)
        self.loc = loc
        self.scale = scale


class Uniform(LocScale):
    """Uniform prior. Implemented as a Gaussian R.V., that is transformed through
    its own CDF.

    Arguments:
       shape: (torch.Size): shape of the parameter
       low (float, torch.Tensor, prior.Prior): lower bound of the Uniform
       high (float, torch.Tensor, prior.Prior): upper bound of the Uniform
    """
    _dist = td.Normal
    def __init__(self, shape, low, high):
        super().__init__(shape, low, high)
        self.low = low
        self.high = high
