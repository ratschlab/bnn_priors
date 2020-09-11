import torch.distributions as td
import torch
import math
from gpytorch.utils.transforms import inv_softplus

from .base import Prior
from .transformed import Gamma, Uniform, HalfCauchy
from .distributions import GeneralizedNormal


__all__ = ('LocScale', 'Normal', 'Laplace', 'Cauchy', 'StudentT', 'LogNormal',
           'Improper', 'PositiveImproper', 'GenNorm')


class LocScale(Prior):
    """Prior with a `loc` and `scale` parameters, implemented as a reparameterised
    version of loc=0 and scale=1.

    Arguments:
       shape (torch.Size): shape of the parameter
       loc (float, torch.Tensor, prior.Prior): location
       scale (float, torch.Tensor, prior.Prior): scale
    """
    def __init__(self, shape, loc, scale):
        super().__init__(shape, loc=loc, scale=scale)


class Normal(LocScale):
    _dist = td.Normal


class Laplace(LocScale):
    _dist = td.Laplace


class Cauchy(LocScale):
    _dist = td.Cauchy


class StudentT(LocScale):
    _dist = td.StudentT
    def __init__(self, shape, loc, scale, df=2):
        Prior.__init__(self, shape, df=df, loc=loc, scale=scale)
        
        
class GenNorm(LocScale):
    _dist = GeneralizedNormal
    def __init__(self, shape, loc, scale, beta=0.5):
        Prior.__init__(self, shape, loc=loc, scale=scale, beta=beta)


class LogNormal(LocScale):
    _dist = td.Normal
    def forward(self):
        return self.p.exp()
    def log_prob(self):
        return super().log_prob() - self.p.sum()


class Improper(Normal):
    "Improper prior that samples like a Normal"
    def log_prob(self):
        return 0.0
    

class PositiveImproper(Improper):
    """Improper prior for positive things."""
    def forward(self):
        return torch.nn.functional.softplus(self.p)
