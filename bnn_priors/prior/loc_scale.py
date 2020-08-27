import torch.distributions as td
import torch
import math
from gpytorch.utils.transforms import inv_softplus

from .base import Prior
from .transformed import Gamma

__all__ = ('LocScale', 'Normal', 'Laplace', 'Cauchy', 'StudentT', 'LogNormal',
           'Improper', 'NormalGamma')


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
    
    
class NormalGamma(Normal):
    def __init__(self, shape, loc, scale, rate=1., gradient_clip=1.):
        scale_prior = Gamma(shape=[], concentration=scale, rate=rate)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale_prior)
        self.scale.p.register_hook(self.hook)
        self.clip = gradient_clip
        
    def log_prob(self):
        # TODO: it seems like the log prob is too high compared to the other loss terms...
        return super().log_prob() + self.scale.log_prob()
    
    def hook(self, grad):
        # TODO: This somehow affects the downstream gradients of the parameters, which it shouldn't
        # It should only affect the actual scale.p parameter
        return torch.clamp(grad, -self.clip, self.clip)
