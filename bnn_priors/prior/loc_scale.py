import torch.distributions as td
import torch
import math

from .base import Prior

__all__ = ('LocScale', 'Normal', 'Laplace', 'Cauchy', 'StudentT', 'Uniform')


class LocScale(Prior):
    """Prior with a `loc` and `scale` parameters, implemented as a reparameterised
    version of loc=0 and scale=1.

    Arguments:
       shape: (torch.Size): shape of the parameter
       loc: ()

    """
    def __init__(self, shape, loc, scale):
        super().__init__(shape, 0., 1.)
        self.loc = loc
        self.scale = scale

    def forward(self):
        return self.loc + self.scale * self.p


class Normal(LocScale):
    _dist = td.Normal


class Laplace(LocScale):
    _dist = td.Laplace


class Cauchy(LocScale):
    _dist = td.Cauchy


class StudentT(LocScale):
    _dist = td.StudentT


class Uniform(LocScale):
    _dist = td.Uniform
    def __init__(self, shape, low, high):
        super().__init__(self, low, high-low)
