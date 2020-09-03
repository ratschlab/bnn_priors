import torch.distributions as td
import torch
import math
from gpytorch.utils.transforms import inv_softplus

from .base import *
from .loc_scale import *
from .transformed import *
from .hierarchical import *


__all__ = ('Mixture', 'get_prior')


def get_prior(prior_name):
    priors = {"gaussian": Normal,
             "lognormal": LogNormal,
             "laplace": Laplace,
             "cauchy": Cauchy,
             "student-t": StudentT,
             "uniform": Uniform,
             "improper": Improper,
             "gaussian_gamma": NormalGamma,
             "gaussian_uniform": NormalUniform,
             "horseshoe": Horseshoe,
             "laplace_gamma": LaplaceGamma,
             "laplace_uniform": LaplaceUniform,
             "student-t_gamma": StudentTGamma,
             "student-t_uniform": StudentTUniform}
    assert prior_name in priors
    return priors[prior_name]


class Mixture(LocScale):
    def __init__(self, shape, loc, scale, components):
        assert len(components) > 0, "Too few mixture components"
        super().__init__(shape, loc, scale)
        self.components = [get_prior(comp)(shape, loc, scale)
                           for comp in components]
        for i, comp in enumerate(self.components):
            self.add_module(f"component_{i}", comp)
        for comp in self.components:
            comp.p = self.p
        # import pdb; pdb.set_trace()
        
    _dist = td.Normal
    
    def log_prob(self):
        probs = [comp.log_prob() for comp in self.components]
        return sum(probs)
    
    
