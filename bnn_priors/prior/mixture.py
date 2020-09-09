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
    def __init__(self, shape, loc, scale, components="g_l_s_c"):
        components=self.get_components(components)
        super().__init__(shape, loc, scale)
        self.mixture_weights = torch.nn.Parameter(torch.zeros(len(components)))
        self.components = [get_prior(comp)(shape, loc, scale)
                           for comp in components]
        for comp in self.components:
            comp.p = self.p
        for i, comp in enumerate(self.components):
            self.add_module(f"component_{i}", comp)
        

    _dist = td.Normal
    

    def log_prob(self):
        """
        Return weighted sum of the components' log probs.
        """
        weights = torch.nn.functional.softmax(self.mixture_weights, dim=0)
        probs = [weight*comp.log_prob() for weight, comp in zip(weights, self.components)]
        return sum(probs)
    

    @staticmethod
    def get_components(comp_string):
        """
        Parse the component string into the components.
        """
        comp_dict = {"g": "gaussian",
                    "ln": "lognormal",
                    "l": "laplace",
                    "c": "cauchy",
                    "s": "student-t",
                    "u": "uniform",
                    "i": "improper",
                    "gg": "gaussian_gamma",
                    "gu": "gaussian_uniform",
                    "h": "horseshoe",
                    "lg": "laplace_gamma",
                    "lu": "laplace_uniform",
                    "sg": "student-t_gamma",
                    "su": "student-t_uniform"}
        abrvs = comp_string.split("_")
        assert all([abrv in comp_dict for abrv
                    in abrvs]), "Unknown mixture components"
        components = [comp_dict[abrv] for abrv in abrvs]
        return components
