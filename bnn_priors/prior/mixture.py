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
    def __init__(self, shape, loc, scale, components=None):
        if components is None:
            components = ["gaussian", "laplace", "student-t", "cauchy"]
        assert len(components) > 0, "Too few mixture components"
        super().__init__(shape, loc, scale)
        self.mixture_weights = torch.nn.Parameter(torch.zeros(len(components)))
        self.components = [get_prior(comp)(shape, loc, scale)
                           for comp in components]
        for comp in self.components:
            comp.p = self.p
            comp._old_log_prob = comp.log_prob
            # Prevent the sum over priors from double-counting this one
            comp.log_prob = (lambda: 0.)

        for i, comp in enumerate(self.components):
            self.add_module(f"component_{i}", comp)

        # Now that all parameters are initialized, sample properly
        self.sample()
        
    _dist = NotImplemented
    def log_prob(self):
        """
        The mixture probability is defined without logs:

        prob(self) = sum(w * exp(comp._old_log_prob(self.p))
                         for w, comp in zip(self.mixture_weights, self.components))

        which we can rewrite as

        log_prob(self) = log_sum_exp(log_w + comp.old_log_prob(self.p)) - log_sum_exp(log_w)
        """
        normaliser = torch.logsumexp(self.mixture_weights, dim=0)
        log_ps = torch.stack([comp._old_log_prob() for comp in self.components])
        return torch.logsumexp(self.mixture_weights + log_ps, dim=0) - normaliser

    def _sample_value(self, shape: torch.Size):
        try:
            mixture_weights = self.mixture_weights
            components = self.components
        except AttributeError:
            return torch.randn(shape)  # Called before initialization of parameters

        idx = td.Categorical(logits=mixture_weights).sample().item()
        return components[idx]._sample_value(shape)
