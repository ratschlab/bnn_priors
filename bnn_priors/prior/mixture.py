import torch.distributions as td
import torch
import math
from gpytorch.utils.transforms import inv_softplus

from .base import *
from .loc_scale import *
from .transformed import *
from .hierarchical import *
from .empirical_bayes import *
from . import conv_loc_scale


__all__ = ('Mixture', 'get_prior', 'ScaleMixture')


def get_prior(prior_name):
    priors = {"gaussian": Normal,
             "convcorrnormal": ConvCorrelatedNormal,
             "convcorrnormal_fitted_ls": ConvCorrelatedNormal,
             "convcorrnormal_empirical": ConvCorrNormalEmpirical,
             "convcorrnormal_gamma": ConvCorrNormalGamma,
             "datadrivencorrnormal": Normal,
             "datadrivencorrdoublegamma": DoubleGamma,
             "fixedcov_normal": conv_loc_scale.FixedCovNormal,
             "fixedcov_gennorm": conv_loc_scale.FixedCovGenNorm,
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
             "student-t_uniform": StudentTUniform,
             "gennorm": GenNorm,
             "gennorm_uniform": GenNormUniform,
             "gaussian_empirical": NormalEmpirical,
             "laplace_empirical": LaplaceEmpirical,
             "student-t_empirical": StudentTEmpirical,
             "gennorm_empirical": GenNormEmpirical,
             "scale_mixture": ScaleMixture,
             "mixture": Mixture,
             "scale_mixture_empirical": ScaleMixtureEmpirical}
    assert prior_name in priors
    return priors[prior_name]


class Mixture(LocScale):
    def __init__(self, shape, loc, scale, components="g_l_s_c_gn"):
        components=self.get_components(components)
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
                    "su": "student-t_uniform",
                    "gn": "gennorm",
                    "gnu": "gennorm_uniform",
                    "ge": "gaussian_empirical",
                    "le": "laplace_empirical",
                    "se": "student-t_empirical",
                    "gne": "gennorm_empirical"}
        abrvs = comp_string.split("_")
        assert all([abrv in comp_dict for abrv
                    in abrvs]), "Unknown mixture components"
        components = [comp_dict[abrv] for abrv in abrvs]
        return components


class ScaleMixture(Mixture):
    def __init__(self, shape, loc, scale, base_dist="gaussian", scales=None):
        if scales is None:
            self.scales = [scale/9, scale/3, scale, scale*3, scale*9]
        else:
            self.scales = scales
        super().__init__(shape, loc, scale)
        self.mixture_weights = torch.nn.Parameter(torch.zeros(len(self.scales)))
        self.components = [get_prior(base_dist)(shape, loc, scl)
                           for scl in self.scales]
        for comp in self.components:
            comp.p = self.p
            comp._old_log_prob = comp.log_prob
            # Prevent the sum over priors from double-counting this one
            comp.log_prob = (lambda: 0.)

        for i, comp in enumerate(self.components):
            self.add_module(f"component_{i}", comp)

        # Now that all parameters are initialized, sample properly
        self.sample()

        
class ScaleMixtureEmpirical(Mixture):
    def __init__(self, shape, loc, scale, base_dist="gaussian", scales=None):
        if scales is None:
            self.scales = [scale/9, scale/3, scale, scale*3, scale*9]
        else:
            self.scales = scales
        super().__init__(shape, loc, scale)
        self.mixture_weights = torch.nn.Parameter(torch.zeros(len(self.scales)))
        scale_priors = [PositiveImproper(shape=[], loc=scl, scale=1.) for scl in self.scales]
        for scale_prior, scl in zip(scale_priors, self.scales):
            with torch.no_grad():
                scale_prior.p.data = inv_softplus(torch.tensor(scl))
        self.components = [get_prior(base_dist)(shape, loc, scl)
                           for scl in scale_priors]
        for comp in self.components:
            comp.p = self.p
            comp._old_log_prob = comp.log_prob
            # Prevent the sum over priors from double-counting this one
            comp.log_prob = (lambda: 0.)

        for i, comp in enumerate(self.components):
            self.add_module(f"component_{i}", comp)

        # Now that all parameters are initialized, sample properly
        self.sample()
