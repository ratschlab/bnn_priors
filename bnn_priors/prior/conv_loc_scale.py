"""
Multivariate loc_scale priors for convolutional layers
"""
import torch.distributions as td
import torch
import math

from .base import Prior, value_or_call
from .distributions import GeneralizedNormal
from . import loc_scale
from .transformed import DoubleGamma


__all__ = ('FixedCovNormal', 'FixedCovLaplace', 'FixedCovDoubleGamma')


class ConvCovarianceMixin:
    def __init__(self, shape, loc, cov, **kwargs):
        vals, vecs = torch.symeig(cov.to(torch.float64), eigenvectors=True)
        sqrt_vals = vals.sqrt()
        scale = sqrt_vals.unsqueeze(-1) * vecs.t()  # PCA whitening
        log_sqrt_vals = sqrt_vals.log()

        dt = torch.get_default_dtype()
        Prior.__init__(self, shape, loc=loc, scale=scale.to(dt),
                       scale_for_logdet=log_sqrt_vals.to(dt),
                       **kwargs)

    def forward(self):
        flat_p = self.p.reshape(self.p.shape[:-2] + (-1,))
        trf_p = flat_p @ self.scale + self.loc
        return trf_p.view(self.p.size())

    def _dist(self, **kwargs):
        # Ignore native location and scale
        return self._base_dist(loc=0., scale=1.)

    def log_prob(self) -> torch.Tensor:
        log_diag = value_or_call(self.scale_for_logdet)
        c = log_diag.sum() * (self.p.numel() / log_diag.numel())
        return super().log_prob() - c


class FixedCovNormal(ConvCovarianceMixin, loc_scale.Normal):
    _base_dist = loc_scale.Normal._dist

class FixedCovLaplace(ConvCovarianceMixin, loc_scale.Laplace):
    def _dist(self, **kwargs):
        # scale=sqrt(1/2) makes it have variance=1
        return td.Laplace(loc=0., scale=math.sqrt(1/2))

class FixedCovDoubleGamma(ConvCovarianceMixin, DoubleGamma):
    def __init__(self, shape, loc, cov, concentration):
        ConvCovarianceMixin.__init__(self, shape, loc, cov, concentration=concentration)

    def _dist(self, concentration, **kwargs):
        """
        Mean of Gamma (with rate=1): concentration
        var. of Gamma (with rate=1): concentration
        thus, var. of DoubleGamma (with rate=1): concentration**2 + concentration

        thus, stddev = sqrt(concentration*(1+concentration)) / rate
        """
        rate = (concentration * (1+concentration)) ** .5
        return td.Gamma(concentration=concentration, rate=rate)
