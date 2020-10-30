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
        inv_scale = vecs / sqrt_vals
        log_sqrt_vals = sqrt_vals.log()

        dt = torch.get_default_dtype()
        Prior.__init__(self, shape, loc=loc, scale=scale.to(dt),
                       scale_for_logdet=log_sqrt_vals.to(dt),
                       inv_scale=inv_scale.to(dt),
                       **kwargs)

    def _sample_value(self, shape: torch.Size):
        flat_shape = shape[:-2] + (shape[-2]*shape[-1],)
        flat_p = super()._sample_value(flat_shape)
        trf_p = flat_p @ self.scale + self.loc
        return trf_p.view(shape)

    def _dist(self, **kwargs):
        # Ignore native location and scale
        return self._base_dist(loc=0., scale=1.)

    def log_prob(self) -> torch.Tensor:
        flat_p = self.p.reshape(self.p.shape[:-2] + (-1,))
        white_p = (flat_p - self.loc) @ self.inv_scale
        p = white_p.view(self.p.size())

        log_diag = value_or_call(self.scale_for_logdet)
        c = log_diag.sum() * (self.p.numel() / log_diag.numel())

        return self._base_log_prob(p) - c

    def _base_log_prob(self, p):
        return self._dist_obj().log_prob(p).sum()


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

    def _base_log_prob(self, p):
        return self._dist_obj().log_prob(p.abs()).sum() - math.log(2)*p.numel()

    def _sample_value(self, shape: torch.Size):
        flat_shape = shape[:-2] + (shape[-2]*shape[-1],)
        flat_p = Prior._sample_value(self, flat_shape).to(self.scale)
        sign = torch.randint(0, 2, flat_shape, dtype=flat_p.dtype).mul_(2).sub_(1)
        trf_p = (flat_p*sign) @ self.scale + self.loc
        return trf_p.view(shape)
