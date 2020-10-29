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


__all__ = ('FixedCovNormal', 'FixedCovLaplace')


class ConvCovarianceMixin:
    def __init__(self, shape, loc, cov):
        vals, vecs = torch.symeig(cov, eigenvectors=True)
        sqrt_vals = vals.sqrt()
        scale = sqrt_vals.unsqueeze(-1) * vecs.t()  # PCA whitening
        Prior.__init__(self, shape, loc=loc, scale=scale,
                       scale_for_logdet=sqrt_vals)

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
