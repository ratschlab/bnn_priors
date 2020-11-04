"""
Multivariate loc_scale priors for convolutional layers
"""
from numbers import Number
import torch.distributions as td
import torch
import math

from .base import Prior
from . import distributions


__all__ = ('ConvCovariance', 'FixedCovNormal', 'FixedCovLaplace', 'FixedCovDoubleGamma', 'FixedCovGenNorm')


class PCATransform(td.Transform):
    domain = td.constraints.real
    codomain = td.constraints.real
    event_dim = 2
    bijective = True

    def __init__(self, loc, scale, inv_scale, log_det, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale = scale
        self.inv_scale = inv_scale
        self._log_det = log_det

    def log_abs_det_jacobian(self, x, y):
        return self._log_det

    def _call(self, x):
        flat_x = x.view(x.shape[:-2] + (-1,))
        trf_x = flat_x @ self.scale + self.loc
        return trf_x.view(x.shape)

    def _inverse(self, y):
        flat_y = y.view(y.shape[:-2] + (-1,))
        trf_y = (flat_y - self.loc) @ self.inv_scale
        return trf_y.view(y.shape)


class ConvCovariance(Prior):
    def __init__(self, shape, loc, cov, **kwargs):
        if isinstance(cov, Number) or len(cov.shape) == 0:
            cov = torch.eye(shape[-2]*shape[-1]) * cov**2  # it is std_dev
            loc = torch.zeros(shape[-2]*shape[-1]) + loc

        scale, inv_scale, log_sqrt_vals = self._break_down_cov(cov)
        dt = torch.get_default_dtype()
        super().__init__(
            shape, loc=loc, scale=scale.to(dt), inv_scale=inv_scale.to(dt),
            log_sqrt_vals=log_sqrt_vals.to(dt),
            event_shape=shape[-2:], **kwargs)

    def _break_down_cov(self, cov):
        vals, vecs = torch.symeig(cov.to(torch.float64), eigenvectors=True)
        sqrt_vals = vals.sqrt()
        scale = sqrt_vals.unsqueeze(-1) * vecs.t()  # PCA whitening
        inv_scale = vecs / sqrt_vals
        log_sqrt_vals = vals.log().sum().view((1, 1)) / 2
        return scale, inv_scale, log_sqrt_vals

    def assign_cov(self, cov):
        scale, inv_scale, log_sqrt_vals = self._break_down_cov(cov)
        self.scale.copy_(scale)
        self.inv_scale.copy_(inv_scale)
        self.log_sqrt_vals.copy_(log_sqrt_vals)


class FixedCovNormal(ConvCovariance):
    def __init__(self, shape, loc, cov):
        super().__init__(shape, loc, cov)

    def _dist(self, loc, scale, inv_scale, log_sqrt_vals, event_shape):
        zeros = torch.zeros((), device=loc.device, dtype=loc.dtype).expand(event_shape)
        return td.TransformedDistribution(
            td.Normal(zeros, zeros+1),
            PCATransform(loc, scale, inv_scale, log_sqrt_vals))


class FixedCovLaplace(ConvCovariance):
    def __init__(self, shape, loc, cov, base_scale=math.sqrt(1/2)):
        # base_scale=sqrt(1/2) makes it have variance=1
        super().__init__(shape, loc, cov, base_scale=base_scale)

    def _dist(self, loc, scale, inv_scale, log_sqrt_vals, base_scale, event_shape):
        zeros = torch.zeros((), device=loc.device, dtype=loc.dtype).expand(event_shape)
        return td.TransformedDistribution(
            td.Laplace(loc=zeros, scale=base_scale.expand(event_shape)),
            PCATransform(loc, scale, inv_scale, log_sqrt_vals))


class FixedCovDoubleGamma(ConvCovariance):
    def __init__(self, shape, loc, cov, concentration, base_scale=None):
        """
        Mean of Gamma (with rate=1): concentration
        var. of Gamma (with rate=1): concentration
        thus, var. of DoubleGamma (with rate=1): concentration**2 + concentration

        thus, stddev = sqrt(concentration*(1+concentration)) / rate
        """
        if base_scale is None:
            base_rate = (concentration * (1+concentration)) ** .5
        else:
            base_rate = 1./base_scale
        super().__init__(shape, loc, cov, concentration=concentration,
                         base_rate=base_rate)

    def _dist(self, loc, scale, inv_scale, log_sqrt_vals, concentration, base_rate, event_shape):
        return td.TransformedDistribution(
            # sum zeros to broadcast
            distributions.DoubleGamma(concentration.expand(event_shape), rate=base_rate.expand(event_shape)),
            PCATransform(loc, scale, inv_scale, log_sqrt_vals))


class FixedCovGenNorm(ConvCovariance):
    """
    TODO: This distribution samples slightly incorrectly, due to inaccuracies in
    evaluating the CDF. This is not important for inference so we'll let it pass.
    """
    def __init__(self, shape, loc, cov, beta, base_scale=None):
        if base_scale is None:
            if isinstance(beta, Number):
                beta = torch.tensor(beta, dtype=torch.float64)
            base_scale = (torch.lgamma(1/beta) - torch.lgamma(3/beta)).div(2).exp()
        super().__init__(shape, loc, cov, beta=beta, base_scale=base_scale.to(torch.get_default_dtype()))

    def _dist(self, loc, scale, inv_scale, log_sqrt_vals, beta, base_scale, event_shape):
        zeros = torch.zeros((), device=loc.device, dtype=loc.dtype).expand(event_shape)
        return td.TransformedDistribution(
            distributions.GeneralizedNormal(loc=zeros, scale=base_scale.expand(event_shape),
                                            beta=beta.expand(event_shape)),
            PCATransform(loc, scale, inv_scale, log_sqrt_vals))
