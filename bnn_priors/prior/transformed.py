import torch.distributions as td
import torch
import math
from gpytorch.utils.transforms import inv_softplus
from . import distributions

from .base import Prior

__all__ = ('Uniform', 'Gamma', 'HalfCauchy', 'JohnsonSU', 'DoubleGamma')


class Uniform(Prior):
    """Uniform prior. Implemented as a Gaussian R.V., that is transformed through
    its own CDF.

    Not implemented as a `torch.distributions.TransformedDistribution` because
    we can easily calculate log_prob(y).

    Arguments:
       shape: (torch.Size): shape of the parameter
       low (float, torch.Tensor, prior.Prior): lower bound of the Uniform
       high (float, torch.Tensor, prior.Prior): upper bound of the Uniform

    """
    _dist = td.Uniform
    def __init__(self, shape, low, high):
        super().__init__(shape, low=low, high=high)

    def forward(self):
        uniform = td.Normal(0., 1.).cdf(self.p)
        return self.low + (self.high-self.low) * uniform

    def log_prob(self):
        """in this case, calculating log_prob(forward(x)) directly is easier than
        calculating log_prob(x) + log(abs(det( dx/dy ))).
        """
        distance = self.high - self.low
        if isinstance(distance, float):
            return -math.log(distance) * self.p.numel()

        log_prob = -torch.log(distance)
        # Account for broadcasting log_prob across self.p
        multiplier = self.p.numel() / log_prob.numel()
        return log_prob.sum() * multiplier

    def _sample_value(self, shape: torch.Size):
        return torch.randn(shape)


class Gamma(Prior):
    _dist = td.Gamma
    def __init__(self, shape, concentration, rate):
        super().__init__(shape, concentration=concentration, rate=rate)

    def _sample_value(self, shape: torch.Size):
        x = super()._sample_value(shape)
        return inv_softplus(x)

    def forward(self):
        return torch.nn.functional.softplus(self.p)

    def log_prob(self):
        return self._dist_obj().log_prob(self()).sum()
    
    
class HalfCauchy(Prior):
    _dist = td.HalfCauchy
    def __init__(self, shape, scale=1., multiplier=1.):
        super().__init__(shape, scale=scale)
        self.multiplier = multiplier

    def _sample_value(self, shape: torch.Size):
        x = super()._sample_value(shape)
        return inv_softplus(x)

    def forward(self):
        return torch.nn.functional.softplus(self.p) * self.multiplier

    def log_prob(self):
        return self._dist_obj().log_prob(self()).sum()


class JohnsonSU(Prior):
    _dist = NotImplemented
    def __init__(self, shape, loc, scale, inner_loc=0., inner_scale=1.):
        super().__init__(shape, loc=loc, scale=scale, inner_loc=inner_loc,
                         inner_scale=inner_scale)

    def log_prob(self):
        p = (self.p - self.loc) / self.scale
        p_sq = p**2
        log_div = torch.log1p(p_sq).sum()

        y = self.inner_loc + self.inner_scale * torch.log(p + torch.sqrt(p_sq + 1))
        base_lp = td.Normal(0., 1.).log_prob(y).sum()

        # Account for broadcasting the scales
        multiplier = y.numel() / self.inner_scale.numel()
        inner_scale_mul = self.inner_scale.log().sum() * multiplier

        multiplier = y.numel() / self.scale.numel()
        scale_mul = self.scale.log().sum() * multiplier

        return base_lp - .5*log_div - scale_mul + inner_scale_mul

    def _sample_value(self, shape: torch.Size):
        x = (torch.randn(shape) - self.inner_loc) / self.inner_scale
        return self.loc + self.scale * torch.sinh(x)



class DoubleGamma(Prior):
    def __init__(self, shape, loc, scale, concentration):
        super().__init__(shape, loc=loc, scale=scale, concentration=concentration)

    def _dist(self, loc, scale, concentration):
        return distributions.DoubleGamma(concentration=concentration, rate=1/scale)

    def _sample_value(self, shape: torch.Size):
        x = super()._sample_value(shape)
        return x + self.loc

    def log_prob(self):
        return self._dist_obj().log_prob(self.p - self.loc).sum()
