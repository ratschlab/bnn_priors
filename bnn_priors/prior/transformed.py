
import torch.distributions as td
import torch
import math

from .base import Prior

__all__ = ('Uniform',)


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
        log_prob = -torch.log(self.high - self.low)
        # Account for broadcasting log_prob across self.p
        multiplier = self.p.numel() / log_prob.numel()
        return log_prob.sum() * multiplier

    def _sample_value(self, shape: torch.Size):
        return torch.randn(shape)

