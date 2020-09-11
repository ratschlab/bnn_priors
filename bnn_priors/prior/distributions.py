import numpy as np
from numbers import Number
import torch
from torch.distributions import constraints, Gamma
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from scipy import stats


__all__ = ('GeneralizedNormal',)


class GeneralizedNormal(Distribution):
    r"""
    Creates a Generalized Normal distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`beta`.

    Example::

        >>> m = GeneralizedNormal(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor(0.5))
        >>> m.sample()  # GeneralizedNormal distributed with loc=0, scale=1, beta=0.5
        tensor([ 0.1337])

    Args:
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
        beta (float or Tensor): shape parameter of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'beta': constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (self.scale.pow(2) * torch.lgamma(3/self.beta).exp()) / torch.lgamma(1/self.beta).exp()

    @property
    def stddev(self):
        return self.variance()**0.5

    def __init__(self, loc, scale, beta, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        (self.beta,) = broadcast_all(beta)
        self.scipy_dist = stats.gennorm(loc=self.loc.cpu().detach().numpy(),
                            scale=self.scale.cpu().detach().numpy(),
                            beta=self.beta.cpu().detach().numpy())
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(GeneralizedNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GeneralizedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(GeneralizedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def sample(self, sample_shape=torch.Size()):
        return torch.tensor(self.scipy_dist.rvs(list(sample_shape)),
                            dtype=self.loc.dtype, device=self.loc.device)


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (-torch.log(2 * self.scale) - torch.lgamma(1/self.beta) + torch.log(self.beta)
                - torch.pow((torch.abs(value - self.loc) / self.scale), self.beta))


    def cdf(self, value):
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        return torch.tensor(self.scipy_dist.cdf(value),
                            dtype=self.loc.dtype, device=self.loc.device)


    def icdf(self, value):
        raise NotImplementedError


    def entropy(self):
        return (1/self.beta) - torch.log(self.beta) + torch.log(2*self.scale) + torch.lgamma(1/self.beta)
