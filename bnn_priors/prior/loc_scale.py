import torch.distributions as td
import torch
import numpy as np

from .base import Prior
from .distributions import GeneralizedNormal


__all__ = ('LocScale', 'Normal', 'ConvCorrelatedNormal', 'Laplace', 'Cauchy', 'StudentT', 'LogNormal',
           'Improper', 'PositiveImproper', 'GenNorm')


class SquaredExponentialNormal(td.MultivariateNormal):
    """Multivariate Normal with a squared exponential kernel as covariance"""
    def __init__(self, loc, scale, distance_matrix, lengthscale):
        cov = torch.exp(- distance_matrix / lengthscale) * scale ** 2.0
        chol = torch.cholesky(cov)
        super().__init__(loc=loc, scale_tril=chol)


class LocScale(Prior):
    """Prior with a `loc` and `scale` parameters, implemented as a reparameterised
    version of loc=0 and scale=1.

    Arguments:
       shape (torch.Size): shape of the parameter
       loc (float, torch.Tensor, prior.Prior): location
       scale (float, torch.Tensor, prior.Prior): scale
    """
    def __init__(self, shape, loc, scale):
        super().__init__(shape, loc=loc, scale=scale)


class Normal(LocScale):
    _dist = td.Normal


class ConvCorrelatedNormal(Prior):
    def __init__(self, shape, loc, scale, *, lengthscale=1.0):
        """Samples `.p` from a correlated Gaussian of dimension
        `shape[-2]*shape[-1]`"""
        if isinstance(loc, float) or len(loc.shape) == 0:
            # `MultivariateNormal` complains about zero-dim loc
            loc = torch.tensor(loc).unsqueeze(0)

        # generates all the points in a grid from (0,0) to (shape[-2], shape[-1])
        p = np.mgrid[:shape[-2], :shape[-1]].reshape(2, -1).T
        # computes the matrix of Euclidean distances between all the points in p
        d = np.sum((p[:, None, :] - p[None, :, :]) ** 2.0, 2) ** 0.5
        super().__init__(shape, loc=loc, scale=scale, distance_matrix=d,
                         lengthscale=lengthscale)

    def log_prob(self) -> torch.Tensor:
        return self._dist_obj().log_prob(self.p.reshape(self.p.shape[:-2] + (-1,))).sum()

    def _sample_value(self, shape: torch.Size):
        dist = self._dist_obj()

        return torch.reshape(dist.sample(sample_shape=shape[:-2]), shape)

    # we have to move the kernel evaluation into the _dist, so that the hierarchical versions of this prior work
    # for fixed parameters, we could consider precomputing it to save a bit of time here
    _dist = SquaredExponentialNormal


class Laplace(LocScale):
    _dist = td.Laplace


class Cauchy(LocScale):
    _dist = td.Cauchy


class StudentT(LocScale):
    _dist = td.StudentT
    def __init__(self, shape, loc, scale, df=3):
        Prior.__init__(self, shape, df=df, loc=loc, scale=scale)
        
        
class GenNorm(LocScale):
    _dist = GeneralizedNormal
    def __init__(self, shape, loc, scale, beta=0.5):
        Prior.__init__(self, shape, loc=loc, scale=scale, beta=beta)


class LogNormal(LocScale):
    _dist = td.Normal
    def forward(self):
        return self.p.exp()
    def log_prob(self):
        return super().log_prob() - self.p.sum()


class Improper(Normal):
    "Improper prior that samples like a Normal"
    def log_prob(self):
        return 0.0
    

class PositiveImproper(Improper):
    """Improper prior for positive things."""
    def forward(self):
        return torch.nn.functional.softplus(self.p)
