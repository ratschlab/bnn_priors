from numbers import Number
import torch
from torch.distributions import constraints, Gamma, MultivariateNormal
from torch.distributions.multivariate_normal import _batch_mv, _batch_mahalanobis
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, _standard_normal
from scipy import stats
import math


__all__ = ('GeneralizedNormal', 'DoubleGamma', 'MultivariateT')


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
    has_rsample = False

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale.pow(2) * (torch.lgamma(3/self.beta) - torch.lgamma(1/self.beta)).exp()

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
        sample_shape = sample_shape + self.loc.size()
        return torch.tensor(self.scipy_dist.rvs(
            list(sample_shape),
            random_state=torch.randint(2**32, ()).item()),  # Make deterministic if torch is seeded
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


class DoubleGamma(Gamma):
    mean = 0.
    @property
    def variance(self):
        return self.concentration * (1 + self.concentration) / self.rate.pow(2)

    def rsample(self, sample_shape=torch.Size()):
        x = super().rsample(sample_shape)
        sign = torch.randint(0, 2, x.size(), device=x.device, dtype=x.dtype).mul_(2).sub_(1)
        return x*sign

    def log_prob(self, value):
        return super().log_prob(value.abs()) - math.log(2)

    entropy = NotImplemented
    _log_normalizer = NotImplemented


class MultivariateT(MultivariateNormal):
    """
    Multivariate Student-t distribution, using hierarchical Gamma sampling.
    (see https://arxiv.org/abs/1402.4306)
    We only allow degrees of freedom > 2 for now,
    because otherwise the covariance is undefined.

    Uses the parameterization from Shah et al. 2014, which makes it covariance
    equal to the covariance matrix.
    """
    arg_constraints = {'df': constraints.positive,
                       'loc': constraints.real_vector,
                       'covariance_matrix': constraints.positive_definite,
                       'precision_matrix': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real
    has_rsample = True
    expand = NotImplemented

    def __init__(self,
                 event_shape: torch.Size,
                 df=3.,
                 loc=0.,
                 covariance_matrix=None,
                 precision_matrix=None,
                 scale_tril=None,
                 validate_args=None):
        super().__init__(loc=loc,
                         covariance_matrix=covariance_matrix,
                         precision_matrix=precision_matrix,
                         scale_tril=scale_tril,
                         validate_args=validate_args)

        # self._event_shape is inferred from the mean vector and covariance matrix.
        old_event_shape = self._event_shape
        if not len(event_shape) >= len(old_event_shape):
            raise NotImplementedError("non-elliptical MVT not in this class")
        assert len(event_shape) >= 1
        assert event_shape[-len(old_event_shape):] == old_event_shape

        # Cut dimensions from the end of `batch_shape` so the `total_shape` is
        # the same
        total_shape = list(self._batch_shape) + list(self._event_shape)
        self._batch_shape = torch.Size(total_shape[:-len(event_shape)])
        self._event_shape = torch.Size(event_shape)

        self.df, _ = broadcast_all(df, torch.ones(self._batch_shape))
        self.gamma = Gamma(concentration=self.df/2., rate=1/2)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)

        r_inv = self.gamma.rsample(sample_shape=sample_shape)
        scale = ((self.df-2) / r_inv).sqrt()
        # We want 1 gamma for every `event` only. The size of self.df and this
        # `.view` provide that
        scale = scale.view(scale.size() + torch.Size([1] * len(self._event_shape)))

        return self.loc + scale * _batch_mv(self._unbroadcasted_scale_tril, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        n_dim = len(self._event_shape)
        p = diff.size()[-n_dim:].numel()
        if n_dim > 1:
            M = M.sum(tuple(range(-n_dim+1, 0)))

        log_diag = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log()
        if n_dim > log_diag.dim():
            half_log_det = log_diag.sum() * (p / log_diag.numel())
        else:
            half_log_det = log_diag.sum(tuple(range(-n_dim, 0))) * (
                p / log_diag.size()[-n_dim:].numel())

        lambda_ = self.df - 2.
        lp = torch.lgamma((p+self.df)/2.) \
                - ((p/2.) * torch.log(math.pi * lambda_)) \
                - torch.lgamma(self.df / 2.) \
                - half_log_det \
                - ((self.df+p)/2.) * torch.log(1 + M/lambda_)
        return lp
