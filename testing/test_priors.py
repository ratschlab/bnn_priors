import unittest
import torch
import numpy as np
import torch.distributions as td

from bnn_priors import prior
from scipy import stats
from .test_eff_dim import requires_float64


def _generic_logp_test(prior_class, size, td_class, **kwargs):
    dist = prior_class(size, **kwargs)
    tdist = td_class(**kwargs)
    assert torch.allclose(dist.log_prob(), tdist.log_prob(dist()).sum())


def _generic_positive_test(prior_class, **kwargs):
    dist = prior_class(torch.Size([]), **kwargs)
    with torch.no_grad():
        dist.p[...] = -100.
    v = dist()
    assert torch.isfinite(v)
    v.backward()
    assert torch.isfinite(dist.p.grad)
    assert torch.isfinite(torch.as_tensor(dist.log_prob()))

@requires_float64
@torch.no_grad()
def _generic_sample_test(prior_class, td_class, n_samples=100000, seed=123, np_cdf=None, **kwargs):
    torch.manual_seed(seed)
    dist = prior_class(torch.Size([n_samples]), **kwargs)
    td_dist = td_class(**kwargs)

    if np_cdf is None:
        @torch.no_grad()
        def np_cdf(x):
            return td_dist.cdf(torch.from_numpy(x)).numpy()
    _, p = stats.ks_1samp(dist().detach(), np_cdf, mode='exact')
    assert p > 0.3


class PriorTest(unittest.TestCase):
    def test_uniform(self):
        _generic_sample_test(prior.Uniform, td.Uniform, low=-0.3, high=1.7)
        _generic_positive_test(prior.Uniform, low=-0.2, high=1.2)

        low = torch.Tensor([0.2, 0.5])
        high = torch.Tensor([1.7, 2.1])
        _generic_logp_test(prior.Uniform, torch.Size([3, 2]), td.Uniform, low=low, high=high)

    def test_gamma(self):
        concentration = 0.3
        rate = 1.2
        def np_cdf(x):
            return stats.gamma.cdf(x, a=concentration, scale=1/rate)
        _generic_sample_test(prior.Gamma, td.Gamma, np_cdf=np_cdf, concentration=concentration, rate=rate)
        _generic_positive_test(prior.Gamma, concentration=concentration, rate=rate)

        concentration = torch.Tensor([0.2, 0.5])
        rate = torch.Tensor([1.7, 2.1])
        _generic_logp_test(prior.Gamma, torch.Size([2, 2]), td.Gamma,
                           concentration=concentration, rate=rate)

    def test_log_normal(self):
        _generic_sample_test(prior.LogNormal, td.LogNormal, loc=0.3, scale=1.2)
        _generic_positive_test(prior.LogNormal, loc=0.3, scale=1.2)

        loc = torch.Tensor([0.2, 0.5])
        scale = torch.Tensor([1.7, 2.1])
        _generic_logp_test(prior.LogNormal, torch.Size([2, 2]), td.LogNormal,
                           loc=loc, scale=scale)

    def test_loc_scale_log_prob(self):
        size = torch.Size([3, 2])
        loc = torch.Tensor([0.4, -1.3])
        scale = torch.Tensor([1.5, 2.3])

        _generic_logp_test(prior.Normal, size, td.Normal, loc=loc, scale=scale)
        _generic_logp_test(prior.Laplace, size, td.Laplace, loc=loc, scale=scale)
        _generic_logp_test(prior.Cauchy, size, td.Cauchy, loc=loc, scale=scale)
        _generic_logp_test(prior.StudentT, size, td.StudentT, df=3, loc=loc, scale=scale)


    def test_loc_scale_sample_positive(self):
        size = torch.Size([])
        loc = -0.3
        scale = 1.5

        _generic_sample_test(prior.Normal, td.Normal, loc=loc, scale=scale)
        _generic_positive_test(prior.Normal, loc=loc, scale=scale)

        _generic_sample_test(prior.Laplace, td.Laplace, loc=loc, scale=scale)
        _generic_positive_test(prior.Laplace, loc=loc, scale=scale)

        _generic_sample_test(prior.Cauchy, td.Cauchy, loc=loc, scale=scale)
        _generic_positive_test(prior.Cauchy, loc=loc, scale=scale)

        def np_cdf(x):
            return stats.t.cdf(x, df=3, loc=loc, scale=scale)
        _generic_sample_test(prior.StudentT, td.StudentT, np_cdf=np_cdf, df=3, loc=loc, scale=scale)
        _generic_positive_test(prior.StudentT, df=3, loc=loc, scale=scale)
