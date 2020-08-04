import unittest
import torch
import numpy as np
import torch.distributions as td

from bnn_priors import prior
from scipy import stats
from .test_eff_dim import requires_float64


def _generic_logp_test(prior_klass, size, td_klass, **kwargs):
    dist = prior_klass(size, **kwargs)
    tdist = td_klass(**kwargs)
    assert torch.allclose(dist.log_prob(), tdist.log_prob(dist()).sum())


class PriorTest(unittest.TestCase):
    @requires_float64
    def test_uniform_dist(self, n_samples=100000, bins=64):
        "test whether the Uniform prior has the correct distribution"
        torch.manual_seed(123)
        dist = prior.Uniform(torch.Size([n_samples]), low=0., high=1.)
        samples = dist().detach()  # already samples in initialization
        _, p = stats.ks_1samp(samples, stats.uniform.cdf, mode='exact')
        assert p > 0.3

    def test_uniform_log_prob(self):
        low = torch.Tensor([0.2, 0.5])
        high = torch.Tensor([1.7, 2.1])
        _generic_logp_test(prior.Uniform, torch.Size([3, 2]), td.Uniform, low=low, high=high)


    def test_loc_scale_log_prob(self):
        size = torch.Size([3, 2])
        loc = torch.Tensor([0.4, -1.3])
        scale = torch.Tensor([1.5, 2.3])

        _generic_logp_test(prior.Normal, size, td.Normal, loc=loc, scale=scale)
        _generic_logp_test(prior.LogNormal, size, td.LogNormal, loc=loc, scale=scale)
        _generic_logp_test(prior.Laplace, size, td.Laplace, loc=loc, scale=scale)
        _generic_logp_test(prior.Cauchy, size, td.Cauchy, loc=loc, scale=scale)

        _generic_logp_test(prior.StudentT, size, td.StudentT, df=3, loc=loc, scale=scale)
