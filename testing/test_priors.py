import unittest
import torch
import numpy as np
import torch.distributions as td

from bnn_priors import prior
from scipy import stats
from .test_eff_dim import requires_float64


def _generic_logp_test(prior_class, size, td_class=NotImplemented, np_logpdf=None, **kwargs):
    dist = prior_class(size, **kwargs)
    if np_logpdf is None:
        tdist = td_class(**kwargs)
        t_logpdf = tdist.log_prob
    else:
        @torch.no_grad()
        def t_logpdf(x):
            return torch.tensor(np_logpdf(x.detach().numpy()))
    assert np.allclose(dist.log_prob().item(), t_logpdf(dist()).sum().item())


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
def _generic_sample_test(prior_class, td_class=NotImplemented, n_samples=100000, seed=123, np_cdf=None, **kwargs):
    torch.manual_seed(seed)
    dist = prior_class(torch.Size([n_samples]), **kwargs)
    if np_cdf is None:
        td_dist = td_class(**kwargs)
        @torch.no_grad()
        def np_cdf(x):
            return td_dist.cdf(torch.from_numpy(x)).numpy()
    _, p = stats.ks_1samp(dist().detach(), np_cdf, mode='exact')
    assert p > 0.3


def _generic_multivariate_test(prior_class, N, atol_mean, atol_cov, **kwargs):
    loc = torch.tensor([1., 2., 3., 4.])
    cov = torch.randn(4, 4)
    cov = cov @ cov.t()
    dist = prior_class((N, 2, 2), loc=loc, cov=cov, **kwargs)

    p = dist().view((-1, 4))
    mean = p.mean(0)
    assert torch.allclose(mean, loc.to(p), atol=atol_mean)

    b = p - mean
    empirical_cov = (b.t() @ b) / len(b)
    assert torch.allclose(empirical_cov, cov.to(p), atol=atol_cov)


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
        _generic_sample_test(prior.Gamma, np_cdf=np_cdf, concentration=concentration, rate=rate)
        _generic_positive_test(prior.Gamma, concentration=concentration, rate=rate)

        concentration = torch.Tensor([0.2, 0.5])
        rate = torch.Tensor([1.7, 2.1])
        _generic_logp_test(prior.Gamma, torch.Size([2, 2]), td.Gamma,
                           concentration=concentration, rate=rate)

    def test_double_gamma(self):
        concentration = 0.3
        scale = 1.4
        loc = -0.4
        ls = dict(loc=loc, scale=scale)
        def np_cdf(x):
            return stats.dgamma.cdf(x, a=concentration, **ls)
        _generic_sample_test(prior.DoubleGamma, np_cdf=np_cdf,
                             concentration=concentration, **ls)
        _generic_positive_test(prior.DoubleGamma, concentration=concentration, **ls)

        def np_logpdf(x):
            return stats.dgamma.logpdf(x, a=concentration, **ls)
        _generic_logp_test(prior.DoubleGamma, torch.Size([2, 2]),
                           np_logpdf=np_logpdf, concentration=concentration,
                           **ls)

    def test_log_normal(self):
        _generic_sample_test(prior.LogNormal, td.LogNormal, loc=0.3, scale=1.2)
        _generic_positive_test(prior.LogNormal, loc=0.3, scale=1.2)

        loc = torch.Tensor([0.2, 0.5])
        scale = torch.Tensor([1.7, 2.1])
        _generic_logp_test(prior.LogNormal, torch.Size([2, 2]), td.LogNormal,
                           loc=loc, scale=scale)

    def test_gen_norm(self):
        loc = 0.3
        scale = 1.2
        beta = 0.7
        def np_cdf(x):
            return stats.gennorm.cdf(x, beta=beta, loc=loc, scale=scale)
        _generic_sample_test(prior.GenNorm, np_cdf=np_cdf, loc=loc, scale=scale,
                             beta=beta)
        _generic_positive_test(prior.GenNorm, loc=loc, scale=scale, beta=beta)

        def np_logpdf(x):
            return stats.gennorm.logpdf(x, beta=beta, loc=loc, scale=scale)
        _generic_logp_test(prior.GenNorm, torch.Size([2, 2]),
                           np_logpdf=np_logpdf, loc=loc, scale=scale, beta=beta)

    def test_johnson_su(self):
        loc = -0.5
        scale = 1.5
        inner_loc = 0.4
        inner_scale = 0.7

        kwargs = dict(loc=loc, scale=scale, inner_loc=inner_loc, inner_scale=inner_scale)
        np_kwargs = dict(loc=loc, scale=scale, a=inner_loc, b=inner_scale)

        def np_cdf(x):
            return stats.johnsonsu.cdf(x, **np_kwargs)
        _generic_sample_test(prior.JohnsonSU, np_cdf=np_cdf, **kwargs)
        _generic_positive_test(prior.JohnsonSU, **kwargs)

        def np_logpdf(x):
            return stats.johnsonsu.logpdf(x, **np_kwargs)
        _generic_logp_test(prior.JohnsonSU, torch.Size([2, 2]),
                           np_logpdf=np_logpdf, **kwargs)


    def test_loc_scale_log_prob(self):
        size = torch.Size([3, 2])
        loc = torch.Tensor([0.4, -1.3])
        scale = torch.Tensor([1.5, 2.3])

        _generic_logp_test(prior.Normal, size, td.Normal, loc=loc, scale=scale)
        _generic_logp_test(prior.Laplace, size, td.Laplace, loc=loc, scale=scale)
        _generic_logp_test(prior.Cauchy, size, td.Cauchy, loc=loc, scale=scale)
        _generic_logp_test(prior.StudentT, size, td.StudentT, df=3, loc=loc, scale=scale)

        assert prior.Improper(size, loc, scale).log_prob() == 0.0


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
        _generic_sample_test(prior.StudentT, np_cdf=np_cdf, df=3, loc=loc, scale=scale)
        _generic_positive_test(prior.StudentT, df=3, loc=loc, scale=scale)

        _generic_sample_test(prior.Improper, td.Normal, loc=loc, scale=scale)
        _generic_positive_test(prior.Improper, loc=loc, scale=scale)


    def test_fixed_cov_normal(self):
        torch.manual_seed(102)
        _generic_multivariate_test(prior.FixedCovNormal, 200000, 0.01, 0.01)

    @requires_float64
    def test_fixed_cov_normal_density(self):
        torch.manual_seed(102)
        loc = torch.tensor([1., 2., 3., 4.])
        cov = torch.randn(4, 4)
        cov = cov @ cov.t()
        dist = prior.FixedCovNormal((10, 2, 2), loc=loc, cov=cov)

        lp_torch = td.MultivariateNormal(loc, cov).log_prob(dist().view((-1, 4)))
        assert np.allclose(lp_torch.sum().item(), dist.log_prob().item())

    def test_fixed_cov_laplace(self):
        torch.manual_seed(102)
        _generic_multivariate_test(prior.FixedCovLaplace, 400000, 0.01, 0.01)

    def test_fixed_cov_double_gamma(self):
        torch.manual_seed(102)
        _generic_multivariate_test(
            prior.FixedCovDoubleGamma, 400000, 0.01, 0.01, concentration=0.3)
