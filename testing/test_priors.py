import unittest
import torch
import numpy as np

from bnn_priors import prior
from scipy.stats import binom


def two_sided_binomial_p(x, n, p):
    """Two-sided p-value for the value `x`, under a Binomial distribution with
    `n` trials and success probability `p`.

    p = 2 min(Pr(X ≤ x), Pr(X ≥ x))
    """
    p_rv_leq_x = binom.cdf(x, n, p)
    p_rv_geq_x = 1-binom.cdf(x-1, n, p)
    # These do not sum to 1 because x has a finite probability.
    p = 2*np.minimum(p_rv_leq_x, p_rv_geq_x)
    return p


class PriorTest(unittest.TestCase):
    def test_uniform_dist(self, n_samples=100000, bins=64):
        "test whether the Uniform prior has the correct distribution"
        dist = prior.Uniform(torch.Size([n_samples]), low=0., high=1.)
        samples = dist()  # already samples in initialization

        with torch.no_grad():
            hist = torch.histc(samples, bins=bins, min=0., max=1.)
            hist = hist.numpy().astype(int)
        p = two_sided_binomial_p(hist, n_samples, 1/bins)
        """
        Sum of p-values still a valid p-value. Checking that p-values are large
        is not a rigorous exercise, but it works, just like MCMC diagnostics.

        If the distribution is slightly wrong (e.g. use the Sigmoid function as
        transformation instead of the Gaussian CDF) the p-value is very small.
        """
        assert p.sum().item() > 20.

