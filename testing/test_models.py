import gpytorch
import torch
import unittest
from gpytorch.distributions import MultivariateNormal

from bnn_priors.models import RaoBDenseNet


class TestRaoBDenseNet(unittest.TestCase):
    def test_likelihood(self):
        torch.set_default_dtype(torch.float64)

        x = torch.randn(10, 3)*2
        y = torch.randn(10, 2)*2

        model = RaoBDenseNet(x.size(-1), y.size(-1), 40, output_std=0.8)

        lik1 = model.log_likelihood(x, y)

        f = model.nn_eval(x)
        noise = model.output_std**2 * torch.eye(x.size(0), dtype=f.dtype, device=f.device)
        dist = MultivariateNormal(torch.zeros_like(y[:, 0]), f@f.t() + noise)

        lik2 = dist.log_prob(y.t()).sum()

        assert torch.allclose(lik1, lik2)
