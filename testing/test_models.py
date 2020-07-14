import gpytorch
import torch
import unittest
from gpytorch.distributions import MultivariateNormal

from bnn_priors.models import RaoBDenseNet


class TestRaoBDenseNet(unittest.TestCase):
    def test_likelihood(self):
        torch.set_default_dtype(torch.float64)

        x = torch.randn(10, 3)*2
        y = torch.randn(10, 1)*2

        model = RaoBDenseNet(x, y, 40, noise_std=0.8)

        lik1 = model.log_likelihood(x, y, len(x))

        f = model.net(x) * model.last_layer_std
        noise = model.noise_std**2 * torch.eye(x.size(0), dtype=f.dtype, device=f.device)
        dist = MultivariateNormal(torch.zeros_like(y[:, 0]), f@f.t() + noise)

        lik2 = dist.log_prob(y.t()).sum()

        assert torch.allclose(lik1, lik2)
