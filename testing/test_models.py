import gpytorch
import torch
import unittest
from gpytorch.distributions import MultivariateNormal

from bnn_priors.models import RaoBDenseNet, RaoBLinearRegression
from .utils import requires_float64


class TestRaoBDenseNet(unittest.TestCase):
    @requires_float64
    def test_likelihood(self):
        x = torch.randn(10, 3)*2
        y = torch.randn(10, 1)*2

        model = RaoBDenseNet(x, y, 40, noise_std=0.8)
        device = next(iter(model.parameters())).device
        x = x.to(device)
        y = y.to(device)

        lik1 = model.log_likelihood(x, y, len(x))

        f = model.net(x) * model.last_layer_std
        noise = model.noise_std**2 * torch.eye(x.size(0), dtype=f.dtype, device=f.device)
        dist = MultivariateNormal(torch.zeros_like(y[:, 0]), f@f.t() + noise)

        lik2 = dist.log_prob(y.t()).sum()

        assert torch.allclose(lik1, lik2)

    @requires_float64
    def test_posterior(self):
        torch.manual_seed(2)
        x = torch.randn(10, 3)*2
        y = x @ torch.randn(x.size(1))

        class LinearGP(gpytorch.models.ExactGP):
            def __init__(self, x, y):
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood.noise = 0.5**2
                return super().__init__(x, y, likelihood)

            def forward(self, x):
                mean = torch.zeros(torch.Size([x.size(0)]),
                                   dtype=x.dtype, device=x.device)
                return MultivariateNormal(mean, gpytorch.lazy.RootLazyTensor(x))

        gp = LinearGP(x, y)
        model = RaoBLinearRegression(
            x, y.unsqueeze(1), noise_std=0.5, std_w=x.size(1)**.5)

        x_test = torch.eye(x.size(1), dtype=x.dtype, device=x.device)
        gp.eval()
        gp_w = gp(x_test)
        mu, L = model.posterior_w()

        assert torch.allclose(gp_w.mean, mu.squeeze(1))
        assert torch.allclose(gp_w.covariance_matrix, L.t()@L)
