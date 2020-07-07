from .base import AbstractModel
from torch.distributions import Normal
from .. import prior
import torch
from typing import Sequence, Union
from torch import nn
import math


__all__ = ('GaussianUnivariateRegression', 'RaoBGaussianUnivariateRegression')


class GaussianUnivariateRegression(AbstractModel):
    """Model for regression with 1 output dimension using an independent Gaussian
    likelihood.

    Arguments:
       num_data: the total number of data points, for minibatching
       noise_var (float_like or Prior): the variance of the Gaussian output
                 likelihood
       latent_fn_modules: modules to evaluate to get the latent function
    """
    def __init__(self, num_data: int,
                 noise_std: Union[float, torch.Tensor, prior.Prior],
                 latent_fn_modules: Sequence[nn.Module]):
        super().__init__(num_data, latent_fn_modules)
        self.noise_std = noise_std

    def likelihood_dist(self, f: torch.Tensor):
        return Normal(f, prior.value_or_call(self.noise_std))


class RaoBGaussianUnivariateRegression(AbstractModel):
    """Rao-Blackwellised version of Gaussian univariate regression. It integrates
    out the weights of the last layer analytically during inference. The last
    layer has to have an iid Normal prior, with variance `last_layer_var`.

    This class is also useful to calculate the marginal likelihood of a small
    data set.
    """
    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor,
                 noise_std: float, last_layer_std: float,
                 latent_fn_modules: Sequence[nn.Module]):
        assert x_train.dim() == 2
        assert x_train.size(0) == y_train.size(0)
        assert y_train.size(1) == 1
        super().__init__(x_train.size(0), latent_fn_modules)
        self.x_train = x_train
        self.y_train = y_train
        self.noise_std = noise_std
        self.last_layer_std = last_layer_std

        self.log_likelihood_precomp = self._log_likelihood_precomp(y_train)

    def _log_likelihood_precomp(self, y):
        "N log(2π) + (N-F) log(σ²) + tr_YY/(D σ²)"
        N, D = y.shape
        n_feat = D

        sig = self.noise_std**2
        log_sig = 2*math.log(self.noise_std)

        tr_YY__Ds = (y.view(-1) @ y.view(-1)).item() / (D*sig)
        const = N*math.log(2*math.pi)
        a = (N-n_feat) * log_sig + tr_YY__Ds
        const_a = const+a
        assert type(const_a) == float
        return const_a

    def log_likelihood(self, x, y):
        """Evaluate the Bayesian linear regression likelihood conditional on
        self.latent_fn(x). The prior over weights is:
            p(w_ij) = Normal(0, 1)

        Thus the distribution of outputs is (remember, x is a matrix)
            p(y | x) = Normal(y | 0, xᵀx + σ²)

        Using the Woodbury lemma, this can be evaluated as
        -D/2 [N log(2π) + (N-F) log(σ²) + tr_YY/(D σ²)
              + log det(xxᵀ + σ²I) - yᵀxᵀ(xxᵀ + σ²I)⁻¹xy / σ²]
        where tr_YY = tr{YᵀY} = (∑_j y_jᵀy_j).

        The first line of the above expression is precomputed in
        `self.log_likelihood_precomp`

        (the terms with logs come from applying Woodbury to the log
        determinant, the trace(YYᵀ) and quadratic form come from applying
        Woodbury to the inverse quadratic form of the Gaussian)

        """
        assert x is None or torch.equal(x, self.x_train)
        assert y is None or torch.equal(y, self.y_train)

        x = self.x_train
        y = self.y_train
        f = self.latent_fn(x)

        N, D = y.shape
        N_, n_feat = f.shape
        assert N == N_
        sig = self.noise_std**2
        last_layer_var = self.last_layer_std**2

        FF = (f.t() @ f) * last_layer_var
        # Switch to float64 for the cholesky bit
        FF_sig = FF.to(torch.float64) + sig*torch.eye(n_feat, dtype=torch.float64, device=f.device)
        L = torch.cholesky(FF_sig)
        logdet = 2*L.diag().log().sum()

        Lfy = (f.t() @ y).to(torch.float64).triangular_solve(L, upper=False)
        Lfy_flat = Lfy.solution.t().view(-1)
        quad = (Lfy_flat @ Lfy_flat) * (last_layer_var/(D*sig))
        likelihood = (-D/2) * (self.log_likelihood_precomp + logdet - quad)
        # Round likelihood down to the original dtype
        likelihood = likelihood.to(f.dtype)
        return likelihood

    def _posterior_w(self, x, y):
        "returns mean and lower triangular precision of p(w | x,y)"
        f = self.latent_fn(x) * self.last_layer_std
        sig = self.noise_std**2
        # Precision matrix
        A = (f.t()@f)/sig + torch.eye(f.size(-1), dtype=f.dtype, device=f.device)
        # switch to float64
        L = torch.cholesky(A.to(torch.float64))
        FY = (f.t()@y).to(torch.float64)
        white_mean = FY.triangular_solve(L, upper=False).solution
        return white_mean/sig, L

    def likelihood_dist(self, f: torch.Tensor):
        white_w_mean, L_w = self._posterior_w(self.x_train, self.y_train)
        f = f * self.last_layer_std
        Lf = f.t().to(torch.float64).triangular_solve(L_w, upper=False).solution
        mean = Lf.t() @ white_w_mean
        var = torch.einsum("in,in->n", Lf, Lf)
        # Switch back to float32
        return Normal(mean.to(f), var.sqrt().to(f).unsqueeze(-1))

