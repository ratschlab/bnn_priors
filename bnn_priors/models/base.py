import math
from .. import prior
from torch import nn
import torch
import abc
from typing import List, Sequence, Dict, Union
from collections import OrderedDict
import contextlib

__all__ = ('RegressionModel', 'RaoBRegressionModel', 'CategoricalModel')


class AbstractModel(nn.Module, abc.ABC):
    """A model that can be used with our SGLD and Pyro MCMC samplers.

    Arguments:
       num_data: the total number of data points, for minibatching
       net: neural net to evaluate to get the latent function
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def log_prior(self):
        "log p(params)"
        return sum(p.log_prob() for _, p in prior.named_priors(self))

    @abc.abstractmethod
    def likelihood_dist(self, f: torch.Tensor):
        "representation of p(y | f, params)"
        pass

    def forward(self, x: torch.Tensor):
        "representation of p(y | x, params)"
        f = self.net(x)
        return self.likelihood_dist(f)

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor, eff_num_data):
        """
        unbiased minibatch estimate of log-likelihood
        log p(y | x, self.parameters)
        """
        # compute batch size using zeroth dim of inputs (log_prob_batch doesn't work with RaoB).
        assert x.shape[0] == y.shape[0]
        batch_size = x.shape[0]
        return self(x).log_prob(y).sum() * (eff_num_data/batch_size)

    def log_likelihood_avg(self, x: torch.Tensor, y: torch.Tensor, eff_num_data):
        """
        unbiased minibatch estimate of log-likelihood per datapoint
        """
        return self.log_likelihood(x, y, eff_num_data)/eff_num_data

    def potential(self, x, y, eff_num_data):
        """
        There are two subtly different ways of altering the "temperature".
        The Wenzel et al. approach is to apply a temperature (here, T) to both the prior and likelihood together.
        However, the VI approach is to in effect replicate each datapoint multiple times (data_mult)
        """
        return - self.log_likelihood(x, y, eff_num_data) + self.log_prior()

    def potential_avg(self, x, y, eff_num_data):
        "-log p(y, params | x)"
        return self.potential(x, y, eff_num_data) / eff_num_data

    def params_with_prior_dict(self):
        return OrderedDict(
            (k, v.data) for (k, v) in prior.named_params_with_prior(self))

    def sample_all_priors(self):
        for _, v in prior.named_priors(self):
            v.sample()

    # Following methods necessary for HMC but not SGLD

    def get_potential(self, x: torch.Tensor, y: torch.Tensor, eff_num_data = None):
        if eff_num_data is None:
            eff_num_data = x.shape[0]

        "returns (potential(param_dict) -> torch.Tensor)"
        def potential_fn(param_dict):
            "-log p(y, params | x)"
            with self.using_params(param_dict):
                return self.potential(x, y, eff_num_data)
        return potential_fn

    @contextlib.contextmanager
    def using_params(self, param_dict: Dict[str, torch.Tensor]):
        try:
            pmd = self._prior_mod_dict
        except AttributeError:
            pmd = self._prior_mod_dict = list(
                (k, v, v.p) for (k, v) in prior.named_priors(self))

        assert len(param_dict) == len(pmd)
        try:      # assign `torch.Tensor`s to `Prior.p`s
            for k, mod, _ in pmd:
                mod._parameters['p'] = param_dict[k]
            yield
        finally:  # Restore `Prior.p`s
            for k, mod, p in pmd:
                mod._parameters['p'] = p



class CategoricalModel(AbstractModel):
    def likelihood_dist(self, f: torch.Tensor):
        return torch.distributions.Categorical(logits=f)



class RegressionModel(AbstractModel):
    """Model for regression using an independent Gaussian likelihood.
    Arguments:
       num_data: the total number of data points, for minibatching
       noise_var (float_like or Prior): the variance of the Gaussian output
                 likelihood
       net: modules to evaluate to get the latent function
    """
    def __init__(self, net: nn.Module,
                 noise_std: Union[float, torch.Tensor, prior.Prior]):
        super().__init__(net)
        self.noise_std = noise_std

    def likelihood_dist(self, f: torch.Tensor):
        return torch.distributions.Normal(f, prior.value_or_call(self.noise_std))



class RaoBRegressionModel(AbstractModel):
    """Rao-Blackwellised version of Gaussian univariate regression. It integrates
    out the weights of the last layer analytically during inference. The last
    layer has to have an iid Normal prior, with variance `last_layer_var`.

    This class is also useful to calculate the marginal likelihood of a small
    data set.
    """
    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor,
                 noise_std: float, last_layer_std: float,
                 net: nn.Module):
        assert x_train.dim() == 2
        assert x_train.size(0) == y_train.size(0)
        assert y_train.size(1) == 1
        super().__init__(net)
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

    def log_likelihood(self, x, y, eff_num_data):
        """Evaluate the Bayesian linear regression likelihood conditional on
        self.net(x). The prior over weights is:
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
        f = self.net(x)

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
        f = self.net(x) * self.last_layer_std
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
        return torch.distributions.Normal(mean.to(f), var.sqrt().to(f).unsqueeze(-1))
