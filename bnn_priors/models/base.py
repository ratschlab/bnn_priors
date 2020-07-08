from .. import prior
from torch import nn
import torch
import abc
from typing import List, Sequence, Dict
from collections import OrderedDict
import contextlib

__all__ = ('AbstractModel',)


class AbstractModel(nn.Module, abc.ABC):
    """A model that can be used with our SGLD and Pyro MCMC samplers.

    Arguments:
       num_data: the total number of data points, for minibatching
       latent_fn_modules: modules to evaluate to get the latent function
    """
    def __init__(self, num_data: int,
                 latent_fn_modules: Sequence[nn.Module]):
        super().__init__()
        self.num_data = num_data
        if len(latent_fn_modules) == 1:
            self.latent_fn = latent_fn_modules[0]
        else:
            self.latent_fn = nn.Sequential(*latent_fn_modules)

    def log_prior(self):
        "log p(params)"
        return sum(p.log_prob() for _, p in prior.named_priors(self))

    @abc.abstractmethod
    def likelihood_dist(self, f: torch.Tensor):
        "representation of p(y | f, params)"
        pass

    def forward(self, x: torch.Tensor):
        "representation of p(y | x, params)"
        f = self.latent_fn(x)
        return self.likelihood_dist(f)

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor):
        "log p(y | x, self.parameters)"
        log_prob_batch = self(x).log_prob(y)
        return log_prob_batch.sum(0) * (self.num_data/log_prob_batch.size(0))

    def potential(self, x, y):
        "-log p(y, params | x)"
        return -self.log_likelihood(x, y) - self.log_prior()

    def get_potential(self, x: torch.Tensor, y: torch.Tensor):
        "returns (potential(param_dict) -> torch.Tensor)"
        def potential_fn(param_dict):
            "-log p(y, params | x)"
            with self.using_params(param_dict):
                return self.potential(x, y)
        return potential_fn

    def params_with_prior_dict(self):
        return OrderedDict(
            (k, v.data) for (k, v) in prior.named_params_with_prior(self))

    def sample_all_priors(self):
        for _, v in prior.named_priors(self):
            v.sample()

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

