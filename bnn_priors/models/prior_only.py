from .base import AbstractModel
from .. import prior
from typing import List

import torch

__all__ = ('GaussianModel', 'NealFunnel', 'NealFunnelT')


class PriorOnlyModel(AbstractModel):
    def __init__(self, priors: List[prior.Prior]):
        super().__init__(torch.nn.Identity())
        for i, p in enumerate(priors):
            setattr(self, str(i), p)

    def likelihood_dist(self, f):
        return torch.distributions.Normal(loc=f, scale=1.)

    def log_likelihood(self, x, y, eff_num_data):
        return torch.zeros((), requires_grad=True)

    def log_likelihood_avg(self, x, y):
        return torch.zeros((), requires_grad=True)

    def split_potential_and_acc(self, x, y, eff_num_data):
        loss = torch.zeros(())
        log_prior = self.log_prior()
        potential_avg = -log_prior
        return loss, log_prior, potential_avg, loss, self.likelihood_dist(y)

    def potential_avg_closure(self):
        self.zero_grad()
        loss = self.potential_avg(None, None, 1.)
        loss.backward()
        return loss


class GaussianModel(PriorOnlyModel):
    def __init__(self, N, D, mean=0., std=1.):
        super().__init__([
            prior.Normal(torch.Size([D]), mean, std) for _ in range(N)])


class NealFunnel(PriorOnlyModel):
    def __init__(self):
        std = torch.linspace(0.01, 1, 100)
        super().__init__([prior.Normal(torch.Size([]), 0., std)])


class NealFunnelT(PriorOnlyModel):
    def __init__(self):
        std = torch.linspace(0.01, 1, 100)
        super().__init__([prior.StudentT(torch.Size([]), 0., std, df=3)])
