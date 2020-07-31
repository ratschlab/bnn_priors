import unittest
import numpy as np
import torch
import scipy.stats

from bnn_priors import prior
from bnn_priors.models import AbstractModel
from bnn_priors.mcmc import SGLD


class GaussianModel(AbstractModel):
    likelihood_dist = NotImplemented
    def log_likelihood(self):
        return 0.
    def __init__(self, N, D, mean=0., std=1.):
        super().__init__(None)
        for i in range(N):
            setattr(self, str(i), prior.Normal(torch.Size([D]), mean, std))

    def potential_avg(self):
        return -self.log_prior()

    def potential_avg_closure(self):
        self.zero_grad()
        loss = self.potential_avg()
        loss.backward()
        return loss

class SGLDTest(unittest.TestCase):
    def test_distribution_preservation(self, n_vars=50, n_dim=1000, n_samples=200):
        """Tests whether VerletSGLD preserves the distribution of a  Gaussian potential correctly.
        """
        torch.manual_seed(124)
        mean, std = 1, 2.
        temperature = 3/4
        model = GaussianModel(N=n_vars, D=n_dim, mean=mean, std=std)
        sgld = SGLD(prior.params_with_prior(model), lr=1/512, num_data=1,
                    momentum=0.9, temperature=temperature)
        model.sample_all_priors()
        with torch.no_grad():
            for p in prior.params_with_prior(model):
                p.mul_(temperature**.5)

        # Set the preconditioner randomly
        for _, state in sgld.state.items():
            state['preconditioner'] = torch.rand(()).item() + 0.2

        sgld.sample_momentum()

        for step in range(n_samples):
            sgld.step(model.potential_avg_closure)

        parameters = np.empty(n_vars*n_dim)
        kinetic_temp = np.empty(n_vars)
        config_temp = np.empty(n_vars)
        for i, (p, state) in enumerate(sgld.state.items()):
            parameters[i*n_dim:(i+1)*n_dim] = p.detach().numpy()
            kinetic_temp[i] = state['est_temperature']
            config_temp[i] = state['est_config_temp']

        # Test whether the parameters are 1-d Gaussian distributed
        statistic, (critical_value, *_), (significance_level, *_
                                          ) = scipy.stats.anderson(parameters, dist='norm')

        assert significance_level == 15, "next line does not check for significance of 15%"
        assert statistic < critical_value, "the samples are not Normal with p<0.15"

        def norm(x): return scipy.stats.norm.cdf(x, scale=temperature**.5)
        _, pvalue = scipy.stats.ks_1samp(parameters, norm, mode='exact')
        assert pvalue >= 0.3, "the samples are not Normal with the correct variance with p<0.3"

        def chi2(x): return scipy.stats.chi2.cdf(x, df=n_dim, loc=0., scale=temperature/n_dim)
        _, pvalue = scipy.stats.ks_1samp(config_temp, chi2, mode='exact')
        assert pvalue >= 0.3, "the configurational temperature is not Chi^2 with p<0.3"
        # _, pvalue = scipy.stats.ks_1samp(kinetic_temp, chi2, mode='exact')
        # assert pvalue >= 0.3, "the kinetic temperature is not Chi^2 with p<0.3"


if __name__ == '__main__':
    unittest.main()
