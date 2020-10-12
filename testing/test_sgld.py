import unittest
import numpy as np
import torch
import math
import scipy.stats

from bnn_priors import prior
from bnn_priors.models import GaussianModel
from bnn_priors.mcmc import SGLD


class SGLDTest(unittest.TestCase):
    def test_distribution_preservation(self, n_vars=50, n_dim=1000, n_samples=200):
        """Tests whether SGLD preserves the distribution of a Gaussian potential correctly.
        """
        torch.manual_seed(123)
        mean, std = 1., 2.
        temperature = 3/4
        model = GaussianModel(N=n_vars, D=n_dim, mean=mean, std=std)
        sgld = SGLD(model.parameters(), lr=1/512, num_data=1,
                    momentum=0.9, temperature=temperature)
        model.sample_all_priors()
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(mean).mul_(temperature**.5).add_(mean)

        # Set the preconditioner randomly
        for _, state in sgld.state.items():
            state['preconditioner'] = (torch.rand(()).item() + 0.2) / math.sqrt(std)

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

        def norm(x): return scipy.stats.norm.cdf(x, loc=mean, scale=std*temperature**.5)
        _, pvalue = scipy.stats.ks_1samp(parameters, norm, mode='exact')
        assert pvalue >= 0.3, "the samples are not Normal with the correct variance with p<0.3"

        def chi2(x): return scipy.stats.chi2.cdf(x, df=n_dim, loc=0., scale=temperature/n_dim)
        _, pvalue = scipy.stats.ks_1samp(config_temp, chi2, mode='exact')
        assert pvalue >= 0.3, "the configurational temperature is not Chi^2 with p<0.3"
        # _, pvalue = scipy.stats.ks_1samp(kinetic_temp, chi2, mode='exact')
        # assert pvalue >= 0.3, "the kinetic temperature is not Chi^2 with p<0.3"

    def test_sgd_equivalence(self, n_vars=1, n_dim=5):
        model = GaussianModel(N=n_vars, D=n_dim, mean=0.5, std=0.25)
        lr = 1.25
        momentum = 0.9
        sgld = SGLD(model.parameters(), lr=lr, num_data=1, momentum=momentum, temperature=0.)
        sgld.sample_momentum()
        sgd = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        initial_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for _ in range(4):
            sgld.step(model.potential_avg_closure)
        sgld_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(initial_sd)
        for _ in range(4):
            sgd.step(model.potential_avg_closure)

        assert all(torch.allclose(v1, v2) for v1, v2 in zip(
            sgld_sd.values(),
            model.state_dict().values()))



if __name__ == '__main__':
    unittest.main()
