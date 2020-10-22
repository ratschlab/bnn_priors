import unittest
import numpy as np
import torch
import scipy.stats
import math

from gpytorch.distributions import MultivariateNormal
from bnn_priors.mcmc import HMC
from bnn_priors.models import DenseNet, GaussianModel
from bnn_priors import prior

from .test_verlet_sgld import store_verlet_state, zip_allclose, new_model_loss
from .utils import requires_float64


class HMCTest(unittest.TestCase):
    @requires_float64
    def test_reversible(self, N=10):
        model, loss = new_model_loss(N=N)

        sgld = HMC(model.parameters(), lr=0.01, num_data=N,
                   raise_on_nan=True, raise_on_no_grad=True)

        # Set the preconditioner randomly
        for _, state in sgld.state.items():
            state['preconditioner'] = torch.rand(()).item() + 0.2

        sgld.sample_momentum()
        p0, m0 = store_verlet_state(sgld)

        sgld.initial_step(loss)
        p1, m_half = store_verlet_state(sgld)

        sgld.step(loss)
        p2, m_3halves = store_verlet_state(sgld)

        sgld.final_step(loss)
        p2_alt, m2 = store_verlet_state(sgld)

        assert not any(zip_allclose(p0, p1))
        assert not any(zip_allclose(p1, p2))
        assert all(zip_allclose(p2, p2_alt))

        assert not any(zip_allclose(m0, m_half))
        assert not any(zip_allclose(m_half, m_3halves))
        assert not any(zip_allclose(m_3halves, m2))

        # Negate momenta and go back
        for _, state in sgld.state.items():
            state['momentum_buffer'].neg_()

        sgld.initial_step(loss)
        p1_alt, m_3halves_neg = store_verlet_state(sgld)
        assert all(zip_allclose(p1, p1_alt))
        assert all(zip_allclose(m_3halves, map(torch.neg, m_3halves_neg)))

        sgld.step(loss)
        p0_alt, m_half_neg = store_verlet_state(sgld)
        assert all(zip_allclose(p0, p0_alt))
        assert all(zip_allclose(m_half, map(torch.neg, m_half_neg)))

        sgld.final_step(loss)
        p0_alt2, m0_neg = store_verlet_state(sgld)
        assert all(zip_allclose(p0, p0_alt2))
        assert all(zip_allclose(m0, map(torch.neg, m0_neg)))


    def test_distribution_preservation(self, n_vars=50, n_dim=1000, n_samples=100, momentum_resample=4):
        """Tests whether HMC preserves the distribution of a  Gaussian potential correctly.
        """
        torch.manual_seed(122)
        mean, std = 1., 2.,
        model = GaussianModel(N=n_vars, D=n_dim, mean=mean, std=std)
        sgld = HMC(model.parameters(), lr=1/32, num_data=1)
        model.sample_all_priors()

        # Set the preconditioner randomly
        for _, state in sgld.state.items():
            state['preconditioner'] = (torch.rand(()).item() + 0.2) / math.sqrt(std)

        sum_acceptance = 0.
        n_acceptance = 0
        assert n_samples % momentum_resample == 0
        for step in range(n_samples+1):
            if step % momentum_resample == 0:
                if step != 0:
                    loss = sgld.final_step(model.potential_avg_closure).item()
                    delta_energy = sgld.delta_energy(prev_loss, loss)
                    rejected, _ = sgld.maybe_reject(delta_energy)
                    if rejected:
                        with torch.no_grad():
                            assert np.allclose(prev_loss, model.potential_avg(None, None, 1.).item())
                    #     print(f"Rejected sample, with P(accept)={math.exp(-delta_energy)}")
                    # else:
                    #     print(f"Accepted sample, with P(accept)={math.exp(-delta_energy)}")
                    n_acceptance += 1
                    sum_acceptance += min(1., math.exp(-delta_energy))

                    if step == n_samples:
                        break

                sgld.sample_momentum()
                prev_loss = sgld.initial_step(model.potential_avg_closure, save_state=True).item()
            else:
                sgld.step(model.potential_avg_closure)

        assert sum_acceptance/n_acceptance > 0.6  # Was 0.65 at commit 56988f7

        parameters = np.empty(n_vars*n_dim)
        kinetic_temp = np.empty(n_vars)
        config_temp = np.empty(n_vars)
        for i, (p, state) in enumerate(sgld.state.items()):
            parameters[i*n_dim:(i+1)*n_dim] = p.detach().numpy()
            kinetic_temp[i] = state['est_temperature']
            config_temp[i] = state['est_config_temp']
        assert i == len(config_temp)-1

        # Test whether the parameters are 1-d Gaussian distributed
        statistic, (critical_value, *_), (significance_level, *_
                                          ) = scipy.stats.anderson(parameters, dist='norm')

        assert significance_level == 15, "next line does not check for significance of 15%"
        assert statistic < critical_value, "the samples are not Normal with p<0.15"

        def norm(x): return scipy.stats.norm.cdf(x, loc=mean, scale=std)
        _, pvalue = scipy.stats.ks_1samp(parameters, norm, mode='exact')
        assert pvalue >= 0.3, "the samples are not Normal with the correct variance with p<0.3"

        def chi2(x): return scipy.stats.chi2.cdf(x, df=n_dim, loc=0., scale=1/n_dim)
        _, pvalue = scipy.stats.ks_1samp(config_temp, chi2, mode='exact')
        assert pvalue >= 0.3, "the configurational temperature is not Chi^2 with p<0.3"

        def chi2(x): return scipy.stats.chi2.cdf(x, df=n_dim, loc=0., scale=1/n_dim)
        _, pvalue = scipy.stats.ks_1samp(kinetic_temp, chi2, mode='exact')
        assert pvalue >= 0.3, "the kinetic temperature is not Chi^2 with p<0.3"
