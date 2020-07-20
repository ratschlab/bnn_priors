import unittest
import numpy as np
import torch

from gpytorch.distributions import MultivariateNormal
from bnn_priors.verlet_sgld import HMC
from bnn_priors.models import DenseNet
from bnn_priors import prior

from .test_verlet_sgld import store_verlet_state, zip_allclose, new_model_loss
from .utils import requires_float64


class HMCTest(unittest.TestCase):
    @requires_float64
    def test_reversible(self, N=10):
        model, loss = new_model_loss(N=N)

        sgld = HMC(prior.params_with_prior(model), lr=0.01, num_data=N,
                   raise_on_nan=True, raise_on_no_grad=True)

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
