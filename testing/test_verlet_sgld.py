import unittest
import numpy as np
import torch
import math
import scipy.stats
from typing import Tuple, List

from gpytorch.distributions import MultivariateNormal
from bnn_priors.mcmc import VerletSGLD
from bnn_priors.models import DenseNet, GaussianModel, NealFunnelT
from bnn_priors import prior

from .utils import requires_float64

def store_verlet_state(sgld) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    return zip(*((p.detach().clone(), state['momentum_buffer'].detach().clone())
                 for p, state in sgld.state.items()))

def store_verlet_grad_state(sgld) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    return zip(*((p.detach().clone(), p.grad.detach().clone(),
                  state['momentum_buffer'].detach().clone())
                 for p, state in sgld.state.items()))

def dot(a, b):
    "return (a*b).sum().item()"
    return (a.view(-1) @ b.view(-1)).item()

def zip_allclose(sequence_a, sequence_b):
    return (torch.allclose(a, b) for a, b in zip(sequence_a, sequence_b))

def new_model_loss(N=10, D=1):
    x = torch.randn(N, 1)
    y = x.sin()
    model = DenseNet(x.size(-1), y.size(-1), 10, noise_std=0.1)
    def loss():
        model.zero_grad()
        v = model.potential_avg(x, y, eff_num_data=1.)
        v.backward()
        return v
    return model, loss


class VerletSGLDTest(unittest.TestCase):
    """
    Q: Why don't we have this kind of test for the VerletSGLD?

    A: because, VerletSGLD is not, and should not be, reversible. If it were
    exactly reversible, the expression for the acceptance probability would be
    the same as for HMC. However, Langevin dynamics do not preserve the volume.
    For example, they shrink the momentum by sqrt(a) at every step, and compensate
    with injected noise.

    Thus the name of the class is a little improper, it's not really a Verlet
    integrator.
    """

    @requires_float64
    def test_distribution_preservation(self, n_vars=50, n_dim=1000, n_samples=200, mh_freq=4, do_rejection=True, seed=145, store_success=False):
        """Tests whether VerletSGLD preserves the distribution of a  Gaussian potential correctly.

        Q: Why is the learning rate different compared to the SGLD test?
        A: VerletSGLD should be able to handle a larger learning rate because
           it uses a more accurate integration scheme, and also corrects errors
           with the M-H step.
        """
        torch.manual_seed(seed)
        mean, std = 1., 2.
        temperature = 3/4
        model = GaussianModel(N=n_vars, D=n_dim, mean=mean, std=std)
        # `num_data=1` to prevent scaling the Gaussian potential
        sgld = VerletSGLD(model.parameters(), lr=1/32, num_data=1,
                          momentum=0.9, temperature=temperature)
        model.sample_all_priors()
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(mean).mul_(temperature**.5).add_(mean)

        success = {}
        def assert_or_store(truth, key):
            if store_success:
                success[key] = truth
            else:
                assert truth, key

        # Set the preconditioner randomly
        for _, state in sgld.state.items():
            state['preconditioner'] = (torch.rand(()).item() + 0.2) / math.sqrt(4)

        sgld.sample_momentum()

        sum_acceptance = 0.
        n_acceptance = 0
        assert n_samples % mh_freq == 0
        for step in range(n_samples+1):
            if step % mh_freq == 0:
                if step != 0:
                    loss = sgld.final_step(model.potential_avg_closure).item()
                    delta_energy = sgld.delta_energy(prev_loss, loss)
                    if do_rejection:
                        rejected, _ = sgld.maybe_reject(delta_energy)
                    else:
                        rejected = False
                    if rejected:
                        with torch.no_grad():
                            assert np.allclose(prev_loss, model.potential_avg().item())
                    #     print(f"Rejected sample, with P(accept)={math.exp(-delta_energy)}")
                    # else:
                    #     print(f"Accepted sample, with P(accept)={math.exp(-delta_energy)}")
                    n_acceptance += 1
                    sum_acceptance += min(1., math.exp(-delta_energy))

                    if step == n_samples:
                        break

                prev_loss = sgld.initial_step(model.potential_avg_closure, save_state=True).item()
            else:
                sgld.step(model.potential_avg_closure)

        assert sum_acceptance/n_acceptance > 0.6  # Was 0.73 at commit 56988f7

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
        assert_or_store(statistic < critical_value, "the samples are not Normal with p<0.15")

        def norm(x): return scipy.stats.norm.cdf(x, loc=mean, scale=std*temperature**.5)
        _, pvalue = scipy.stats.ks_1samp(parameters, norm, mode='exact')
        assert_or_store(pvalue >= 0.3, "the samples are not Normal with the correct variance with p<0.3")

        def chi2(x): return scipy.stats.chi2.cdf(x, df=n_dim, loc=0., scale=temperature/n_dim)
        _, pvalue = scipy.stats.ks_1samp(config_temp, chi2, mode='exact')
        assert_or_store(pvalue >= 0.3, "the configurational temperature is not Chi^2 with p<0.3")
        _, pvalue = scipy.stats.ks_1samp(kinetic_temp, chi2, mode='exact')
        assert_or_store(pvalue >= 0.3, "the kinetic temperature is not Chi^2 with p<0.3")
        return success

    def test_accept_prob(self, n_samples=10, seed=145):
        torch.manual_seed(seed)
        model = NealFunnelT()
        temperature = 3/4
        momentum = 127/128
        # `num_data=1` to prevent scaling the Gaussian potential
        sgld = VerletSGLD(model.parameters(), lr=1/32, num_data=1,
                          momentum=momentum, temperature=temperature)
        # Because `num_data=1`, we can say that time step squared is learning rate
        time_step_sq = sgld.param_groups[0]['lr']
        preconditioners = []

        model.sample_all_priors()
        # Set the preconditioner randomly
        for p in model.parameters():
            state = sgld.state[p]
            state['preconditioner'] = (torch.rand(()).item() + 0.2) / math.sqrt(4)
            preconditioners.append(state['preconditioner'])
        sgld.sample_momentum()

        # Run sampler and store intermediate states
        states = []
        U0 = model.potential_avg_closure().item()
        states.append(list(store_verlet_grad_state(sgld)))
        sgld.initial_step()
        for s in range(1, n_samples):
            model.potential_avg_closure()
            states.append(list(store_verlet_grad_state(sgld)))
            sgld.step()
            if s == n_samples-1:
                U1 = model.potential_avg_closure().item()
                sgld.final_step()
                states.append(list(store_verlet_grad_state(sgld)))

        # Calculate delta_energy clearly
        delta_energy_ref = 0.
        _, grads0, _ = states[0]
        _, grads1, _ = states[-1]
        for g0, g1, precond in zip(grads0, grads1, preconditioners):
            C = (time_step_sq * precond**2 / 8)
            delta_energy_ref += C*(dot(g1, g1) - dot(g0, g0))

        point_energies = 0.
        group = sgld.param_groups[0]
        for g0, g1, p in zip(grads0, grads1, group['params']):
            p.grad = g0
            point_energies -= sgld._point_energy(group, p, sgld.state[p])
            p.grad = g1
            point_energies += sgld._point_energy(group, p, sgld.state[p])

        assert np.allclose(delta_energy_ref, point_energies)

        for i in range(1, len(states)):
            params0, grads0, _ = states[i-1]
            params1, grads1, _ = states[i]
            for p0, p1, g0, g1 in zip(params0, params1, grads0, grads1):
                delta_energy_ref += -.5 * dot(p1-p0, g1+g0)

        delta_energy_ref += (U1 - U0)

        # Now compare with iterative
        delta_energy = sgld.delta_energy(U0, U1)

        assert np.allclose(delta_energy_ref, delta_energy), f"{delta_energy_ref} != {delta_energy}"


if __name__ == '__main__':
    """ There are 4 probabilistic assertions in the test in `verlet_sgld.py`.
    Assuming they are independent, the test should pass `(1-.15)*(1-.3)**3 =
    29.15%` of the time. Using this script that goes through random seeds 0-50,
    the test succeeds 34% of the time, 32% of the time if you omit the M-H
    correction. Probably good enough. """
    import tqdm
    from collections import defaultdict
    test = VerletSGLDTest()

    rand_succ = defaultdict(lambda: 0)
    rand_total = 0
    for seed in tqdm.trange(100):
        success = test.test_distribution_preservation(do_rejection=True, seed=seed, store_success=True, n_dim=2000, n_samples=500)
        for k, v in success.items():
            rand_succ[k] += (1 if v else 0)
        rand_total += 1

    for k, v in rand_succ.items():
        print(f"M-H test for not '{k}': {v/rand_total*100}% success (should be >~70% or >~85%)")
    print()


    norand_total = 0
    norand_succ = defaultdict(lambda: 0)
    for seed in tqdm.trange(100):
        success = test.test_distribution_preservation(do_rejection=False, seed=seed, store_success=True, n_dim=2000, n_samples=500)
        for k, v in success.items():
            norand_succ[k] += (1 if v else 0)
        norand_total += 1

    for k, v in norand_succ.items():
        print(f"No-reject test for not '{k}': {v/rand_total*100}% success (should be >~70% or >~85%)")
    print()
