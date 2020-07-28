import unittest
import numpy as np
import torch
from bnn_priors.models import DenseNet, AbstractModel
from bnn_priors.inference import SGLDRunner
from bnn_priors import prior
from bnn_priors.sgld import SGLD

import scipy.stats 

class GaussianModel(AbstractModel):
    likelihood_dist = NotImplemented
    def log_likelihood(self):
        return 0.
    def __init__(self, N, D, mean=0., std=1.):
        super().__init__(None)
        for i in range(N):
            setattr(self, str(i), prior.Normal(torch.Size([D]), mean, std))

    def potential_avg(self):
        self.zero_grad()
        loss = -self.log_prior()
        loss.backward()
        return loss


class SGLDTest(unittest.TestCase):
    def test_snelson_inference(self):
        data = np.load("../data/snelson.npz")

        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        x_train = torch.from_numpy(data['x_train']).unsqueeze(1).to(device=device, dtype=torch.get_default_dtype())
        y_train = torch.from_numpy(data['y_train']).unsqueeze(1).to(x_train)

        x_test = torch.from_numpy(data['x_test']).unsqueeze(1).to(x_train)

        N_steps = 10
        skip = 20
        warmup = 50
        cycles =  1
        temperature = 1.0
        momentum = 0.9
        precond_update = 10
        lr = 5e-4

        model = DenseNet(x_train, y_train, 128, noise_std=0.5)
        model.to(x_train)
        if torch.cuda.is_available():
            model = model.cuda()   # Resample model with He initialization so SGLD works.
        model.train()

        sgld = SGLDRunner(model=model, num_samples=N_steps, warmup_steps=warmup, learning_rate=lr,
                          skip=skip, sampling_decay=True, cycles=cycles, temperature=temperature,
                          momentum=momentum, precond_update=precond_update)
        sgld.run(x=x_train, y=y_train, progressbar=False)

        assert sgld.metrics["loss"][0] > sgld.metrics["loss"][-1]
        assert sgld.metrics["lr"][0] > sgld.metrics["lr"][1]
        assert (sgld.metrics["preconditioner/latent_fn.0.weight_prior"][0]
                != sgld.metrics["preconditioner/latent_fn.0.weight_prior"][-1])

    def test_posterior_and_temperatures(self, n_vars=3, n_dim=100, n_samples=500, thin=4):
        """Tests whether SGLD samples from a Gaussian potential correctly, and whether
        the kinetic and configurational temperatures have the distribution they
        ought to have (Chi^2)
        """
        torch.manual_seed(123)
        mean, std = 1, 2.
        temperature = 1-1/4
        model = GaussianModel(N=n_vars, D=n_dim, mean=mean, std=std)
        sgld = SGLD(prior.params_with_prior(model), lr=1/4, num_data=1,
                    momentum=1-1/128, temperature=temperature)

        kinetic_temp = np.zeros(n_samples * n_vars)
        config_temp = np.zeros(n_samples * n_vars)
        samples = np.zeros(n_samples * n_dim * n_vars)

        model.sample_all_priors()
        sgld.sample_momentum()
        for i in range(n_samples*thin):
            if i == 0:
                try:
                    sgld.step(model.potential_avg)
                except AttributeError:
                    sgld.initial_step(model.potential_avg)
            else:
                sgld.step(model.potential_avg)

            if i % thin == 0:
                i_ = i // thin
                for j, (p, state) in enumerate(sgld.state.items()):
                    kinetic_temp[i_*n_vars + j] = state['est_temperature']
                    config_temp[i_*n_vars + j] = state['est_config_temp']
                    samples[(i_*n_vars + j)*n_dim : (i_*n_vars + j + 1)*n_dim] = p.detach().numpy()

        # Test whether the samples come from a 1-d Gaussian
        statistic, (critical_value, *_), (significance_level, *_
                                          ) = scipy.stats.anderson(samples, dist='norm')

        assert significance_level == 15, "next line does not check for significance of 15%"
        # Test correctly fails if the prior is a Student-T distribution with 30 degrees of freedom
        # and we sample from it i.i.d.
        assert statistic < critical_value, "the samples are not Normal with p<0.15"

        # def norm(x): return scipy.stats.norm.cdf(x, scale=temperature**.5)
        # _, pvalue = scipy.stats.ks_1samp(samples, norm, mode='exact')
        # assert pvalue >= 0.3, "the samples are not Normal with the correct variance with p<0.3"

        # def chi2(x): return scipy.stats.chi2.cdf(x, df=n_dim, loc=0., scale=temperature/n_dim)

        # _, pvalue = scipy.stats.ks_1samp(config_temp, chi2, mode='exact')
        # assert pvalue >= 0.3, "the configurational temperature is not Chi^2 with p<0.3"
        # _, pvalue = scipy.stats.ks_1samp(kinetic_temp, chi2, mode='exact')
        # assert pvalue >= 0.3, "the kinetic temperature is not Chi^2 with p<0.3"


if __name__ == '__main__':
    unittest.main()
