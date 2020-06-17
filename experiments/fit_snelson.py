import sacred
import numpy as np
import torch
from pathlib import Path
import os
from bnn_priors.models import DenseNet
import matplotlib.pyplot as plt


from pyro.infer.mcmc import NUTS, HMC
from pyro.infer.mcmc.api import MCMC

experiment = sacred.Experiment("fit_snelson")

@experiment.config
def config():
    warmup_steps = 50
    num_samples = 100

@experiment.automain
def main(warmup_steps, num_samples):
    data = np.load(Path(os.path.dirname(__file__))/"../data/snelson.npz")

    model = DenseNet(1, 1, 32)
    if torch.cuda.is_available():
        model = model.cuda()

    x_train = torch.from_numpy(data['x_train']).unsqueeze(1).to(model.lin1.weight)
    y_train = torch.from_numpy(data['y_train']).unsqueeze(1).to(x_train)

    x_test = torch.from_numpy(data['x_test']).unsqueeze(1).to(x_train)

    with torch.no_grad():
        model.sample_all_priors()
        y = model(x_test).loc.cpu()
        plt.plot(x_test.cpu(), y)

        model.sample_all_priors()
        y = model(x_test).loc.cpu()
        plt.plot(x_test.cpu(), y)

        model.sample_all_priors()
        y = model(x_test).loc.cpu()
        plt.plot(x_test.cpu(), y)

        plt.show()

    kernel = NUTS(potential_fn=model.get_potential(x_train, y_train),
                  adapt_step_size=True, step_size=0.1)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps,
                initial_params=model.params_with_prior_dict())
    mcmc.run()

    samples = mcmc.get_samples()

    for i in range(-10, 0):
        sample = dict((k, v[i]) for k, v in samples.items())
        with torch.no_grad(), model.using_params(sample):
            plt.plot(x_test.cpu(), model(x_test).loc.cpu(), color="C2", alpha=0.7)
    plt.scatter(x_train.cpu(), y_train.cpu())
    plt.show()
    # plt.savefig("/tmp/fig.png")


