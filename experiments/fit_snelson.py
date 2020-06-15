import sacred
import numpy as np
import torch
from pathlib import Path
import os
from bnn_priors.models import make_dense
import matplotlib.pyplot as plt


from pyro.infer.mcmc import NUTS, HMC
from pyro.infer.mcmc.api import MCMC

experiment = sacred.Experiment("fit_snelson")

@experiment.automain
def main():
    data = np.load(Path(os.path.dirname(__file__))/"../data/snelson.npz")

    model = make_dense(1, 1, 32)

    x_train = torch.from_numpy(data['x_train']).unsqueeze(1).to(model.lin1.weight)
    y_train = torch.from_numpy(data['y_train']).unsqueeze(1).to(x_train)

    x_test = torch.from_numpy(data['x_test']).unsqueeze(1).to(x_train)

    with torch.no_grad():
        # model.sample_all_priors()
        y = model.predict(x_test)
        plt.plot(x_test, y)

        # model.sample_all_priors()
        y = model.predict(x_test)
        plt.plot(x_test, y)

        # model.sample_all_priors()
        y = model.predict(x_test)
        plt.plot(x_test, y)

        plt.show()

    kernel = NUTS(model=model, adapt_step_size=True, step_size=0.1)
    mcmc = MCMC(kernel, num_samples=100, warmup_steps=50)
    mcmc.run(x=x_train, y=y_train)

    samples = mcmc.get_samples()

    for i in range(90, 100):
        for k in samples.keys():
            path = k.split('.')
            setattr(getattr(model, path[0]), path[1], samples[k][i])
        with torch.no_grad():
            plt.plot(x_test, model.predict(x_test), color="C2", alpha=0.7)
    plt.scatter(x_train, y_train)
    plt.show()


