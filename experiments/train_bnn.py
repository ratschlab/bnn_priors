"""
Training script for the BNN experiments with different data sets and priors.
"""

import os
import math

import numpy as np
import torch as t
from pathlib import Path
from pyro.infer.mcmc import NUTS, HMC
from pyro.infer.mcmc.api import MCMC
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from bnn_priors.data import UCI
from bnn_priors.models import RaoBDenseNet, DenseNet
from bnn_priors.prior import LogNormal
from bnn_priors import prior
from bnn_priors.inference import SGLDRunner

ex = Experiment("bnn_training")
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver('../logs'))

@ex.config
def config():
    data = "UCI_boston"
    inference = "SGLD"
    model = "densenet"
    width = 50
    weight_prior = "gaussian"
    bias_prior = "gaussian"
    weight_loc = 0.
    weight_scale = 2.**0.5
    bias_loc = 0.
    bias_scale = 1.
    n_samples = 1000
    warmup = 2000
    burnin = 2000
    skip = 5
    cycles =  5
    temperature = 1.0
    momentum = 0.9
    precond_update = None
    lr = 5e-4


@ex.capture
def get_data(data):
    assert data[:3] == "UCI" or data in []
    if data[:3] == "UCI":
        uci_dataset = data.split("_")[1]
        assert uci_dataset in ["boston", "concrete", "energy", "kin8nm",
                               "naval", "power", "protein", "wine", "yacht"]
        # TODO: do we ever use a different split than 0?
        dataset = UCI(uci_dataset, 0)
    return dataset


def get_prior(prior_name):
    priors = {"gaussian": prior.Normal,
             "lognormal": prior.LogNormal,
             "laplace": prior.Laplace,
             "cauchy": prior.Cauchy,
             "student-t": prior.StudentT,
             "uniform": prior.Uniform}
    assert prior_name in priors
    return priors[prior_name]


@ex.capture
def get_model(x_train, y_train, model, width, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale):
    assert model in ["densenet", "raobdensenet"]
    if weight_prior in ["cauchy"]:
        #TODO: which other distributions should use this? Laplace?
        scaling_fn = lambda std, dim: std/dim
    else:
        scaling_fn = lambda std, dim: std/dim**0.5
    weight_prior = get_prior(weight_prior)
    bias_prior = get_prior(bias_prior)
    if model == "densenet":
        net = DenseNet(x_train.size(-1), y_train.size(-1), width, noise_std=LogNormal((), -1., 0.2),
                        prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                        prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn).to(x_train)
    elif model == "raobdensenet":
        net = RaoBDenseNet(x_train, y_train, width, noise_std=LogNormal((), -1., 0.2)).to(x_train)
    return net


@ex.automain
def main(inference, model, width, n_samples, warmup,
        burnin, skip, cycles, temperature, momentum,
        precond_update, lr):
    assert inference in ["SGLD", "HMC"]
    assert width > 0
    assert n_samples > 0
    assert cycles > 0
    assert temperature > 0

    data = get_data()

    device = ('cuda' if t.cuda.is_available() else 'cpu')
    x_train = data.norm.train_X
    y_train = data.norm.train_y

    x_test = data.norm.test_X
    y_test = data.norm.test_y

    model = get_model(x_train=x_train, y_train=y_train)

    if inference == "HMC":
        kernel = HMC(potential_fn=lambda p: model.get_potential(x_train, y_train, eff_num_data=1*x_train.shape[0])(p),
             adapt_step_size=False, adapt_mass_matrix=False,
             step_size=1e-3, num_steps=32)
        mcmc = MCMC(kernel, num_samples=n_samples, warmup_steps=warmup, initial_params=model.params_with_prior_dict())
    elif inference == "SGLD":
        sample_epochs = n_samples * skip // cycles
        epochs_per_cycle = warmup + burnin + sample_epochs
        dataloader = t.utils.data.DataLoader(data.norm.train, batch_size = len(data.norm.train), shuffle=True)
        mcmc = SGLDRunner(model=model, dataloader=dataloader, epochs_per_cycle=epochs_per_cycle,
                  warmup_epochs=warmup, sample_epochs=sample_epochs, learning_rate=lr,
                  skip=skip, sampling_decay=True, cycles=cycles, temperature=temperature,
                  momentum=momentum, precond_update=precond_update)

    mcmc.run(progressbar=True)
    samples = mcmc.get_samples()

    # TODO: solve this more elegantly
    if inference == "SGLD":
        samples = {(key[:-2] if key[-2:] == ".p" else key) : val for key, val in samples.items()}
        del samples["lr"]

    lps = t.zeros(n_samples, *y_test.shape)

    for i in range(n_samples):
        sample = dict((k, v[i]) for k, v in samples.items())
        with t.no_grad(), model.using_params(sample):
            lps[i] = model(x_test).log_prob(y_test)

    final_params = dict((k, v[-1]) for k, v in samples.items())
    with t.no_grad(), model.using_params(sample):
        P = model(x_test)
        noise_std = model.noise_std()

    lps = lps.logsumexp(0) - math.log(n_samples)

    results = {"lp_mean": lps.mean().item(),
              "lp_std": lps.std().item(),
              "lp_stderr": float(lps.std().item()/np.sqrt(len(lps)))}
    return results
