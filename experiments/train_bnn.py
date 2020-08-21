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

from bnn_priors.data import UCI, CIFAR10
from bnn_priors.models import RaoBDenseNet, DenseNet, PreActResNet18, PreActResNet34
from bnn_priors.prior import LogNormal
from bnn_priors import prior
from bnn_priors.inference import SGLDRunner
from bnn_priors import exp_utils
from bnn_priors.exp_utils import get_prior

# Makes CUDA faster
if t.cuda.is_available():
    t.backends.cudnn.benchmark = True
    
TMPDIR = "/tmp"

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
    batch_size = None
    batchnorm = True
    save_samples = False
    device = "try_cuda"


@ex.capture
def device(device):
    return exp_utils.device(device)


@ex.capture
def get_data(data):
    return exp_utils.get_data(data, device())


@ex.capture
def get_model(x_train, y_train, model, width, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm):
    return exp_utils.get_model(x_train, y_train, model, width, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm)


@ex.capture
def evaluate_model(model, dataloader_test, samples, bn_params, data, n_samples):
    return exp_utils.evaluate_model(model, dataloader_test, samples, bn_params, n_samples,
                   eval_data=data, likelihood_eval=True, accuracy_eval=True, calibration_eval=False)


@ex.automain
def main(inference, model, width, n_samples, warmup,
        burnin, skip, cycles, temperature, momentum,
        precond_update, lr, batch_size, save_samples):
    assert inference in ["SGLD", "HMC"]
    assert width > 0
    assert n_samples > 0
    assert cycles > 0
    assert temperature > 0

    data = get_data()

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
        if batch_size is None:
            batch_size = len(data.norm.train)
        dataloader = t.utils.data.DataLoader(data.norm.train, batch_size=batch_size, shuffle=True, drop_last=True)
        mcmc = SGLDRunner(model=model, dataloader=dataloader, epochs_per_cycle=epochs_per_cycle,
                  warmup_epochs=warmup, sample_epochs=sample_epochs, learning_rate=lr,
                  skip=skip, sampling_decay=True, cycles=cycles, temperature=temperature,
                  momentum=momentum, precond_update=precond_update)

    mcmc.run(progressbar=True)
    samples = mcmc.get_samples()

    bn_params = {k:v for k,v in model.state_dict().items() if "bn" in k}
    
    model.eval()
    
    batch_size = min(batch_size, len(data.norm.test))
    dataloader_test = t.utils.data.DataLoader(data.norm.test, batch_size=batch_size)

    return evaluate_model(model, dataloader_test, samples, bn_params)
