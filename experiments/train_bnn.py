"""
Training script for the BNN experiments with different data sets and priors.
"""

import os
import math
import uuid
import json
import contextlib

import numpy as np
import torch as t
from pathlib import Path
from pyro.infer.mcmc import NUTS, HMC
from pyro.infer.mcmc.api import MCMC
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from bnn_priors.data import UCI, CIFAR10, Synthetic
from bnn_priors.models import RaoBDenseNet, DenseNet, PreActResNet18, PreActResNet34
from bnn_priors.prior import LogNormal
from bnn_priors import prior
import bnn_priors.inference
from bnn_priors import exp_utils
from bnn_priors.exp_utils import get_prior

# Makes CUDA faster
if t.cuda.is_available():
    t.backends.cudnn.benchmark = True

TMPDIR = "/tmp"

ex = Experiment("bnn_training")
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    data = "UCI_boston"
    inference = "SGLD"
    model = "densenet"
    width = 50
    depth = 3
    weight_prior = "gaussian"
    bias_prior = "gaussian"
    weight_loc = 0.
    weight_scale = 2.**0.5
    bias_loc = 0.
    bias_scale = 1.
    weight_prior_params = {}
    bias_prior_params = {}
    if not isinstance(weight_prior_params, dict):
        print(weight_prior_params)
        weight_prior_params = json.loads(weight_prior_params)
    if not isinstance(bias_prior_params, dict):
        bias_prior_params = json.loads(bias_prior_params)
    n_samples = 1000
    warmup = 2000
    burnin = 2000
    skip = 5
    metrics_skip = 1
    cycles =  5
    temperature = 1.0
    momentum = 0.9
    precond_update = 1
    lr = 5e-4
    he_init = True
    batch_size = None
    batchnorm = True
    device = "try_cuda"
    save_samples = True
    run_id = uuid.uuid4().hex
    log_dir = str(Path(__file__).resolve().parent.parent/"logs")
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        ex.observers.append(FileStorageObserver(log_dir))
    

@ex.capture
def device(device):
    return exp_utils.device(device)


@ex.capture
def get_data(data, batch_size):
    if data[:9] == "synthetic":
        _, data, prior = data.split(".")
        dataset = get_data(data)
        x_train = dataset.norm.train_X
        y_train = dataset.norm.train_y
        net = get_model(x_train=x_train, y_train=y_train, weight_prior=prior, weight_prior_params={})
        net.sample_all_priors()
        return Synthetic(dataset=dataset, model=net, batch_size=batch_size, device=device())
    else:
        return exp_utils.get_data(data, device())


@ex.capture
def get_model(x_train, y_train, model, width, depth, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm,
             weight_prior_params, bias_prior_params):
    return exp_utils.get_model(x_train, y_train, model, width, depth, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm, weight_prior_params,
                               bias_prior_params)


@ex.capture
def evaluate_model(model, dataloader_test, samples, data, n_samples):
    return exp_utils.evaluate_model(model, dataloader_test, samples, n_samples,
                   eval_data=data, likelihood_eval=True, accuracy_eval=True, calibration_eval=False)


@ex.automain
def main(inference, model, width, n_samples, warmup, he_init,
         burnin, skip, metrics_skip, cycles, temperature, momentum,
         precond_update, lr, batch_size, save_samples, run_id, log_dir, _run):
    assert inference in ["SGLD", "HMC", "VerletSGLD", "OurHMC"]
    assert width > 0
    assert n_samples > 0
    assert cycles > 0
    assert temperature >= 0

    data = get_data()

    x_train = data.norm.train_X
    y_train = data.norm.train_y

    x_test = data.norm.test_X
    y_test = data.norm.test_y

    model = get_model(x_train=x_train, y_train=y_train)
    
    if he_init:
        exp_utils.he_initialize(model)

    this_run_dir = Path(_run.observers[0].dir)
    if save_samples:
        model_saver_fn = (lambda: exp_utils.HDF5ModelSaver(this_run_dir/"samples.pt", "w"))
    else:
        @contextlib.contextmanager
        def model_saver_fn():
            yield None

    with exp_utils.HDF5Metrics(this_run_dir/"metrics.h5", "w") as metrics_saver,\
         model_saver_fn() as model_saver:
        if inference == "HMC":
            kernel = HMC(potential_fn=lambda p: model.get_potential(x_train, y_train, eff_num_data=1*x_train.shape[0])(p),
                adapt_step_size=False, adapt_mass_matrix=False,
                step_size=1e-3, num_steps=32)
            mcmc = MCMC(kernel, num_samples=n_samples, warmup_steps=warmup, initial_params=model.params_dict())
        else:
            if inference == "SGLD":
                runner_class = bnn_priors.inference.SGLDRunner
            elif inference == "VerletSGLD":
                runner_class = bnn_priors.inference.VerletSGLDRunner
            elif inference == "OurHMC":
                runner_class = bnn_priors.inference.HMCRunner

            assert (n_samples * skip) % cycles == 0
            sample_epochs = n_samples * skip // cycles
            epochs_per_cycle = warmup + burnin + sample_epochs
            if batch_size is None:
                batch_size = len(data.norm.train)
            dataloader = t.utils.data.DataLoader(data.norm.train, batch_size=batch_size, shuffle=True, drop_last=True)
            mcmc = runner_class(model=model, dataloader=dataloader, epochs_per_cycle=epochs_per_cycle,
                                warmup_epochs=warmup, sample_epochs=sample_epochs, learning_rate=lr,
                                skip=skip, metrics_skip=metrics_skip, sampling_decay=True, cycles=cycles, temperature=temperature,
                                momentum=momentum, precond_update=precond_update,
                                metrics_saver=metrics_saver, model_saver=model_saver)

        mcmc.run(progressbar=True)
    samples = mcmc.get_samples()
    model.eval()

    batch_size = min(batch_size, len(data.norm.test))
    dataloader_test = t.utils.data.DataLoader(data.norm.test, batch_size=batch_size)

    return evaluate_model(model, dataloader_test, samples)
