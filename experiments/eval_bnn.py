"""
Evaluation script for the BNN experiments with different data sets and priors.
"""

import os
import math
import json

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

ex = Experiment("bnn_evaluation")
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    config_file = None
    eval_data = None
    eval_samples = None
    likelihood_eval = True
    accuracy_eval = True
    calibration_eval = False
    ood_eval = False
    marglik_eval = False
    skip_first = 0

    assert config_file is not None, "No config_file provided"
    ex.add_config(config_file)
    run_dir = os.path.dirname(config_file)
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(eval_dir))


@ex.capture
def device(device):
    return exp_utils.device(device)


@ex.capture
def get_eval_data(data, eval_data):
    if eval_data is not None:
        return exp_utils.get_data(eval_data, device())
    else:
        return exp_utils.get_data(data, device())
    
    
@ex.capture
def get_train_data(data):
    return exp_utils.get_data(data, device())


@ex.capture
def get_model(x_train, y_train, model, width, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm,
             weight_prior_params, bias_prior_params):
    return exp_utils.get_model(x_train, y_train, model, width, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm, weight_prior_params,
                               bias_prior_params)


@ex.capture
def evaluate_model(model, dataloader_test, samples, bn_params, eval_data, n_samples,
                   skip_first, likelihood_eval, accuracy_eval, calibration_eval):
    return exp_utils.evaluate_model(model, dataloader_test, samples, bn_params, n_samples-skip_first,
                   eval_data, likelihood_eval, accuracy_eval, calibration_eval)


@ex.capture
def evaluate_ood(model, dataloader_train, dataloader_test, samples, bn_params, n_samples, skip_first):
    return exp_utils.evaluate_ood(model, dataloader_train, dataloader_test,
                                  samples, bn_params, n_samples-skip_first)#


@ex.capture
def evaluate_marglik(model, samples, bn_params, n_samples, skip_first):
    return exp_utils.evaluate_marglik(model, samples, bn_params, n_samples-skip_first)


@ex.automain
def main(config_file, batch_size, n_samples, run_dir, eval_data, data, skip_first, eval_samples,
        likelihood_eval, accuracy_eval, calibration_eval, ood_eval, marglik_eval):
    assert skip_first < n_samples, "We don't have that many samples to skip"
    runfile = os.path.join(run_dir, "run.json")
    with open(runfile) as infile:
        run_data = json.load(infile)

    assert "samples.pt" in run_data["artifacts"], "No samples found"
    assert "bn_params.pt" in run_data["artifacts"], "No bn_params found"

    samples = t.load(os.path.join(run_dir, "samples.pt"))
    bn_params = t.load(os.path.join(run_dir, "bn_params.pt"))
    
    samples = {param: sample[skip_first:] for param, sample in samples.items()}
    
    if eval_data is None:
        eval_data = data

    data = get_eval_data()

    x_train = data.norm.train_X
    y_train = data.norm.train_y

    x_test = data.norm.test_X
    y_test = data.norm.test_y

    model = get_model(x_train=x_train, y_train=y_train)

    model.eval()

    if batch_size is None:
        batch_size = len(data.norm.test)
    else:
        batch_size = min(batch_size, len(data.norm.test))
    dataloader_test = t.utils.data.DataLoader(data.norm.test, batch_size=batch_size)

    if calibration_eval and not (eval_data[:7] == "cifar10" or eval_data[-5:] == "mnist"):
        raise NotImplementedError("The calibration is not defined for this type of data.")
        
    if ood_eval and not (eval_data[:7] == "cifar10" or eval_data[-5:] == "mnist"):
        raise NotImplementedError("The OOD error is not defined for this type of data.")

    results = evaluate_model(model, dataloader_test, samples, bn_params, eval_data)
    
    if ood_eval:
        train_data = get_train_data()
        dataloader_train = t.utils.data.DataLoader(train_data.norm.test, batch_size=batch_size)
        ood_results = evaluate_ood(model, dataloader_train, dataloader_test, samples, bn_params)
        results = {**results, **ood_results}
        
    if marglik_eval:
        if eval_samples is None:
            eval_samples = samples
        else:
            eval_samples = t.load(eval_samples)
        marglik_results = evaluate_marglik(model, eval_samples, bn_params)
        results = {**results, **marglik_results}
        
    return results
