"""
Evaluation script for the BNN experiments with different data sets and priors.
"""

import os
import math
import json
import h5py

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
    ex.add_config(config_file)  # Adds config entries from the previous script
    run_dir = os.path.dirname(config_file)
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(eval_dir))

device = ex.capture(exp_utils.device)
get_model = ex.capture(exp_utils.get_model)
evaluate_model = ex.capture(exp_utils.evaluate_model)
evaluate_ood = ex.capture(exp_utils.evaluate_ood)
evaluate_marglik = ex.capture(exp_utils.evaluate_marglik)

@ex.capture
def get_eval_data(data, eval_data):
    # TODO load synthetic data if present
    if eval_data is not None:
        return exp_utils.get_data(eval_data, device())
    else:
        return exp_utils.get_data(data, device())
    
    
@ex.capture
def get_train_data(data):
    return exp_utils.get_data(data, device())


@ex.automain
def main(config_file, batch_size, n_samples, run_dir, eval_data, data, skip_first, eval_samples,
        likelihood_eval, accuracy_eval, calibration_eval, ood_eval, marglik_eval):
    assert skip_first < n_samples, "We don't have that many samples to skip"
    run_dir = Path(run_dir)
    with open(run_dir/"run.json") as infile:
        run_data = json.load(infile)

    assert "samples.pt" in run_data["artifacts"], "No samples found"

    samples = exp_utils.load_samples(run_dir/"samples.pt",
                                     idx=np.s_[skip_first:])
    with h5py.File(run_dir/"metrics.h5", "r") as metrics_file:
        exp_utils.reject_samples_(samples, metrics_file)
    del samples["steps"]
    del samples["timestamps"]

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
        
    if ood_eval and not (eval_data[:7] == "cifar10" or eval_data[-5:] == "mnist" or eval_data == "svhn"):
        raise NotImplementedError("The OOD error is not defined for this type of data.")

    results = evaluate_model(model=model, dataloader_test=dataloader_test,
                             samples=samples)
    
    if ood_eval:
        train_data = get_train_data()
        dataloader_train = t.utils.data.DataLoader(train_data.norm.test, batch_size=batch_size)
        ood_results = evaluate_ood(model=model,
                                   dataloader_train=dataloader_train,
                                   dataloader_test=dataloader_test,
                                   samples=samples)
        results = {**results, **ood_results}
        
    if marglik_eval:
        if eval_samples is None:
            eval_samples = samples
        else:
            eval_samples = exp_utils.load_samples(eval_samples, idx=np.s_[skip_first:])
        marglik_results = evaluate_marglik(model=model, train_samples=samples,
                                           eval_samples=eval_samples)
        results = {**results, **marglik_results}
        
    return results
