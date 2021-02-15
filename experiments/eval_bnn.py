"""
Evaluation script for the BNN experiments with different data sets and priors.
"""

import os
import json
import h5py

import numpy as np
import torch
from pathlib import Path
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from bnn_priors import exp_utils

# Makes CUDA faster
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

TMPDIR = "/tmp"

ex = Experiment("bnn_evaluation")
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    # path of the config.json file of the training run
    config_file = None
    # evaluation dataset to use, if it differs from the training data
    eval_data = None
    # whether the likelihood should be evaluated
    likelihood_eval = True
    # whether the accuracy should be evaluated
    accuracy_eval = True
    # whether the uncertainty calibration should be evaluated
    calibration_eval = False
    # whether the OOD detection should be evaluated
    ood_eval = False
    # number of first samples from the Markov chain to discard
    skip_first = 0

    # Whether the run to be evaluated is an SGD run (and thus lacks the Sacred
    # run.json file, etc)
    is_run_sgd = False

    # Batch size during evaluation. Reduce if you run out of GPU memory.
    batch_size = 128
    # Device to use: cuda, cpu, or try_cuda
    device = 'try_cuda'

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
    if eval_data is not None:
        return exp_utils.get_data(eval_data, "cpu")
    else:
        return exp_utils.get_data(data, "cpu")


@ex.capture
def get_train_data(data):
    return exp_utils.get_data(data, device())


@ex.automain
def main(config_file, batch_size, run_dir, eval_data, data, skip_first, model,
         width, eval_samples, calibration_eval, ood_eval, marglik_eval,
         is_run_sgd):
    run_dir = Path(run_dir)
    with open(run_dir/"run.json") as infile:
        run_data = json.load(infile)

    samples = exp_utils.load_samples(run_dir/"samples.pt",
                                     idx=np.s_[skip_first:])
    with h5py.File(run_dir/"metrics.h5", "r") as metrics_file:
        exp_utils.reject_samples_(samples, metrics_file)
    del samples["steps"]
    del samples["timestamps"]
    for s in samples.items ():
        assert len(s)>0, f"we have less than {skip_first} samples"

    if eval_data is None:
        eval_data = data

    data = get_eval_data()

    x_train = data.norm.train_X
    y_train = data.norm.train_y
    if is_run_sgd:
        model = get_model(x_train=x_train, y_train=y_train, model=model,
                          width=width, depth=3, weight_prior="improper",
                          weight_loc=0., weight_scale=1., bias_prior="improper",
                          bias_loc=0., bias_scale=1., batchnorm=True,
                          weight_prior_params={}, bias_prior_params={})
    else:
        model = get_model(x_train=x_train, y_train=y_train)
    model = model.to(device())
    model.eval()

    if batch_size is None:
        batch_size = len(data.norm.test)
    else:
        batch_size = min(batch_size, len(data.norm.test))
    dataloader_test = torch.utils.data.DataLoader(data.norm.test, batch_size=batch_size)

    if calibration_eval and not (eval_data[:7] == "cifar10" or eval_data[-5:] == "mnist"):
        raise NotImplementedError("The calibration is not defined for this type of data.")

    if ood_eval and not (eval_data[:7] == "cifar10" or eval_data[-5:] == "mnist" or eval_data == "svhn"):
        raise NotImplementedError("The OOD error is not defined for this type of data.")

    results = evaluate_model(model=model, dataloader_test=dataloader_test,
                             samples=samples)

    if ood_eval:
        train_data = get_train_data()
        dataloader_train = torch.utils.data.DataLoader(train_data.norm.test, batch_size=batch_size)
        ood_results = evaluate_ood(model=model,
                                   dataloader_train=dataloader_train,
                                   dataloader_test=dataloader_test,
                                   samples=samples)
        results = {**results, **ood_results}


    return results
