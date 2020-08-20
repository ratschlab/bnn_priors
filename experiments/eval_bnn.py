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
    likelihood_eval = True
    accuracy_eval = True
    
    assert config_file is not None, "No config_file provided"
    ex.add_config(config_file)
    log_dir = os.path.dirname(config_file)
    eval_dir = os.path.join(log_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(eval_dir))


@ex.capture
def device(device):
    return exp_utils.device(device)


@ex.capture
def get_data(data, eval_data):
    if eval_data is not None:
        return exp_utils.get_data(eval_data, device())
    else:
        return exp_utils.get_data(data, device())


@ex.capture
def get_model(x_train, y_train, model, width, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm):
    return exp_utils.get_model(x_train, y_train, model, width, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm)


@ex.automain
def main(config_file, batch_size, n_samples, log_dir, eval_data, data,
        likelihood_eval, accuracy_eval):
    runfile = os.path.join(log_dir, "run.json")
    with open(runfile) as infile:
        run_data = json.load(infile)

    assert "samples.pt" in run_data["artifacts"], "No samples found"
    assert "bn_params.pt" in run_data["artifacts"], "No bn_params found"

    samples = t.load(os.path.join(log_dir, "samples.pt"))
    bn_params = t.load(os.path.join(log_dir, "bn_params.pt"))
    
    if eval_data is None:
        eval_data = data
    
    data = get_data()

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

    # TODO: refactor this into a method in the exp_utils
    
    lps = []
    accs = []

    for i in range(n_samples):
        sample = dict((k, v[i].to(device())) for k, v in samples.items())
        sampled_state_dict = {**sample, **bn_params}
        with t.no_grad():
            # TODO: get model.using_params() to work with batchnorm params
            model.load_state_dict(sampled_state_dict)
            lps_sample = []
            accs_sample = []
            for batch_x, batch_y in dataloader_test:
                pred = model(batch_x)
                lps_batch = pred.log_prob(batch_y)
                if eval_data[:7] == "cifar10" or eval_data[-5:] == "mnist":
                    accs_batch = (t.argmax(pred.probs, dim=1) == batch_y).float()
                else:
                    accs_batch = (pred.mean - batch_y)**2.
                lps_sample.extend(list(lps_batch.cpu().numpy()))
                accs_sample.extend(list(accs_batch.cpu().numpy()))
            lps.append(lps_sample)
            accs.append(accs_sample)
            
    lps = t.tensor(lps)    
    lps = lps.logsumexp(0) - math.log(n_samples)
    accs = t.tensor(accs)
    accs = accs.mean(dim=1)
    
    results = {}
    if likelihood_eval:
        results["lp_mean"] = lps.mean().item()
        results["lp_std"] =  lps.std().item()
        results["lp_stderr"] =  float(lps.std().item()/np.sqrt(len(lps)))
    if accuracy_eval:
        results["acc_mean"] = accs.mean().item()
        results["acc_std"] =  accs.std().item()
        results["acc_stderr"] =  float(accs.std().item()/np.sqrt(len(accs)))
        
    return results
