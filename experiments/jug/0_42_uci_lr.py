#!/usr/bin/env python3

import jug
import subprocess
from pathlib import Path
import sys
import math
import os
import time
import random
import socket

experiments_dir = Path(__file__).resolve().parent.parent

@jug.TaskGenerator
def train_bnn(**config):
    script = experiments_dir / "train_bnn.py"
    args = [sys.executable, script, "with",
            *[f"{k}={v}" for k, v in config.items()]]
    print(f"Running " + " ".join(map(repr, args)))
    complete = subprocess.run(args)
    if complete.returncode != 0:
        err_msg = f"Process returned with code {complete.returncode} in host {socket.gethostname()}"
        with open(Path(config["log_dir"])/"error.txt", "w") as f:
            f.write(err_msg)
        raise SystemError(err_msg)
    return complete

@jug.TaskGenerator
def train_sgd(log_dir, **config):
    for i in range(10000):
        path = log_dir/str(i)
        try:
            os.mkdir(path)
            log_dir = path
            break
        except FileExistsError:
            time.sleep(0.1 + random.random()*0.1)

    script = experiments_dir / "train_sgd.py"
    args = [sys.executable,  "-m", "pdb", script,
            *[f"--{k}={v}" for k, v in config.items()]]
    print(f"Running in cwd={log_dir} " + " ".join(map(repr, args)))
    complete = subprocess.run(args, cwd=log_dir)
    if complete.returncode != 0:
        err_msg = f"Process returned with code {complete.returncode} in host {socket.gethostname()}"
        with open(Path(log_dir)/"error.txt", "w") as f:
            f.write(err_msg)
        raise SystemError(err_msg)
    return log_dir

name = Path(__file__).name[:-3]
log_dir = experiments_dir.parent/"logs"/name
jug.set_jugdir(str(log_dir/"jugdir"))

prior = "gaussian"
temps=( 1.0, )
datasets=("UCI_protein", "UCI_boston")
model = "densenet"
learning_rates=(0.01, 0.001, 0.0001, 1e-5)
batch_sizes=(125, None)
sampling_decay="flat"
temp=1.0

# momentum = e^(-gamma * h)
# lr = h^2 * num_data = h^2 * 50000
# log momentum = -gamma*h
# gamma = -log(momentum) / sqrt(lr/50000)
#
# Baseline settings:
# lr = 0.1
# momentum = 0.98
# -> gamma = 14.28
gamma = 14.2854713425451
momenta = [math.exp(-gamma * math.sqrt(lr / 50000)) for lr in learning_rates]

# Also ran the same experiments but with 60 cycles of length 50 each
for dataset, batch_size in zip(datasets, batch_sizes):
    if batch_size is None:
        n_epochs = 10000
    else:
        n_epochs = 1000
    for i in reversed(range(len(learning_rates))):
        n_samples = 100
        skip_first = n_samples // 3
        config = dict(weight_prior=prior, data=dataset,
                      model=model, depth=3, width=64, warmup=(n_epochs // n_samples - 1),
                      burnin=0,
                      skip=1, n_samples=n_samples, cycles=n_samples, temperature=temp,
                      sampling_decay=sampling_decay, lr=learning_rates[i],
                      init_method="he", load_samples=None, skip_first=skip_first,
                      batch_size=batch_size, save_samples=True,
                      log_dir=str(log_dir), batchnorm=True)

        train_bnn(**config, inference="VerletSGLDReject", momentum=momenta[i])
        train_bnn(**config, inference="SGLDReject", momentum=momenta[i])
        if temp == 1.0:
            train_bnn(**config, inference="HMCReject", momentum=1.0)


        config = dict(model=model, data=dataset, width=64, # depth=3,
                      batch_size=batch_size, sampling_decay=sampling_decay,
                      n_epochs=n_epochs, epochs_per_sample=n_epochs // n_samples,
                      skip_first=skip_first)

        train_sgd(log_dir, **config, momentum=momenta[i], lr=learning_rates[i])
