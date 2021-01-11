#!/usr/bin/env python3

import jug
import subprocess
from pathlib import Path
import sys
import math
import os
import time
import random

experiments_dir = Path(__file__).resolve().parent.parent

@jug.TaskGenerator
def train_bnn(**config):
    script = experiments_dir / "train_bnn.py"
    args = [sys.executable, script, "with",
            *[f"{k}={v}" for k, v in config.items()]]
    print(f"Running " + " ".join(map(repr, args)))
    complete = subprocess.run(args)
    if complete.returncode != 0:
        raise SystemError(f"Process returned with code {complete.returncode}")
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
    args = [sys.executable, script,
            *[f"--{k}={v}" for k, v in config.items()]]
    print(f"Running in cwd={log_dir} " + " ".join(map(repr, args)))
    complete = subprocess.run(args, cwd=log_dir)
    if complete.returncode != 0:
        raise SystemError(f"Process returned with code {complete.returncode}")
    return log_dir

name = Path(__file__).name[:-3]
log_dir = experiments_dir.parent/"logs"/name
jug.set_jugdir(str(log_dir/"jugdir"))

prior = "gaussian"
temps=( 1.0, 0.1, 0.01 )
models=["googleresnet", "resnet18"]
learning_rates=[0.01]
batch_size=128
sampling_decay="cosine"

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


config = dict(data="cifar10_augmented", width=16, batch_size=batch_size,
              sampling_decay="flat", n_epochs=3000)
for model in models:
    for i in range(len(learning_rates)):
        train_sgd(log_dir, **config, model=model, momentum=momenta[i], lr=learning_rates[i])


config = dict(weight_prior=prior, data="cifar10_augmented", depth=20, width=16,
              warmup=29, burnin=20, skip=1, n_samples=60, cycles=60,
              sampling_decay=sampling_decay, init_method="he",
              load_samples=None, batch_size=batch_size, save_samples=True,
              log_dir=str(log_dir), batchnorm=True)

for model in models:
    for temp in temps:
        for i in range(len(learning_rates)):
            train_bnn(**config, inference="VerletSGLDReject", model=model,
                      lr=learning_rates[i], temperature=temp, momentum=momenta[i])
