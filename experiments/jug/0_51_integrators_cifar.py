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
temps=( 1.0, 0.01 )
model="googleresnet"
learning_rates=(0.1, 0.01, 0.001, 0.0001, 1e-6)
batch_size=125
sampling_decay="flat"

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

for temp in temps:
    for i in range(len(learning_rates)):
        config = dict(weight_prior=prior, data="cifar10_augmented",
                      model=model, depth=20, width=16, warmup=9, burnin=0,
                      skip=1, n_samples=300, cycles=300, temperature=temp,
                      sampling_decay=sampling_decay, lr=learning_rates[i],
                      init_method="he", load_samples=None,
                      batch_size=batch_size, save_samples=True,
                      log_dir=str(log_dir), batchnorm=True)

        train_bnn(**config, inference="VerletSGLDReject", momentum=momenta[i])
        train_bnn(**config, inference="SGLD", momentum=momenta[i])
        if temp == 1.0:
            train_bnn(**config, inference="HMCReject", momentum=1.0)


config = dict(model=model, data="cifar10_augmented", width=16,
              batch_size=batch_size, sampling_decay=sampling_decay,
              n_epochs=3000)

for i in range(len(learning_rates)):
    train_sgd(log_dir, **config, momentum=momenta[i], lr=learning_rates[i])

for i in range(len(learning_rates)):
    train_sgd(log_dir, **config, momentum=0.9, lr=learning_rates[i])

for i in range(len(learning_rates)):
    train_sgd(log_dir, **config, momentum=0.0, lr=learning_rates[i])


# SGD run with Florian's settings
config["sampling_decay"] = "stairs2"
config["n_epochs"] = 200
config["batch_size"] = 128
florian_path = train_sgd(log_dir, **config, lr=0.1, momentum=0.9, weight_decay=0.002)

# Run everything again but now with an initial load_samples
for temp in temps:
    for i in range(len(learning_rates)):
        config = dict(weight_prior=prior, data="cifar10_augmented",
                      model=model, depth=20, width=16, warmup=9, burnin=0,
                      skip=1, n_samples=300, cycles=300, temperature=temp,
                      sampling_decay=sampling_decay, lr=learning_rates[i],
                      init_method="he", load_samples=str(jug.bvalue(florian_path)/"samples.pt"),
                      batch_size=batch_size, save_samples=True,
                      log_dir=str(log_dir), batchnorm=True)

        train_bnn(**config, inference="VerletSGLDReject", momentum=momenta[i])
        train_bnn(**config, inference="SGLD", momentum=momenta[i])
        if temp == 1.0:
            train_bnn(**config, inference="HMCReject", momentum=1.0)
