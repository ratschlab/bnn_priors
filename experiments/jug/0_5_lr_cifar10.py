#!/usr/bin/env python3

import jug
import subprocess
from pathlib import Path
import sys

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

name = Path(__file__).name[:-3]
log_dir = experiments_dir.parent/"logs"/name
jug.set_jugdir(str(log_dir/"jugdir"))

priors=( "gaussian", "convcorrnormal" )
temps=( 0.01, 1.0 )
model="thin_resnet18"
learning_rates=(0.1, 0.03162277660168379, 0.01, 0.0031622776601683794, 0.001,
                0.00031622776601683794, 0.0001, "3.1622776601683795e-05")
batch_size=125
sampling_decay="flat"
load_samples=experiments_dir.parent/"logs"/"0_3_cifar10_sgd"/"0"/"samples.pt"

# momentum = e^(-gamma * h)
# lr = h^2 * num_data = h^2 * 50000
# log momentum = -gamma*h
# gamma = -(log momentum) / sqrt(lr/50000)
#
# Baseline settings:
# lr = 0.1
# momentum = 0.98
# -> gamma = 14.28
#
# [math.exp(-gamma * math.sqrt(lr / 50000)) for lr in 10**np.arange(-1, -5, -0.5)]
momenta=(0.98, 0.988703473184311, 0.9936317070771853, 0.9964138398421518,
         0.9979817686415696, 0.998864563375984, 0.9993613383309885,
         0.9996408039418584)


for prior in priors:
    for temp in temps:
        for i in range(len(learning_rates)):
            config = dict(weight_prior=prior, data="cifar10_augmented",
                          model=model, depth=20, width=16, warmup=9, burnin=0,
                          skip=1, n_samples=100, cycles=100, temperature=temp,
                          sampling_decay=sampling_decay, lr=learning_rates[i],
                          init_method="he", load_samples=str(load_samples),
                          batch_size=batch_size, save_samples=True,
                          log_dir=str(log_dir), batchnorm=True)

            train_bnn(**config, inference="VerletSGLDReject", momentum=momenta[i])
            if temp == 1.0:
                train_bnn(**config, inference="HMCReject", momentum=1.0)
