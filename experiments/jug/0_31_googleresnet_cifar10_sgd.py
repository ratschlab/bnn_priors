#!/usr/bin/env python3

import jug
import subprocess
from pathlib import Path
import sys
import os
import time
import random

experiments_dir = Path(__file__).resolve().parent.parent

@jug.TaskGenerator
def train_sgd(j, log_dir, **config):
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
base_dir = experiments_dir.parent/"logs"/name
jug.set_jugdir(str(base_dir/"jugdir"))

for i in range(10):
    train_sgd(i, base_dir, model="googleresnet", data="cifar10_augmented",
              lr=0.05, momentum=0.9, batch_size=128, sampling_decay="stairs",
              n_epochs=150*4)
    train_sgd(i, base_dir, model="googleresnet", data="cifar10_augmented",
              lr=0.1, momentum=0.9, batch_size=128, sampling_decay="stairs2",
              n_epochs=200)
    train_sgd(i, base_dir, model="googleresnet", data="cifar10_augmented",
              lr=0.1, momentum=0.9, batch_size=128, sampling_decay="stairs2",
              n_epochs=200, weight_decay=0.002)

    # unaugmented CIFAR10
    train_sgd(i, base_dir, model="googleresnet", data="cifar10",
              lr=0.05, momentum=0.9, batch_size=128, sampling_decay="stairs",
              n_epochs=150*4)

    # classificationconvnet
    train_sgd(i, base_dir, model="classificationconvnet", data="mnist",
              lr=0.05, momentum=0.9, batch_size=128, sampling_decay="stairs",
              n_epochs=150*4)

    train_sgd(i, base_dir, model="classificationconvnet", data="fashion_mnist",
              lr=0.05, momentum=0.9, batch_size=128, sampling_decay="stairs",
              n_epochs=150*4)

    # classificationdensenet
    train_sgd(i, base_dir, model="classificationdensenet", data="mnist",
              lr=0.05, momentum=0.9, batch_size=128, sampling_decay="stairs",
              n_epochs=150*4)

    train_sgd(i, base_dir, model="classificationdensenet", data="fashion_mnist",
              lr=0.05, momentum=0.9, batch_size=128, sampling_decay="stairs",
              n_epochs=150*4)
