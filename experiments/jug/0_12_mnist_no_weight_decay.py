#!/usr/bin/env python3

import jug
import subprocess
from pathlib import Path
import sys
import os

experiments_dir = Path(__file__).parent.parent

@jug.TaskGenerator
def train_sgd(log_dir, **config):
    os.makedirs(log_dir, exist_ok=False)

    script = experiments_dir / "train_sgd.py"
    args = ["nice", "-n19", sys.executable, script,
            *[f"--{k}={v}" for k, v in config.items()]]
    print(f"Running in cwd={log_dir} " + " ".join(map(repr, args)))
    complete = subprocess.run(args, cwd=log_dir)
    if complete.returncode != 0:
        raise SystemError(f"Process returned with code {complete.returncode}")
    return complete

base_dir = experiments_dir.parent/"logs/sgd-no-weight-decay"
jug.set_jugdir(str(base_dir/"jugdir"))
for net in ["dense", "conv"]:

    for i in reversed(range(10)):
        log_dir = base_dir/f"mnist_classification{net}net"/str(i)
        print(log_dir)
        if net == "dense":
            train_sgd(str(log_dir), model="classificationdensenet", data="mnist", width=100)
        elif net == "conv":
            train_sgd(str(log_dir), model="classificationconvnet", data="mnist", width=64)
        else:
            raise ValueError(net)
