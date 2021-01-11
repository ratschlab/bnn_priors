#!/usr/bin/env python3

import jug
import subprocess
from pathlib import Path
import sys
import os

experiments_dir = Path(__file__).resolve().parent.parent

@jug.TaskGenerator
def train_sgd(log_dir, **config):
    os.mkdir(log_dir)
    script = experiments_dir / "train_sgd.py"
    args = ["nice", "-n19", sys.executable, script,
            *[f"--{k}={v}" for k, v in config.items()]]
    print(f"Running in cwd={log_dir} " + " ".join(map(repr, args)))
    complete = subprocess.run(args, cwd=log_dir)
    if complete.returncode != 0:
        raise SystemError(f"Process returned with code {complete.returncode}")
    return complete

name = Path(__file__).name[:-3]
base_dir = experiments_dir.parent/"logs"/name
jug.set_jugdir(str(base_dir/"jugdir"))

for i in range(10):
    log_dir = base_dir/str(i)
    train_sgd(str(log_dir), model="thin_resnet18", data="cifar10")

train_sgd(str(base_dir/"mom1"), model="thin_resnet18", data="cifar10",
          momentum=1.0)
