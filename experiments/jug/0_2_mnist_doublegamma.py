#!/usr/bin/env python3

import jug
import subprocess
from pathlib import Path
import sys

experiments_dir = Path(__file__).parent.parent

@jug.TaskGenerator
def train_bnn(**config):
    script = experiments_dir / "train_bnn.py"
    args = [sys.executable, script, "with",
            *[f"{k}={v}" for k, v in config.items()]]
    print(f"Running " + " ".join(map(repr, args)))
    return subprocess.run(args)


log_dir = experiments_dir.parent/"logs/0_2_mnist_doublegamma"
jug.set_jugdir(str(log_dir/"jugdir"))

config = dict(data="mnist", inference="SGLD", width=64, warmup=30, burnin=10, weight_scale=1.41,
              skip=1, n_samples=100, lr=0.1, cycles=20, batch_size=128, save_samples=True,
              log_dir=str(log_dir))

model = "datadrivendoublegammaconv"
prior = "datadrivencorrdoublegamma"
for temperature in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    train_bnn(model=model, weight_prior=prior, temperature=temperature, **config)


for model, prior in [("classificationconvnet", "gaussian"),
                     ("classificationconvnet", "improper"),
                     ("correlatedclassificationconvnet", "convcorrnormal"),
                     ("datadrivengaussconv", "datadrivencorrnormal"),
                     ("datadrivendoublegammaconv", "datadrivencorrdoublegamma")]:
    for temperature in [0.0001, 0.005]:
        train_bnn(model=model, weight_prior=prior, temperature=temperature, **config)
