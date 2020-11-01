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
    complete = subprocess.run(args)
    if complete.returncode != 0:
        raise SystemError(f"Process returned with code {complete.returncode}")
    return complete

name = Path(__file__).name[:-3]
log_dir = experiments_dir.parent/"logs"/name
jug.set_jugdir(str(log_dir/"jugdir"))

config = dict(data="cifar10", inference="SGLD", width=64, warmup=30, burnin=10, weight_scale=1.41,
              skip=1, n_samples=100, lr=0.1, cycles=20, batch_size=128, save_samples=True,
              log_dir=str(log_dir), batchnorm=True, model="resnet18")

weight_prior = "gaussian"
bias_prior = "gaussian"
samples_dir = experiments_dir.parent/"logs/0_3_cifar10_sgd/4/samples.pt"
for temperature in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    train_bnn(weight_prior=weight_prior, temperature=temperature,
              load_samples=str(samples_dir), **config)

    train_bnn(weight_prior=weight_prior, temperature=temperature,
              init_method="he", **config)


config["model"] = "datadriven_resnet18"

mean_covs_file = str(experiments_dir.parent/"bnn_priors/models/cifar10_mean_covs.pkl.gz")
fits_dict_file = str(experiments_dir.parent/"bnn_priors/models/cifar10_weight_fits.pkl.gz")

for temperature in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    train_bnn(weight_prior="fixedcov_normal", temperature=temperature,
              load_samples=str(samples_dir), **config,
              weight_prior_params={"mean_covs_file": mean_covs_file})

    train_bnn(weight_prior="fixedcov_gennorm", temperature=temperature,
              load_samples=str(samples_dir), **config,
              weight_prior_params={"beta": 2.0,
                                   "mean_covs_file": mean_covs_file,
                                   "fits_dict_file": fits_dict_file})

for temperature in [1.0]:
    train_bnn(weight_prior="fixedcov_normal", temperature=temperature,
              init_method="he", **config,
              weight_prior_params={"mean_covs_file": mean_covs_file})

    train_bnn(weight_prior="fixedcov_gennorm", temperature=temperature,
              init_method="he", **config,
              weight_prior_params={"beta": 2.0,
                                   "mean_covs_file": mean_covs_file,
                                   "fits_dict_file": fits_dict_file})
