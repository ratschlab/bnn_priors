# Bayesian Neural Network Priors Revisited

This repository contains the code for the paper [Bayesian Neural Network Priors Revisited](https://arxiv.org/abs/2102.06571), as described in the accompanying paper [BNNpriors: A library for Bayesian neural network inference with different prior distributions](https://www.sciencedirect.com/science/article/pii/S2665963821000270).
It allows to perform SG-MCMC inference in BNNs with different architectures and priors on a range of tasks.


## Installation

After cloning the repository, the package can be installed from inside the main directory with

```sh
pip install -e .
```

The `-e` makes the installation be in "development mode", so any changes you
make to the code in the repository will be reflected in the `bnn_priors` package
you can import.

The code has run at some point with Python 3.6, 3.7 and 3.8.


## Running experiments

We are using `sacred` (https://github.com/IDSIA/sacred) to manage the experiments.

### Running the training script

The training script (`experiments/train_bnn.py`) takes several different parameters that are defined in the `config()` function within that script. Please refer to the comments in that script for a more detailed documentation of the different parameters. To deviate from the default parameters, sacred uses the command line keyword `with`.
The usage could be for instance

```sh
python train_bnn.py with weight_prior=laplace inference=SGLD data=mnist
```

The last of the samples from a past run can be loaded using the `load_samples=/path/to/samples.pt` option.

### Running the SGD training script

The logs are saved to the current directory when running. The script is

``` sh
mkdir -p logs/1
cd logs/1
python ../../experiments/train_sgd.py --lr=0.05 --model=thin_resnet18
```

### Reading out the results

Each experiment (so each run of the training script) generates a numbered subdirectory in `logs/`.
The used configuration parameters are stored in this subdirectory in the `config.json` file and the results (the return values of the `main()` method in the training script, e.g., performance metrics) are stored in `run.json`.


### Parameters for the experiments in the paper

The experiments in the paper where run with the following parameter settings.

MNIST FCNN:

```sh
python train_bnn.py with data=mnist model=classificationdensenet weight_prior=gaussian inference=VerletSGLDReject warmup=45 burnin=0 skip=1 n_samples=300 lr=0.01 momentum=0.994 weight_scale=1.41 cycles=60 batch_size=128 temperature=1.0 save_samples=True progressbar=False log_dir=../results/exp_mnist_fcnn batchnorm=True
```

MNIST CNN:

```sh
python train_bnn.py with data=mnist model=classificationconvnet weight_prior=gaussian inference=VerletSGLDReject warmup=45 burnin=0 skip=1 n_samples=300 lr=0.01 momentum=0.994 weight_scale=1.41 cycles=60 batch_size=128 temperature=1.0 save_samples=True progressbar=False log_dir=../results/exp_mnist_cnn batchnorm=True
```

FMNIST FCNN:

```sh
python train_bnn.py with data=fashion_mnist model=classificationdensenet weight_prior=gaussian inference=VerletSGLDReject warmup=45 burnin=0 skip=1 n_samples=300 lr=0.01 momentum=0.994 weight_scale=1.41 cycles=60 batch_size=128 temperature=1.0 save_samples=True progressbar=False log_dir=../results/exp_fmnist_fcnn batchnorm=True
```

FMNIST CNN:

```sh
python train_bnn.py with data=fashion_mnist model=classificationconvnet weight_prior=gaussian inference=VerletSGLDReject warmup=45 burnin=0 skip=1 n_samples=300 lr=0.01 momentum=0.994 weight_scale=1.41 cycles=60 batch_size=128 temperature=1.0 save_samples=True progressbar=False log_dir=../results/exp_fmnist_cnn batchnorm=True
```

CIFAR10:

```sh
python train_bnn.py with data=cifar10_augmented model=googleresnet weight_prior=gaussian inference=VerletSGLDReject warmup=45 burnin=0 skip=1 n_samples=300 lr=0.01 momentum=0.994 weight_scale=1.41 cycles=60 batch_size=128 temperature=1.0 save_samples=True progressbar=False log_dir=../results/exp_cifar batchnorm=True
```

To run these experiments with different priors and temperatures, such that tempering curves similar to the ones in the paper can be plotted, one can use the bash script `experiments/run_experiment.sh`.

### Evaluating the trained models

During training, the models automatically evaluate the accuracy and negative log-likelihood on the test dataset and save the results into the training run directory.
To also evaluate the uncertainty calibration and out-of-distribution detection, the `experiments/eval_bnn.py` script can be used.
For OOD detection of a trained MNIST model, the script could for instance be run as

```sh
python eval_bnn.py with config_file=../results/exp_mnist_cnn/config.json ood_eval=True eval_data=fashion_mnist skip_first=50
```

The evaluation that we used for the experiments in our paper (including accuracy, NLL, ECE, and OOD AUROC) can be run for a trained model using `experiments/run_evaluation.sh`.
The respective settings for training data, calibration data, and OOD data would be ("mnist", "rotated_mnist", "fashion_mnist"), ("fashion_mnist", "fashion_mnist", "mnist"), and ("cifar10", "cifar10c-gaussian_blur", "svhn").


## Running the tests

### Python's unittest

The easiest way is to run them using Python's own test library. Assuming you're
in the repository root:

```sh
python -m unittest
```
To run a single test, you have to use module path loading syntax:

```sh
# All tests in file
python -m unittest testing.test_models
# Run all tests in a class
python -m unittest testing.test_models.TestRaoBDenseNet
# Run a single test
python -m unittest testing.test_models.TestRaoBDenseNet.test_likelihood
```
which requires that `testing` be a valid module, so it must have an `__init__.py` file.

### Py.test (easier but needs installation)

Alternatively, you can use other runners, such as `py.test`.

```sh
pip install pytest
py.test .
```

To run a class of tests and a test:
```sh
# File
py.test testing/test_models.py
# Class
py.test testing/test_models.py::TestRaoBDenseNet
# single test
py.test testing/test_models.py::TestRaoBDenseNet::test_likelihood
```


## Cite this work

If you are using this codebase in your work, please cite it as

```
@article{fortuin2021bnnpriors,
  title={{BNNpriors}: A library for {B}ayesian neural network inference with different prior distributions},
  author={Fortuin, Vincent and Garriga-Alonso, Adri{\`a} and van der Wilk, Mark and Aitchison, Laurence},
  journal={Software Impacts},
  volume={9},
  pages={100079},
  year={2021},
  publisher={Elsevier}
}
```

If you would also like to cite our results regarding different BNN priors, please cite

```
@article{fortuin2021bayesian,
  title={{B}ayesian neural network priors revisited},
  author={Fortuin, Vincent and Garriga-Alonso, Adri{\`a} and Wenzel, Florian and R{\"a}tsch, Gunnar and Turner, Richard and van der Wilk, Mark and Aitchison, Laurence},
  journal={arXiv preprint arXiv:2102.06571},
  year={2021}
}
```

Finally, if you would like to cite the GG-MC inference algorithm used in this package, please cite

```
@article{garriga2021exact,
  title={Exact langevin dynamics with stochastic gradients},
  author={Garriga-Alonso, Adri{\`a} and Fortuin, Vincent},
  journal={arXiv preprint arXiv:2102.01691},
  year={2021}
}
```
