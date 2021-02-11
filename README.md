# projects2020_BNN-priors
Benchmark suite for different Bayesian neural network priors



## Using sacred

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
