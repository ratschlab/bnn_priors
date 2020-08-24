# projects2020_BNN-priors
Benchmark suite for different Bayesian neural network priors

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

## Using sacred

We are using `sacred` (https://github.com/IDSIA/sacred) to manage the experiments.

### Running the training script

The training script (`experiments/train_bnn.py`) takes several different parameters that are defined in the `config()` function within that script. To deviate from the default parameters, sacred uses the command line keyword `with`.
The usage could be for instance

```sh
python train_bnn.py with weight_prior=laplace inference=HMC data=UCI_wine
```

### Reading out the results

Each experiment (so each run of the training script) generates a numbered subdirectory in `logs/`.
The used configuration parameters are stored in this subdirectory in the `config.json` file and the results (the return values of the `main()` method in the training script, e.g., performance metrics) are stored in `run.json`.
