# projects2020_BNN-priors
Benchmark suite for different Bayesian neural network priors

## Running the tests

### Python's unittest

The easiest way is to run them using Python's own test library. Assuming you're
in the repository root:

```sh
python -m unittest discover ./testing
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
