# BNN priors library

This package contains code for specifying NNs with a prior and performing
inference. The overall structure is as follows:

## `exp_utils.py`
This is the highest-level module, that the experiment scripts directly use. It
contains functions to initialize, given a string:

- some specified `Model` with a `Prior` (not all models are compatible with all priors)
- a dataset

It also contains various functions and classes that:
- initialize NN weights in a certain way (He, He with uniform bias, He with zero bias)
- Evaluate the model on a test data set (including test log-probability, calibration error...)
- Store metrics and NN parameters in an HDF5 file
- Load such parameters

## `data/`
Utility code for loading data sets. The data sets themselves also live under
this directory, after they have been downloaded.

## `mcmc/`
The core MCMC algorithms are implemented as subclasses of `torch.Optimizer`,
though their interface is slightly different. **Most of the time, you want to use
one of the higher-level classes in `inference.py` or `inference_reject.py`**

### Naming: "SGLD" is not SGLD
In this project, "SGLD" (stochastic gradient Langevin dynamics) refers to an
algorithm that simulates (*underdamped*) Langevin dynamics using stochastic
gradients. That is commonly known in the deep learning literature as SGHMC.

The algorithm by [Welling and Teh
(2011)](https://dl.acm.org/doi/10.5555/3104482.3104568) simulates *overdamped*
Langevin dynamics, also commonly known as Brownian motion.


### Available algorithms
- `SGLD`: implements Stochastic gradient Hamiltonian Monte Carlo [(Chen et al., 2014)](https://www.arxiv.org/abs/1402.4102), That is, Langevin dynamics with the symplectic Euler integrator. Its acceptance probability is always 0.
- `VerletSGLD`: implements Gradient-guided Monte Carlo [(GGMC; Garriga-Alonso and Fortuin, 2021)](https://arxiv.org/abs/2102.01691). That is, Langevin dynamics with the symmetrised OBABO integrator by [Leimkuhler and Matthews, 2012](https://doi.org/10.1093/amrx/abs010).
- `HMC`: Hamiltonian Monte Carlo. Basically GGMC with zero noise.

## `inference.py` and `inference_reject.py`
Higher-level wrappers over the algorithms in `mcmc/`. They automatically handle
learning rate schedules, adaptation refreshment rates, saving the correct
metrics (including acceptance probability)...

The ones in `inference_reject.py` take an exact gradient step from time to time.
This is to calculate the exact acceptance probability. The ones in
`inference.py` use only stochastic gradients and their acceptance probability
metrics are only illustrative.

## `prior/`
Specify the prior over weights of a neural network, as well as its functional form

The main class used for this purpose is the `Prior` class in `prior/base.py`.
When the neural network code wants the value of the parameter to calculate its
output, it calls `Prior.forward`.

`Prior`s are often defined using a `torch.distribution.Distribution`. We have
defined extra ones in `prior/distributions.py`.

## `models/`
Contains various models (fully-connected nets, convnets, resnets, ...) with
various kinds of priors.

Also defines the basic model class at `models/base.py`, which handles operations
that the MCMC algorithm needs to know about a model: calculating the likelihood,
and calculating the log-prior of the parameters.

## `plot.py`, `notebook_utils.py`
Tools used in the notebooks to load training runs and plot their results.

## `third_party`
Implement various calibration metrics.
