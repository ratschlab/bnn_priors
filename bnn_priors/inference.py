import numpy as np
from tqdm import tqdm
import torch
from .utils import get_cosine_schedule
from . import mcmc
import math
from bnn_priors import prior


class SGLDRunner:
    def __init__(self, model, dataloader, epochs_per_cycle, warmup_epochs,
                 sample_epochs, learning_rate=1e-2, skip=1, metrics_skip=1,
                 temperature=1., data_mult=1., momentum=0., sampling_decay=True,
                 grad_max=1e6, cycles=1, precond_update=None,
                 add_scalar_fn=None):
        """Stochastic Gradient Langevin Dynamics for posterior sampling.

        On calling `run`, this class runs SGLD for `cycles` sampling cycles. In
        each cycle, there are 3 phases: descent, warmup and sampling. The cycle
        lasts for `epochs_per_cycle` epochs in total, and the warmup and
        sampling phases last for `warmup_epochs` and `sample_epochs` epochs
        respectively.

        The descent phase performs regular gradient descent with momentum, i.e.
        SGLD with temperature=0. The warmup phase raises the temperature to 1.
        During the sample phase, samples get stored.

        The learning rate keep decreasing all throughout the cycle following a
        cosine function, from learning_rate=1 at the beginning to
        learning_rate=0 at the end.

        The preconditioner gets updated every `precond_update` epochs,
        regardless of the phase in the cycle.

        Args:
            model (torch.Module, PriorMixin): BNN model to sample from
            num_data (int): Number of datapoints in training sest
            warmup_epochs (int): Number of epochs per cycle for warming up the Markov chain, at the beginning.
            sample_epochs (int): Number of epochs per cycle where the samples are kept, at the end.

            learning_rate (float): Initial learning rate
            skip (int): Number of samples to skip between saved samples during the sampling phase. Sometimes called "thinning".
            metrics_skip (int): Number of samples to skip between saved metrics of the sampler
            temperature (float): Temperature for tempering the posterior
            data_mult (float): Effective replication of each datapoint (which is the usual approach to tempering in VI).
            momentum (float): Momentum decay parameter for SGLD
            sampling_decay (bool): Flag to control whether the learning rate should decay during sampling
            grad_max (float): maximum absolute magnitude of an element of the gradient
            cycles (int): Number of warmup and sampling cycles to perform
            precond_update (int): Number of steps after which the preconditioner should be updated. None disables the preconditioner.
            add_scalar_fn (optional, (str, float, int) -> None): function to log metric with a certain name and value

        """
        self.model = model
        self.dataloader = dataloader

        assert warmup_epochs >= 0
        assert sample_epochs >= 0
        assert epochs_per_cycle >= warmup_epochs + sample_epochs
        self.epochs_per_cycle = epochs_per_cycle
        self.descent_epochs = epochs_per_cycle - warmup_epochs - sample_epochs
        self.warmup_epochs = warmup_epochs
        self.sample_epochs = sample_epochs

        self.skip = skip
        self.metrics_skip = metrics_skip
        # num_samples (int): Number of recorded per cycle
        self.num_samples = sample_epochs // skip
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.eff_num_data = len(dataloader.dataset) * data_mult
        self.momentum = momentum
        self.sampling_decay = sampling_decay
        self.grad_max = grad_max
        self.cycles = cycles
        self.precond_update = precond_update
        self.add_scalar_fn = add_scalar_fn
        self._samples = {name: torch.zeros(torch.Size([self.num_samples*cycles])+param.shape)
                         for name, param in prior.named_params_with_prior(model)}

        self.metrics = {}

    def _make_optimizer(self, params):
        return mcmc.SGLD(
            params=params,
            lr=self.learning_rate, num_data=self.eff_num_data,
            momentum=self.momentum, temperature=self.temperature)

    def run(self, progressbar=False):
        """
        Runs the sampling on the model.

        Args:
            x (torch.tensor): Training input data
            y (torch.tensor): Training labels
            progressbar (bool): Flag that controls whether a progressbar is printed
        """
        self.param_names, params = zip(*prior.named_params_with_prior(self.model))
        self.optimizer = self._make_optimizer(params)
        self.optimizer.sample_momentum()

        schedule = get_cosine_schedule(len(self.dataloader) * self.epochs_per_cycle)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer, lr_lambda=schedule)

        epochs_since_start = -1
        step = 0  # only used for `add_scalar`, must start at 0 and never reset
        for cycle in range(self.cycles):
            metrics_step = 0  # Used for metrics skip

            if progressbar:
                epochs = tqdm(range(self.epochs_per_cycle), position=0,
                              leave=True, desc=f"Cycle {cycle}, Sampling")
            else:
                epochs = range(self.epochs_per_cycle)

            for epoch in epochs:
                epochs_since_start += 1

                for g in self.optimizer.param_groups:
                    g['temperature'] = 0 if epoch < self.descent_epochs else self.temperature

                for (x, y) in self.dataloader:
                    store_metrics = (metrics_step % self.metrics_skip == 0)
                    self.step(step, x, y, store_metrics)
                    step += 1
                    metrics_step += 1

                if self.precond_update is not None and epoch % self.precond_update == 0:
                    self.optimizer.update_preconditioner()

                sampling_epoch = epoch - (self.descent_epochs + self.warmup_epochs)
                if (0 <= sampling_epoch) and (sampling_epoch % self.skip == 0):
                    for name, param in prior.named_params_with_prior(self.model):
                        self._samples[name][(self.num_samples*cycle)+(sampling_epoch//self.skip)] = param

    def add_scalar(self, name, value, step):
        try:
            self.metrics[name].append(value)
        except KeyError:
            self.metrics[name] = []
            self.metrics[name].append(value)
        if self.add_scalar_fn is not None:
            self.add_scalar_fn(name, value, step)

    def step(self, i, x, y, store_metrics, lr_decay=True):
        """
        Perform one step of SGLD on the model.

        Args:
            x (torch.Tensor): Training input data
            y (torch.Tensor): Training labels
            lr_decay (bool): Flag that controls whether the learning rate should decay after this step

        Returns:
            loss (float): The current loss of the model for x and y
        """
        self.optimizer.zero_grad()
        loss = self.model.potential_avg(x, y, self.eff_num_data)
        loss.backward()
        for p in self.optimizer.param_groups[0]["params"]:
            p.grad.clamp_(min=-self.grad_max, max=self.grad_max)
        self.optimizer.step()

        lr = self.optimizer.param_groups[0]["lr"]
        if lr_decay:
            self.scheduler.step()

        loss_ = loss.item()
        if store_metrics:
            self.store_metrics(i=i, loss_=loss_, lr=lr)
        return loss_

    def get_samples(self):
        """
        Returns the acquired SGLD samples from the last run.

        Returns:
            samples (dict): Dictionary of torch.tensors with num_samples*cycles samples for each parameter of the model
        """
        return self._samples

    def store_metrics(self, i, loss_, lr, delta_energy=None, log_accept_prob=None, rejected=None):
        est_temperature_all = 0.
        est_config_temp_all = 0.
        all_numel = 0
        for n, p in zip(self.param_names, self.optimizer.param_groups[0]["params"]):
            state = self.optimizer.state[p]
            self.add_scalar("preconditioner/"+n, state["preconditioner"], i)
            self.add_scalar("est_temperature/"+n, state["est_temperature"], i)
            self.add_scalar("est_config_temp/"+n, state["est_config_temp"], i)

            est_temperature_all += state["est_temperature"] * p.numel()
            est_config_temp_all += state["est_config_temp"] * p.numel()
            all_numel += p.numel()
        self.add_scalar("est_temperature/all", est_temperature_all / all_numel, i)
        self.add_scalar("est_config_temp/all", est_config_temp_all / all_numel, i)

        temperature = self.optimizer.param_groups[0]["temperature"]
        self.add_scalar("temperature", temperature, i)
        self.add_scalar("loss", loss_, i)
        self.add_scalar("lr", lr, i)

        if delta_energy is not None:
            self.add_scalar("energy", delta_energy, i)
            self.add_scalar("acceptance/log_prob", log_accept_prob, i)
            self.add_scalar("acceptance/rejected", int(rejected), i)


class VerletSGLDRunner(SGLDRunner):
    def _make_optimizer(self, params):
        return mcmc.VerletSGLD(
            params=params,
            lr=self.learning_rate, num_data=self.eff_num_data,
            momentum=self.momentum, temperature=self.temperature)

    def step(self, i, x, y, store_metrics, lr_decay=True):
        if i == 0:
            self.optimizer.zero_grad()
            loss = self.model.potential_avg(x, y, self.eff_num_data)
            loss.backward()
            for p in self.optimizer.param_groups[0]["params"]:
                p.grad.clamp_(min=-self.grad_max, max=self.grad_max)
            self._initial_loss = loss.item()

        self.optimizer.initial_step()

        self.optimizer.zero_grad()
        loss = self.model.potential_avg(x, y, self.eff_num_data)
        loss.backward()
        for p in self.optimizer.param_groups[0]["params"]:
            p.grad.clamp_(min=-self.grad_max, max=self.grad_max)

        self.optimizer.final_step()

        lr = self.optimizer.param_groups[0]["lr"]
        if lr_decay:
            self.scheduler.step()

        loss_ = loss.item()
        delta_energy = self.optimizer.delta_energy(self._initial_loss, loss_)
        # Because we never `commit` the sampler by calling `self.optimizer.final_step`,
        # the delta_energy is relative to the initial state.
        rejected, log_accept_prob = self.optimizer.maybe_reject(delta_energy)
        if isinstance(self.optimizer, mcmc.HMC):
            self.optimizer.sample_momentum()
        self._initial_loss = loss_

        if store_metrics:
            self.store_metrics(i=i, loss_=loss_, lr=lr, delta_energy=delta_energy,
                               log_accept_prob=log_accept_prob,
                               rejected=rejected)
        return loss_

class HMCRunner(VerletSGLDRunner):
    def _make_optimizer(self, params):
        assert self.temperature == 1.0, "HMC only implemented for temperature=1."
        assert self.momentum == 1.0, "HMC only works with momentum=1."
        return mcmc.HMC(
            params=params,
            lr=self.learning_rate, num_data=self.eff_num_data)
