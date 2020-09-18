import numpy as np
import warnings
from tqdm import tqdm
import torch
from .utils import get_cosine_schedule
from . import mcmc
import math
from bnn_priors import prior


# def _named_params_and_buffers(model):
#     # TODO don't copy parameters twice
#     # See issue #76 , https://github.com/ratschlab/projects2020_BNN-priors/issues/76
#     return itertools.chain(
#         model.named_parameters(),
#         model.named_buffers())

class SGLDRunner:
    def __init__(self, model, dataloader, epochs_per_cycle, warmup_epochs,
                 sample_epochs, learning_rate=1e-2, skip=1, metrics_skip=1,
                 temperature=1., data_mult=1., momentum=0., sampling_decay=True,
                 grad_max=1e6, cycles=1, precond_update=None,
                 metrics_saver=None, model_saver=None):
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
            metrics_saver : HDF5Metrics to log metric with a certain name and value
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
        assert sample_epochs % skip == 0

        self.learning_rate = learning_rate
        self.temperature = temperature
        self.eff_num_data = len(dataloader.dataset) * data_mult
        self.momentum = momentum
        self.sampling_decay = sampling_decay
        self.grad_max = grad_max
        self.cycles = cycles
        self.precond_update = precond_update
        self.metrics_saver = metrics_saver
        self.model_saver = model_saver
        if model_saver is None:
            self._samples = {
                name: torch.zeros(torch.Size([self.num_samples*cycles])+p_or_b.shape, dtype=p_or_b.dtype)
                for name, p_or_b in model.state_dict().items()}

        self.param_names, self._params = zip(*model.named_parameters())

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
        self.optimizer = self._make_optimizer(self._params)
        self.optimizer.sample_momentum()

        schedule = get_cosine_schedule(len(self.dataloader) * self.epochs_per_cycle)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer, lr_lambda=schedule)

        epochs_since_start = -1
        step = 0  # used for `self.metrics_saver.add_scalar`, must start at 0 and never reset
        prev_saved_sample = False
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
                    g['temperature'] = 0. if epoch < self.descent_epochs else self.temperature

                for (x, y) in self.dataloader:
                    self.step(step, x, y,
                              store_metrics=(metrics_step % self.metrics_skip == 0
                                             or prev_saved_sample),
                              initial_step=prev_saved_sample)
                    step += 1
                    metrics_step += 1
                self.metrics_saver.flush(every_s=120)

                if self.precond_update is not None and epoch % self.precond_update == 0:
                    self.optimizer.update_preconditioner()

                sampling_epoch = epoch - (self.descent_epochs + self.warmup_epochs)
                if (0 <= sampling_epoch) and (sampling_epoch % self.skip == 0):
                    state_dict = self.model.state_dict()
                    if self.model_saver is None:
                        for name, param in state_dict.items():
                            self._samples[name][(self.num_samples*cycle)+(sampling_epoch//self.skip)] = param
                    else:
                        self.model_saver.add_state_dict(state_dict, step)
                        self.model_saver.flush()
                    prev_saved_sample = True
                else:
                    prev_saved_sample = False

        # Save metrics for the last sample
        (x, y) = next(iter(self.dataloader))
        self.step(step, x, y, store_metrics=True, initial_step=prev_saved_sample)

    def _model_potential_and_grad(self, x, y):
        self.optimizer.zero_grad()
        loss, log_prior, potential = self.model.split_potential_avg(x, y, self.eff_num_data)
        potential.backward()
        for p in self.optimizer.param_groups[0]["params"]:
            p.grad.clamp_(min=-self.grad_max, max=self.grad_max)
        return loss.item(), log_prior.item(), potential.item()

    def step(self, i, x, y, store_metrics, lr_decay=True, initial_step=False):
        """
        Perform one step of SGLD on the model.

        Args:
            x (torch.Tensor): Training input data
            y (torch.Tensor): Training labels
            lr_decay (bool): Flag that controls whether the learning rate should decay after this step

        Returns:
            loss (float): The current loss of the model for x and y
        """
        loss, log_prior, potential = self._model_potential_and_grad(x, y)
        self.optimizer.step(calc_metrics=store_metrics)

        lr = self.optimizer.param_groups[0]["lr"]
        if lr_decay:
            self.scheduler.step()

        if store_metrics:
            self.store_metrics(i=i-1, loss=loss, log_prior=log_prior,
                               potential=potential, lr=lr,
                               corresponds_to_sample=initial_step)
        return loss

    def get_samples(self):
        """
        Returns the acquired SGLD samples from the last run.

        Returns:
            samples (dict): Dictionary of torch.tensors with num_samples*cycles samples for each parameter of the model
        """
        if self.model_saver is None:
            return self._samples
        return self.model_saver.load_samples()

    def store_metrics(self, i, loss, log_prior, potential, lr,
                      corresponds_to_sample,
                      delta_energy=None, total_energy=None, rejected=None):
        est_temperature_all = 0.
        est_config_temp_all = 0.
        all_numel = 0
        add_scalar = self.metrics_saver.add_scalar
        for n, p in zip(self.param_names, self.optimizer.param_groups[0]["params"]):
            state = self.optimizer.state[p]
            add_scalar("preconditioner/"+n, state["preconditioner"], i)
            add_scalar("est_temperature/"+n, state["est_temperature"], i)
            add_scalar("est_config_temp/"+n, state["est_config_temp"], i)

            est_temperature_all += state["est_temperature"] * p.numel()
            est_config_temp_all += state["est_config_temp"] * p.numel()
            all_numel += p.numel()
        add_scalar("est_temperature/all", est_temperature_all / all_numel, i)
        add_scalar("est_config_temp/all", est_config_temp_all / all_numel, i)

        temperature = self.optimizer.param_groups[0]["temperature"]
        add_scalar("temperature", temperature, i)
        add_scalar("loss", loss, i)
        add_scalar("log_prior", log_prior, i)
        add_scalar("potential", potential, i)
        add_scalar("lr", lr, i)

        if i <= 0:
            add_scalar("acceptance/is_sample", int(corresponds_to_sample), i)
        elif corresponds_to_sample:
            add_scalar("acceptance/is_sample", 1, i)

        if delta_energy is not None:
            add_scalar("delta_energy", delta_energy, i)
            add_scalar("total_energy", total_energy, i)
        if rejected is not None:
            add_scalar("acceptance/rejected", int(rejected), i)


class VerletSGLDRunner(SGLDRunner):
    def _make_optimizer(self, params):
        return mcmc.VerletSGLD(
            params=params,
            lr=self.learning_rate, num_data=self.eff_num_data,
            momentum=self.momentum, temperature=self.temperature)

    def step(self, i, x, y, store_metrics, lr_decay=True, initial_step=False):
        loss, log_prior, potential = self._model_potential_and_grad(x, y)
        lr = self.optimizer.param_groups[0]["lr"]

        if i == 0:  # The very first step
            store_metrics = True
            self.optimizer.initial_step(calc_metrics=True)
            self._initial_loss = loss
            self._total_energy = 0.
            delta_energy = self.optimizer.delta_energy(self._initial_loss, loss)
            total_energy = delta_energy

        elif initial_step:  # It's the first step of an epoch, but not the very first
            self.optimizer.final_step(calc_metrics=False)
            if isinstance(self.optimizer, mcmc.HMC):
                self.optimizer.sample_momentum()

            delta_energy = self.optimizer.delta_energy(self._initial_loss, loss)
            self._initial_loss = loss
            self._total_energy += delta_energy
            total_energy = self._total_energy

            store_metrics = True
            # This will calculate metrics using the momentum and parameter left
            # by `final_ste`
            self.optimizer.initial_step(calc_metrics=True)

        else:  # Any intermediate step
            self.optimizer.step(calc_metrics=store_metrics)
            if store_metrics:
                delta_energy = self.optimizer.delta_energy(self._initial_loss, loss)
                total_energy = self._total_energy + delta_energy

        if store_metrics:
            self.store_metrics(i=i-1, loss=loss, log_prior=log_prior,
                               potential=potential, lr=lr,
                               delta_energy=delta_energy,
                               total_energy=total_energy, rejected=None,
                               corresponds_to_sample=initial_step)

        if lr_decay:
            with warnings.catch_warnings():
                # TODO: PyTorch complains about calling the LR step before the optimizer step
                warnings.simplefilter("ignore")
                self.scheduler.step()
        return loss

class HMCRunner(VerletSGLDRunner):
    def _make_optimizer(self, params):
        assert self.temperature == 1.0, "HMC only implemented for temperature=1."
        assert self.momentum == 1.0, "HMC only works with momentum=1."
        return mcmc.HMC(
            params=params,
            lr=self.learning_rate, num_data=self.eff_num_data)
