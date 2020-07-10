import numpy as np
from tqdm import tqdm
import torch
from .utils import get_cosine_schedule
from .sgld import SGLD
import math
from bnn_priors import prior


class SGLDRunner:
    def __init__(self, model, num_samples, warmup_steps, burnin_steps=None, learning_rate=5e-4,
                 skip=1, temperature=1., momentum=0., sampling_decay=True,
                 grad_max=1e6, cycles=1, precond_update=None, summary_writer=None):
        """
        Stochastic Gradient Langevin Dynamics for posterior sampling.

        Args:
            model (torch.Module, PriorMixin): BNN model to sample from
            num_samples (int): Number of samples to draw per cycle
            warmup_steps (int): Number of steps per cycle for warming up the Markov chain
            burnin_steps (int): Number of steps per cycle between warmup and sampling. When None, uses the same as warmup_steps.
            learning_rate (float): Initial learning rate
            skip (int): Number of samples to skip between saved samples during the sampling phase
            temperature (float): Temperature for tempering the posterior
            momentum (float): Momentum decay parameter for SGLD
            sampling_decay (bool): Flag to control whether the learning rate should decay during sampling
            grad_max (float): maximum absolute magnitude of an element of the gradient
            cycles (int): Number of warmup and sampling cycles to perform
            precond_update (int): Number of steps after which the preconditioner should be updated. None disables the preconditioner.
            summary_writer (optional, tensorboardX.SummaryWriter): where to write the self.metrics
        """
        self.model = model
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.burnin_steps = warmup_steps if burnin_steps is None else burnin_steps
        self.learning_rate = learning_rate
        self.skip = skip
        self.temperature = temperature
        self.momentum = momentum
        self.sampling_decay = sampling_decay
        self.grad_max = grad_max
        self.cycles = cycles
        self.precond_update = precond_update
        self.summary_writer = summary_writer
        # TODO: is there a nicer way than adding this ".p" here?
        self._samples = {name+".p" : torch.zeros(torch.Size([num_samples*cycles])+param.shape)
                         for name, param in self.model.params_with_prior_dict().items()}
        self._samples["lr"] = torch.zeros(torch.Size([num_samples*cycles]))
        self.samples_per_cycle = self.warmup_steps + self.burnin_steps + (self.skip * self.num_samples)

        self.metrics = {}

    def run(self, x, y, progressbar=False):
        """
        Runs the sampling on the model.

        Args:
            x (torch.tensor): Training input data
            y (torch.tensor): Training labels
            progressbar (bool): Flag that controls whether a progressbar is printed
        """
        self.param_names, params = zip(*prior.named_params_with_prior(self.model))
        self.optimizer = SGLD(
            params=params,
            lr=self.learning_rate, num_data=len(x),
            momentum=self.momentum, temperature=self.temperature)

        schedule = get_cosine_schedule(self.samples_per_cycle)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer, lr_lambda=schedule)

        for cycle in range(self.cycles):
            if progressbar:
                warmup_iter = tqdm(range(self.warmup_steps), position=0,
                                   leave=False, desc=f"Cycle {cycle}, Warmup")
                burnin_iter = tqdm(range(self.burnin_steps), position=0,
                                   leave=False, desc=f"Cycle {cycle}, Burn-in")
                sampling_iter = tqdm(range(self.num_samples * self.skip), position=0,
                                     leave=True, desc=f"Cycle {cycle}, Sampling")
            else:
                warmup_iter = range(self.warmup_steps)
                burnin_iter = range(self.burnin_steps)
                sampling_iter = range(self.num_samples * self.skip)

            for g in self.optimizer.param_groups:
                g['temperature'] = 0
            for warmup_i in warmup_iter:
                self.step(warmup_i, x, y)
                # TODO: should we also do this during sampling?
                if self.precond_update is not None and warmup_i % self.precond_update == 0:
                    # TODO: how do we actually handle minibatches here?
                    self.optimizer.estimate_preconditioner(closure=lambda x: x, K=1)
            warmup_i += 1

            for g in self.optimizer.param_groups:
                g['temperature'] = self.temperature
                
            for burnin_i in burnin_iter:
                self.step(warmup_i+burnin_i, x, y)
                # TODO: should we also do this during sampling?
                if self.precond_update is not None and burnin_i % self.precond_update == 0:
                    # TODO: how do we actually handle minibatches here?
                    self.optimizer.estimate_preconditioner(closure=lambda x: x, K=1)
            burnin_i += 1

            # TODO: should it be possible to change the learning rate before sampling?
            for i in sampling_iter:
                self.step(warmup_i+burnin_i+i, x, y, lr_decay=self.sampling_decay)
                if i % self.skip == 0:
                    for name, param in self.model.params_with_prior_dict().items():
                        # TODO: is there a more elegant way than adding this ".p" here?
                        self._samples[name+".p"][(self.num_samples*cycle)+(i//self.skip)] = param
                    self._samples["lr"][(self.num_samples*cycle)+(i//self.skip)] = self.optimizer.param_groups[0]["lr"]

    def add_scalar(self, name, value, step):
        try:
            self.metrics[name].append(value)
        except KeyError:
            self.metrics[name] = []
            self.metrics[name].append(value)
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(name, value, step)

    def step(self, i, x, y, lr_decay=True):
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
        # TODO: this only works when the full data is used,
        # otherwise the log_likelihood should be rescaled according to the batch size
        # TODO: should we multiply this by the batch size somehow?
        loss = self.model.potential(x, y) / len(x)
        loss.backward()
        for p in self.optimizer.param_groups[0]["params"]:
            p.grad.clamp_(min=-self.grad_max, max=self.grad_max)
        self.optimizer.step()
        if lr_decay:
            self.scheduler.step()

        for n, p in zip(self.param_names, self.optimizer.param_groups[0]["params"]):
            state = self.optimizer.state[p]
            self.add_scalar("preconditioner/"+n, state["preconditioner"], i)
            self.add_scalar("est_temperature/"+n, state["est_temperature"], i)
            self.add_scalar("est_config_temp/"+n, state["est_config_temp"], i)

        self.add_scalar("lr", self.optimizer.param_groups[0]["lr"], i)
        loss_ = loss.item()
        self.add_scalar("loss", loss_, i)
        if i > 0:
            self.add_scalar("log_prob_accept", self.prev_loss - loss_, i-1)

        self.prev_loss = loss_
        return loss_


    def get_samples(self):
        """
        Returns the acquired SGLD samples from the last run.

        Returns:
            samples (dict): Dictionary of torch.tensors with num_samples*cycles samples for each parameter of the model
        """
        return self._samples
