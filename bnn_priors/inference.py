import numpy as np
from tqdm import tqdm
import torch
from .utils import get_cosine_schedule
from .sgld import SGLD
import math


class SGLDRunner:
    def __init__(self, model, num_samples, warmup_steps,
                 learning_rate=5e-4, skip=1, temperature=1.,
                 sampling_decay=True, cycles=1):
        """
        Stochastic Gradient Langevin Dynamics for posterior sampling.
        
        Args:
            model (torch.Module, PriorMixin): BNN model to sample from
            num_samples (int): Number of samples to draw per cycle
            warmup_steps (int): Number of steps per cycle for warming up the Markov chain
            learning_rate (float): Initial learning rate
            skip (int): Number of samples to skip between saved samples during the sampling phase
            temperature (float): Temperature for tempering the posterior
            sampling_decay (bool): Flag to control whether the learning rate should decay during sampling
            cycles (int): Number of warmup and sampling cycles to perform
        """
        self.model = model
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.skip = skip
        self.sampling_decay = sampling_decay
        self.temperature = temperature
        self.cycles = cycles
        self._samples = {name : torch.zeros(torch.Size([num_samples*cycles])+param.shape)
                         for name, param in self.model.params_with_prior()}
        self._samples["lr"] = torch.zeros(torch.Size([num_samples*cycles]))
        self.samples_per_cycle = warmup_steps + (skip * num_samples)


    def run(self, x, y, progressbar=False):
        """
        Runs the sampling on the model.

        Args:
            x (torch.tensor): Training input data
            y (torch.tensor): Training labels
            progressbar (bool): Flag that controls whether a progressbar is printed
        """
        self.optimizer = SGLD(
            params=[v for _, v in self.model.params_with_prior()],
            lr=self.learning_rate, num_data=len(x),
            momentum=self.momentum, temperature=self.temperature)
        schedule = get_cosine_schedule(self.samples_per_cycle)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer, lr_lambda=schedule)

        for cycle in range(self.cycles):
            if progressbar:
                warmup_iter = tqdm(range(self.warmup_steps), position=0,
                                   leave=False, desc=f"Cycle {cycle}, Warmup")
                sampling_iter = tqdm(range(self.num_samples * self.skip), position=0,
                                     leave=True, desc=f"Cycle {cycle}, Sampling")
            else:
                warmup_iter = range(self.warmup_steps)
                sampling_iter = range(self.num_samples * self.skip)

            self.optimizer.param_groups['temperature'] = 0
            for _ in warmup_iter:
                self.step(x, y, noise_injection=False)
            for g in self.optimizer.param_groups:
                g['temperature'] = self.temperature

            for i in sampling_iter:
                self.step(x, y, lr_decay=self.sampling_decay)
                if i % self.skip == 0:
                    for name, param in self.model.params_with_prior():
                        self._samples[name][(self.num_samples*cycle)+(i//self.skip)] = value
                    self._samples["lr"][(self.num_samples*cycle)+(i//self.skip)] = self.optimizer.param_groups[0]["lr"]

                
    def step(self, x, y, lr_decay=True, noise_injection=True):
        """
        Perform one step of SGLD on the model.
        
        Args:
            x (torch.Tensor): Training input data
            y (torch.Tensor): Training labels
            lr_decay (bool): Flag that controls whether the learning rate should decay after this step
            noise_injection (bool): Flag that controls whether noise should be injected (False yields SGD, True SGLD)
            
        Returns:
            loss (float): The current loss of the model for x and y
        """
        self.optimizer.zero_grad()
        # TODO: this only works when the full data is used,
        # otherwise the log_likelihood should be rescaled according to the batch size
        loss = self.model.potential(x, y)
        loss.backward()
        # for param in self.model.parameters():
        #     param.grad.mul_(self.temperature)  Or something like this
        self.optimizer.step()
        if lr_decay:
            self.scheduler.step()
        if noise_injection:
            current_lr = self.optimizer.param_groups[0]["lr"]
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(
                        torch.randn(param.size(), dtype=param.dtype,
                                    device=param.device)
                        .mul_(np.sqrt(2*current_lr*self.temperature)))
        return loss.item()

    
    def get_samples(self):
        """
        Returns the acquired SGLD samples from the last run.
        
        Returns:
            samples (dict): Dictionary of torch.tensors with num_samples*cycles samples for each parameter of the model
        """
        return self._samples
