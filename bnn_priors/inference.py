import numpy as np
from tqdm import tqdm
import torch
from pyro.distributions import Normal
from bnn_priors.utils import get_cosine_schedule

class SGLD:
    def __init__(self, model, num_samples, warmup_steps,
                 learning_rate=5e-4, skip=1, temperature=1.,
                 sampling_decay=True, cycles=1):
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
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=learning_rate)
        samples_per_cycle = warmup_steps + (skip * num_samples)
        schedule = get_cosine_schedule(samples_per_cycle)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                                    lr_lambda=schedule)

        
    def run(self, x, y, progressbar=False):
        for cycle in range(self.cycles):
            if progressbar:
                warmup_iter = tqdm(range(self.warmup_steps), position=0, leave=False, desc="Warmup")
                sampling_iter = tqdm(range(self.num_samples * self.skip), position=0, leave=True, desc="Sampling")
            else:
                warmup_iter = range(self.warmup_steps)
                sampling_iter = range(self.num_samples * self.skip)
            for _ in warmup_iter:
                self.step(x, y, noise_injection=False)
            for i in sampling_iter:
                self.step(x, y, lr_decay=self.sampling_decay)
                if i % self.skip == 0:
                    for param, value in self.model.state_dict().items():
                        self._samples[param][(self.num_samples*cycle)+(i//self.skip)] = value

                
    def step(self, x, y, lr_decay=True, noise_injection=True):
        self.optimizer.zero_grad()
        # TODO: this only works when the full data is used,
        # otherwise the log_likelihood should be rescaled according to the batch size
        loss = self.model.potential(x, y)
        loss.backward()
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
        return self._samples
