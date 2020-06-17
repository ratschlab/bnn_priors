import numpy as np
from tqdm import tqdm
import torch
from pyro.distributions import Normal


class SGLD:
    def __init__(self, model, num_samples, warmup_steps, learning_rate=5e-4):
        self.model = model
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self._samples = {name : torch.zeros(torch.Size([num_samples])+param.shape)
                         for name, param in self.model.params_with_prior()}
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=learning_rate)

        
    def run(self, x, y, progressbar=False):
        if progressbar:
            warmup_iter = tqdm(range(self.warmup_steps), position=0, leave=True)
            sampling_iter = tqdm(range(self.num_samples), position=0, leave=True)
        else:
            warmup_iter = range(self.warmup_steps)
            sampling_iter = range(self.num_samples)
        for _ in warmup_iter:
            self.step(x, y)
        for i in sampling_iter:
            self.step(x, y)
            for param, value in self.model.state_dict().items():
                self._samples[param][i] = value

                
    def step(self, x, y):
        self.optimizer.zero_grad()
        # TODO: this only works when the full data is used, otherwise the log_likelihood should be rescaled according to the batch size
        loss = self.model.potential(x, y)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(
                    torch.randn(param.size(), dtype=param.dtype,
                                device=param.device)
                    .mul_(np.sqrt(2*self.learning_rate)))
        return loss.item()

    
    def get_samples(self):
        return self._samples
