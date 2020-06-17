import numpy as np
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

    def run(self, x, y):
        for _ in range(self.warmup_steps):
            self.step(x, y)
        for i in range(self.num_samples):
            self.step(x, y)
            for param, value in self.model.state_dict().items():
                self._samples[param][i] = value

    def step(self, x, y):
        self.optimizer.zero_grad()
        loss = self.model.potential(x, y)
        loss.backward()
        for param in self.model.parameters():
            param.grad.add_(
                torch.randn(param.grad.size(), dtype=param.dtype,
                            device=param.device)
                .mul_(np.sqrt(2*self.learning_rate)))
        self.optimizer.step()
        return loss.item()

    def get_samples(self):
        return self._samples
