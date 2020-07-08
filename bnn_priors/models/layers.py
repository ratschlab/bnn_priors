import torch.nn as nn

__all__ = ('Linear',)

class Linear(nn.Linear):
    def __init__(self, weight_prior, bias_prior=None):
        nn.Module.__init__(self)
        (self.out_features, self.in_features) = weight_prior.p.shape
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior

    @property
    def weight(self):
        return self.weight_prior()

    @property
    def bias(self):
        return (None if self.bias_prior is None else self.bias_prior())
