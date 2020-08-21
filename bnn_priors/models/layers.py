import torch.nn as nn

__all__ = ('Linear','Conv2d')

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


class Conv2d(nn.Conv2d):
    def __init__(self, weight_prior, bias_prior=None, stride=1,
            padding=0, dilation=1, groups=1, padding_mode='zeros'):
        nn.Module.__init__(self)
        
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)
        self.dilation = nn.modules.utils._pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.transposed = False
        self.output_padding = nn.modules.utils._pair(0)
        
        (self.out_channels, in_channels, ksize_0, ksize_1) = weight_prior.p.shape
        self.in_channels = in_channels * self.groups
        self.kernel_size = (ksize_0, ksize_1)
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior

    @property
    def weight(self):
        return self.weight_prior()

    @property
    def bias(self):
        return (None if self.bias_prior is None else self.bias_prior())
