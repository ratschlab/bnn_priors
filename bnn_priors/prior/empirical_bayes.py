import torch.distributions as td
import torch
import math
from gpytorch.utils.transforms import inv_softplus

from .base import Prior
from .loc_scale import Normal, Laplace, StudentT, GenNorm


__all__ = ('NormalEmpirical',)


class NormalEmpirical(Normal):
    def __init__(self, shape, loc, scale):
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        self.scale_param = torch.nn.Parameter(scale)
        scale = torch.nn.functional.softplus(self.scale_param)
        super().__init__(shape, loc, scale=scale)
