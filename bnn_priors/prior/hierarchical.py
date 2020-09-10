import torch.distributions as td
import torch
import math
from gpytorch.utils.transforms import inv_softplus

from .base import Prior
from .loc_scale import Normal, Laplace, StudentT
from .transformed import Gamma, Uniform, HalfCauchy


__all__ = ('NormalGamma', 'NormalUniform', 'Horseshoe', 'LaplaceGamma',
          'LaplaceUniform', 'StudentTGamma', 'StudentTUniform')

class _ClipHookMixin:
    def _init(self, clip):
        self.scale.p.register_hook(self.hook)
        self.clip = clip

    def hook(self, grad):
        # TODO: This somehow affects the downstream gradients of the parameters, which it shouldn't
        # It should only affect the actual scale.p parameter
        return torch.clamp(grad, -self.clip, self.clip)

class NormalGamma(Normal, _ClipHookMixin):
    def __init__(self, shape, loc, scale, rate=1., gradient_clip=1.):
        scale_prior = Gamma(shape=[], concentration=scale, rate=rate)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale_prior)
        _ClipHookMixin._init(self, gradient_clip)


class NormalUniform(Normal):
    def __init__(self, shape, loc, scale, gradient_clip=1.):
        scale_prior = Uniform(shape=[], low=0., high=scale*2.)
        with torch.no_grad():
            scale_prior.p.data = torch.tensor(0.)
        super().__init__(shape, loc, scale_prior)
        _ClipHookMixin._init(self, gradient_clip)


    
class LaplaceGamma(Laplace):
    def __init__(self, shape, loc, scale, rate=1., gradient_clip=1.):
        scale_prior = Gamma(shape=[], concentration=scale, rate=rate)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale_prior)
        _ClipHookMixin._init(self, gradient_clip)

    
class LaplaceUniform(Laplace):
    def __init__(self, shape, loc, scale, gradient_clip=1.):
        scale_prior = Uniform(shape=[], low=0., high=scale*2.)
        with torch.no_grad():
            scale_prior.p.data = torch.tensor(0.)
        super().__init__(shape, loc, scale_prior)
        _ClipHookMixin._init(self, gradient_clip)


class StudentTGamma(StudentT):
    def __init__(self, shape, loc, scale, rate=1., df=2, gradient_clip=1.):
        scale_prior = Gamma(shape=[], concentration=scale, rate=rate)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale_prior, df=df)
        _ClipHookMixin._init(self, gradient_clip)

    
class StudentTUniform(StudentT):
    def __init__(self, shape, loc, scale, df=2, gradient_clip=1.):
        scale_prior = Uniform(shape=[], low=0., high=scale*2.)
        with torch.no_grad():
            scale_prior.p.data = torch.tensor(0.)
        super().__init__(shape, loc, scale_prior, df=df)
        _ClipHookMixin._init(self, gradient_clip)

    
class Horseshoe(Normal):
    def __init__(self, shape, loc, scale, hyperscale=1., gradient_clip=1.):
        scale_prior = HalfCauchy(shape=[], scale=hyperscale, multiplyer=scale)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(1.))
        super().__init__(shape, loc, scale_prior)
        _ClipHookMixin._init(self, gradient_clip)
