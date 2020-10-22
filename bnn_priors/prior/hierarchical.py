import torch.distributions as td
import torch
import math
from gpytorch.utils.transforms import inv_softplus

from .base import Prior
from .loc_scale import Normal, Laplace, StudentT, GenNorm, ConvCorrelatedNormal
from .transformed import Gamma, Uniform, HalfCauchy


__all__ = ('NormalGamma', 'NormalUniform', 'Horseshoe', 'LaplaceGamma',
          'LaplaceUniform', 'StudentTGamma', 'StudentTUniform',
           'GenNormUniform', 'ConvCorrNormalGamma')


class NormalGamma(Normal):
    def __init__(self, shape, loc, scale, rate=1., gradient_clip=1.):
        scale_prior = Gamma(shape=[], concentration=scale, rate=rate)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale_prior)


class NormalUniform(Normal):
    def __init__(self, shape, loc, scale, gradient_clip=1.):
        scale_prior = Uniform(shape=[], low=0., high=scale*2.)
        with torch.no_grad():
            scale_prior.p.data = torch.tensor(0.)
        super().__init__(shape, loc, scale_prior)
        
        
class ConvCorrNormalGamma(ConvCorrelatedNormal):
    def __init__(self, shape, loc, scale, lengthscale=1., rate=1.):
        lengthscale_prior = Gamma(shape=[], concentration=lengthscale, rate=rate)
        scale_prior = Gamma(shape=[], concentration=scale, rate=rate)
        with torch.no_grad():
            lengthscale_prior.p.data = inv_softplus(torch.tensor(lengthscale))
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale=scale_prior, lengthscale=lengthscale_prior)

    
class LaplaceGamma(Laplace):
    def __init__(self, shape, loc, scale, rate=1., gradient_clip=1.):
        scale_prior = Gamma(shape=[], concentration=scale, rate=rate)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale_prior)

    
class LaplaceUniform(Laplace):
    def __init__(self, shape, loc, scale, gradient_clip=1.):
        scale_prior = Uniform(shape=[], low=0., high=scale*2.)
        with torch.no_grad():
            scale_prior.p.data = torch.tensor(0.)
        super().__init__(shape, loc, scale_prior)


class StudentTGamma(StudentT):
    def __init__(self, shape, loc, scale, rate=1., df=2, gradient_clip=1.):
        scale_prior = Gamma(shape=[], concentration=scale, rate=rate)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale_prior, df=df)

    
class StudentTUniform(StudentT):
    def __init__(self, shape, loc, scale, df=2, gradient_clip=1.):
        scale_prior = Uniform(shape=[], low=0., high=scale*2.)
        with torch.no_grad():
            scale_prior.p.data = torch.tensor(0.)
        super().__init__(shape, loc, scale_prior, df=df)
        
        
class GenNormUniform(GenNorm):
    def __init__(self, shape, loc, scale, beta=1., gradient_clip=1.):
        beta_prior = Uniform(shape=[], low=0., high=beta*2.)
        with torch.no_grad():
            beta_prior.p.data = torch.tensor(0.)
        super().__init__(shape, loc, scale, beta=beta_prior)

    
class Horseshoe(Normal):
    def __init__(self, shape, loc, scale, hyperscale=1., gradient_clip=1.):
        scale_prior = HalfCauchy(shape=[], scale=hyperscale, multiplier=scale)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(1.))
        super().__init__(shape, loc, scale_prior)
