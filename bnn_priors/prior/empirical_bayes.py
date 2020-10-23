import torch.distributions as td
import torch
import math
from gpytorch.utils.transforms import inv_softplus

from .base import Prior
from .loc_scale import Normal, Laplace, StudentT, GenNorm, PositiveImproper, ConvCorrelatedNormal


__all__ = ('NormalEmpirical', 'LaplaceEmpirical', 'StudentTEmpirical',
           'GenNormEmpirical', 'ConvCorrNormalEmpirical')


class ConvCorrNormalEmpirical(ConvCorrelatedNormal):
    def __init__(self, shape, loc, scale, lengthscale=1.0):
        lengthscale_prior = PositiveImproper(shape=[], loc=lengthscale, scale=1.)
        scale_prior = PositiveImproper(shape=[], loc=scale, scale=1.)
        with torch.no_grad():
            lengthscale_prior.p.data = inv_softplus(torch.tensor(lengthscale))
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale=scale_prior, lengthscale=lengthscale_prior)


class NormalEmpirical(Normal):
    def __init__(self, shape, loc, scale):
        scale_prior = PositiveImproper(shape=[], loc=scale, scale=1.)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale_prior)


class LaplaceEmpirical(Laplace):
    def __init__(self, shape, loc, scale):
        scale_prior = PositiveImproper(shape=[], loc=scale, scale=1.)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
        super().__init__(shape, loc, scale_prior)
        

class StudentTEmpirical(StudentT):
    def __init__(self, shape, loc, scale, df=2.):
        scale_prior = PositiveImproper(shape=[], loc=scale, scale=1.)
        df_prior = PositiveImproper(shape=[], loc=df, scale=1.)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
            df_prior.p.data = inv_softplus(torch.tensor(df))
        super().__init__(shape, loc, scale=scale_prior, df=df_prior)
        

class GenNormEmpirical(GenNorm):
    def __init__(self, shape, loc, scale, beta=0.5):
        scale_prior = PositiveImproper(shape=[], loc=scale, scale=1.)
        beta_prior = PositiveImproper(shape=[], loc=beta, scale=1.)
        with torch.no_grad():
            scale_prior.p.data = inv_softplus(torch.tensor(scale))
            beta_prior.p.data = inv_softplus(torch.tensor(beta))
        super().__init__(shape, loc, scale=scale_prior, beta=beta_prior)
