import torch
from torch import distributions


def normal_prior(loc=0., scale=1.):
    def prior(param):
        return distributions.Normal(torch.zeros_like(param) + loc,
                                    torch.ones_like(param) * scale)
    return prior


def laplace_prior(loc=0., scale=1.):
    def prior(param):
        return distributions.Laplace(torch.zeros_like(param) + loc,
                                    torch.ones_like(param) * scale)
    return prior


def cauchy_prior(loc=0., scale=1.):
    def prior(param):
        return distributions.Cauchy(torch.zeros_like(param) + loc,
                                    torch.ones_like(param) * scale)
    return prior


def uniform_prior(low=-1., high=1.):
    def prior(param):
        return distributions.Uniform(torch.ones_like(param) * low,
                                    torch.ones_like(param) * high)
    return prior


def student_t_prior(df=2., loc=0., scale=1.):
    def prior(param):
        return distributions.StudentT(df, torch.zeros_like(param) + loc,
                                    torch.ones_like(param) * scale)
    return prior
