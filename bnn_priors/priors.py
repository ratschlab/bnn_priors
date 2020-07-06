import torch
from torch import distributions


import torch as t
import torch.nn as nn
from torch.distributions import Normal


def value_sample(vs):
    if isinstance(vs, t.Tensor):
        raise Exception("Should be scalar or Prior, not Tensor")
    elif isinstance(vs, Prior):
        return vs()
    else:
        return vs

class Prior(nn.Module):
    """
    The key class.
    _dist should be overwritten, and we can use PyTorch base distributions
    """
    pass
    def __init__(self, shape, *args):
        super().__init__()
        self.shape = shape
        self.args = args
        self.sample = nn.Parameter(self.dist().sample(sample_shape=self.shape))

        for idx, arg in enumerate(args):
            if isinstance(arg, nn.Module):
                # check that all modules are priors
                assert isinstance(arg, Prior)
                # register child modules
                self.add_module(str(idx), arg)

    def dist(self):
        return self._dist(*[value_sample(arg) for arg in self.args])

    def log_prob(self):
        return self.dist().log_prob(self.sample).sum()

    def forward(self):
        return self.sample

def prior_mod_dict(mod):
    """
    A generator for all the Prior Modules in the network
    """
    result = {}
    for k, v in mod.named_modules():
        if isinstance(v, Prior):
            result[k] = v
    return result

def prior_val_dict(mod):
    """
    A generator for the values of all prior modules
    """
    result = {}
    for k, v in mod.named_modules():
        if isinstance(v, Prior):
            result[k] = v()
    return result

def log_prob(mod):
    return sum(prior.log_prob() for prior in prior_mod_dict(mod).values())

def set_prior(prior_mod_dict, prior_val_dict):
    """
    Sets sample in all of prior_mod_dict to the value in prior_val_dict
    """
    for k in prior_mod_dict:
        assert k in prior_val_dict
    for k in prior_val_dict:
        assert k in prior_mod_dict

    for k in prior_mod_dict:
        prior_mod_dict[k].sample = prior_val_dict[k]


class NormalPrior(Prior):
    _dist = Normal

class NormalPriorLogScale(Prior):
    def dist(self):
        mean = value_sample(arg[0])
        log_scale = value_sample(arg[1])
        return Normal(mean, log_scale.exp())
    
a = NormalPrior((1,2,3), 0., 1.)
b = NormalPrior((1,2,3), a, 1.)
c = NormalPrior((1,2,3), b, 1.)

class Linear(nn.Linear):
    def __init__(self, weight_prior, bias_prior=None):
        nn.Module.__init__(self)
        (self.out_features, self.in_features) = weight_prior.sample.shape
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior

    @property
    def weight(self):
        return self.weight_prior()
    @property
    def bias(self):
        return None if self.bias_prior is None else self.bias_prior()

lin = Linear(NormalPrior((3, 4), 0., 1.))
lin(t.ones(3, 4))







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
