import os
import math
import numpy as np
import torch as t
from pathlib import Path
from bnn_priors.models import RaoBDenseNet, DenseNet
from bnn_priors.prior import LogNormal
import matplotlib.pyplot as plt

from pyro.infer.mcmc import NUTS, HMC
from pyro.infer.mcmc.api import MCMC

from bnn_priors.data import UCI


data = UCI("yacht", 0)

device = ('cuda' if t.cuda.is_available() else 'cpu')
x_train = data.norm.train_X
y_train = data.norm.train_y

x_test = data.norm.test_X
y_test = data.norm.test_y

#model = RaoBDenseNet(x_train, y_train, 50, noise_std=LogNormal((), -1., 0.2)).to(x_train)
model = DenseNet(x_train.shape[-1], y_train.shape[-1], 50, noise_std=LogNormal((), -1., 0.2)).to(x_train)

N_steps = 1000
warmup = 1000

#opt = t.optim.SGD(model.parameters(), lr=1E-3)
#
#for i in range(100000): 
#    opt.zero_grad()
#    loss = model.potential_avg(None, None, eff_num_data=x_train.shape[0])
#    loss.backward()
#    if i % 100 == 0:
#        print(loss.item())
#        print(model.noise_std())
#        print()
#    opt.step()

kernel = HMC(potential_fn=lambda p: model.get_potential(x_train, y_train, eff_num_data=1*x_train.shape[0])(p),
             adapt_step_size=False, adapt_mass_matrix=False,
             step_size=1E-4, num_steps=32)
mcmc = MCMC(kernel, num_samples=N_steps, warmup_steps=warmup, initial_params = model.params_with_prior_dict())
mcmc.run()

samples = mcmc.get_samples()

lps = t.zeros(N_steps, *y_test.shape)

for i in range(N_steps):
    sample = dict((k, v[i]) for k, v in samples.items())
    with t.no_grad(), model.using_params(sample):
        lps[i] = model(x_test).log_prob(y_test)

final_params = dict((k, v[-1]) for k, v in samples.items())
with t.no_grad(), model.using_params(sample):
    P = model(x_test)

lps = lps.logsumexp(0) - math.log(N_steps)
lp = lps.mean()
print(lp)

