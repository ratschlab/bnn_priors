import torch
import math
from scipy.stats import chi2
import numpy as np
from collections import OrderedDict
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
import typing


def dot(a, b):
    "return (a*b).sum().item()"
    return (a.view(-1) @ b.view(-1)).item()


class SGLD(torch.optim.Optimizer):
    """SGLD with momentum, preconditioning and diagnostics from Wenzel et al. 2020.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        num_data (int): the number of data points in this learning task
        momentum (float): momentum factor (default: 0)
        temperature (float): Temperature for tempering the posterior.
                             temperature=0 corresponds to SGD with momentum.
        rmsprop_alpha: decay for the moving average of the squared gradients
        rmsprop_eps: the regularizer parameter for the RMSProp update
        raise_on_no_grad (bool): whether to complain if a parameter does not
                                 have a gradient
        raise_on_nan: whether to complain if a gradient is not all finite.
    """
    def __init__(self, params: Sequence[Union[torch.nn.Parameter, Dict]], lr: float,
                 num_data: int, momentum: float=0, temperature: float=1.,
                 rmsprop_alpha: float=0.99, rmsprop_eps: float=1e-8,
                 raise_on_no_grad: bool=True, raise_on_nan: bool=False):
        assert lr >= 0 and num_data >= 0 and momentum >= 0 and temperature >= 0
        defaults = dict(lr=lr, num_data=num_data, momentum=momentum,
                        rmsprop_alpha=rmsprop_alpha, rmsprop_eps=rmsprop_eps,
                        temperature=temperature)
        super(SGLD, self).__init__(params, defaults)
        self.raise_on_no_grad = raise_on_no_grad
        self.raise_on_nan = raise_on_nan
        # OK to call this one, but not `sample_momentum`, because
        # `update_preconditioner` uses no random numbers.
        self.update_preconditioner()

    def _preconditioner_default(self, state, p) -> float:
        try:
            return state['preconditioner']
        except KeyError:
            v = state['preconditioner'] = 1.
            return v

    @torch.no_grad()
    def sample_momentum(self):
        "Sample the momenta for all the parameters"
        for group in self.param_groups:
            std = math.sqrt(group['temperature'])
            for p in group['params']:
                self.state[p]['momentum_buffer'] = torch.randn_like(p).mul_(std)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[..., torch.Tensor]]=None):
        return self._step_internal(closure)

    def _step_internal(self, closure, **step_fn_kwargs):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        try:
            for group in self.param_groups:
                self._update_group(group)
                for p in group['params']:
                    if p.grad is None:
                        if self.raise_on_no_grad:
                            raise RuntimeError(
                                f"No gradient for parameter with shape {p.shape}")
                        continue
                    if self.raise_on_nan and not torch.isfinite(p.grad).all():
                        raise ValueError(
                            f"Gradient of shape {p.shape} is not finite: {p.grad}")
                    self._step_fn(group, p, self.state[p], **step_fn_kwargs)

        except KeyError as e:
            if e.args[0] == "momentum_buffer":
                raise RuntimeError("No 'momentum_buffer' stored in state. "
                                   "Perhaps you forgot to call `sample_momentum`?")
            raise e
        return loss

    def _update_group(self, g):
        g['hn'] = math.sqrt(g['lr'] * g['num_data'])
        g['h'] = math.sqrt(g['lr'] / g['num_data'])
        g['noise_std'] = math.sqrt(2*(1 - g['momentum']) * g['temperature'])

    def _step_fn(self, group, p, state):
        M_rsqrt = self._preconditioner_default(state, p)

        # Update the momentum with the gradient
        if group['momentum'] > 0:
            momentum = state['momentum_buffer']
            momentum.mul_(group['momentum']).add_(p.grad, alpha=-group['hn']*M_rsqrt)
        else:
            momentum = p.grad.detach().mul(-group['hn']*M_rsqrt)

        # Add noise to momentum
        if group['temperature'] > 0:
            momentum.add_(torch.randn_like(momentum), alpha=group['noise_std'])

        # Take the gradient step
        p.add_(momentum, alpha=group['h']*M_rsqrt)

        # Temperature diagnostics
        d = p.numel()
        state['est_temperature'] = dot(momentum, momentum) / d
        state['est_config_temp'] = dot(p, p.grad) * (group['num_data']/d)

        # RMSProp moving average
        alpha = group['rmsprop_alpha']
        state['square_avg'].mul_(alpha).addcmul_(p.grad, p.grad, value=1 - alpha)

    @torch.no_grad()
    def update_preconditioner(self):
        """Updates the preconditioner for each parameter `state['preconditioner']` using
        the estimated `state['square_avg']`.
        """
        precond = OrderedDict()
        min_s = math.inf

        for group in self.param_groups:
            eps = group['rmsprop_eps']
            for p in group['params']:
                state = self.state[p]
                try:
                    square_avg = state['square_avg']
                except KeyError:
                    square_avg = state['square_avg'] = torch.ones_like(p)

                precond[p] = square_avg.mean() + eps
                min_s = min(min_s, precond[p])

        for p, new_M in precond.items():
            self.state[p]['preconditioner'] = new_M**(-1/4)

    def kinetic_temperature_intervals(self, c: Union[float, np.ndarray]=0.95) -> Dict[
            torch.nn.Parameter, Tuple[np.ndarray, np.ndarray]]:
        """Calculates the confidence intervals for the kinetic temperature of the
        momentum of each parameter. Assumes the target temperature is 1.

        Arguments:
            c: the target confidence levels for the intervals
        Returns:
            "parameter -> (lower, upper)" dictionary of confidence intervals
            per parameters. `lower` and `upper` have the same shape as `c`.
        """
        d = OrderedDict()
        for group in self.param_groups:
            for p in group["params"]:
                df = p.numel()
                lower = chi2.ppf((1-c)/2, df=df, scale=1/df)
                upper = chi2.ppf((1+c)/2, df=df, scale=1/df)
                d[p] = (lower, upper)
        return d
