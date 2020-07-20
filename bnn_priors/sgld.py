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

    def _preconditioner_default(self, state, p) -> torch.Tensor:
        try:
            return state['preconditioner']
        except KeyError:
            v = state['preconditioner'] = torch.ones_like(p)
            return v

    @torch.no_grad()
    def sample_momentum(self):
        "Sample the momenta for all the parameters"
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                std = self._momentum_std(group, p, state)
                state['momentum_buffer'] = torch.randn_like(p).mul_(std)

    def _momentum_std(self, group, p, state) -> torch.Tensor:
        temperature = group['temperature']
        M = self._preconditioner_default(state, p)
        return M.sqrt().mul_(math.sqrt(temperature))


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

    def _step_fn(self, group, p, state):
        mom_decay = group['momentum']
        temperature = group['temperature']
        num_data = group['num_data']
        hn = math.sqrt(group['lr'] * num_data)
        h = math.sqrt(group['lr'] / num_data)

        # Update the momentum
        if mom_decay > 0:
            momentum = state['momentum_buffer']
            momentum.mul_(mom_decay).add_(p.grad, alpha=-hn)
        else:
            momentum = p.grad.detach().mul(-hn)

        M = state['preconditioner']

        # Add noise to momentum
        if temperature > 0:
            c = math.sqrt((1 - mom_decay) * temperature)
            momentum.addcmul_(torch.randn_like(momentum), M.sqrt(), value=c)

        # Take the gradient step
        p.addcdiv_(momentum, M, value=h)

        # Temperature diagnostics
        d = p.numel()
        state['est_temperature'] = dot(momentum, momentum/M) / d
        state['est_config_temp'] = dot(p, p.grad) * (num_data/d)

        # RMSProp
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

                precond[p] = square_avg + eps
                min_s = min(min_s, precond[p].mean())

        for p, new_M in precond.items():
            new_M.div_(min_s).pow_(self._preconditioner_pow)

            state = self.state[p]
            old_M = self._preconditioner_default(state, p)
            self._convert_momentum_buffer(state['momentum_buffer'], old_M, new_M)
            state['preconditioner'] = new_M

    _preconditioner_pow = 1/2
    def _convert_momentum_buffer(self, momentum_buffer, old_M, new_M):
        conversion = new_M.div(old_M).sqrt_()
        momentum_buffer.mul_(conversion)


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
                lower = chi2.ppf((1-c)/2, df) / df
                upper = chi2.ppf((1+c)/2, df) / df
                d[p] = (lower, upper)
        return d
