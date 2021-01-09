import torch
import math
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
                 rmsprop_alpha: float=0.99, rmsprop_eps: float=1e-8,  # Wenzel et al. use 1e-7
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
        self._step_count = 0  # keep the `torch.optim.scheduler` happy

    def _preconditioner_default(self, state, p) -> float:
        try:
            return state['preconditioner']
        except KeyError:
            v = state['preconditioner'] = 1.
            return v

    def delta_energy(self, a, b) -> float:
        return math.inf

    @torch.no_grad()
    def sample_momentum(self, keep=0.0):
        "Sample the momenta for all the parameters"
        assert 0 <= keep and keep <= 1.
        if keep == 1.:
            return
        for group in self.param_groups:
            std = math.sqrt(group['temperature']*(1-keep))
            for p in group['params']:
                if keep == 0.0:
                    self.state[p]['momentum_buffer'] = torch.randn_like(p).mul_(std)
                else:
                    self.state[p]['momentum_buffer'].mul_(math.sqrt(keep)).add_(torch.randn_like(p), alpha=std)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[..., torch.Tensor]]=None,
             calc_metrics=True, save_state=False):
        assert save_state is False
        return self._step_internal(self._update_group_fn, self._step_fn,
                                   closure, calc_metrics=calc_metrics)
    initial_step = step

    @torch.no_grad()
    def final_step(self, closure: Optional[Callable[..., torch.Tensor]]=None,
                   calc_metrics=True, save_state=False):
        assert save_state is False
        return self._step_internal(self._update_group_fn, self._step_fn,
                                   closure, calc_metrics=calc_metrics,
                                   is_final=True)


    def _step_internal(self, update_group_fn, step_fn, closure, **step_fn_kwargs):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        try:
            for group in self.param_groups:
                update_group_fn(group)
                for p in group['params']:
                    if p.grad is None:
                        if self.raise_on_no_grad:
                            raise RuntimeError(
                                f"No gradient for parameter with shape {p.shape}")
                        continue
                    if self.raise_on_nan and not torch.isfinite(p.grad).all():
                        raise ValueError(
                            f"Gradient of shape {p.shape} is not finite: {p.grad}")
                    step_fn(group, p, self.state[p], **step_fn_kwargs)

        except KeyError as e:
            if e.args[0] == "momentum_buffer":
                raise RuntimeError("No 'momentum_buffer' stored in state. "
                                   "Perhaps you forgot to call `sample_momentum`?")
            raise e
        return loss

    def _update_group_fn(self, g):
        g['hn'] = math.sqrt(g['lr'] * g['num_data'])
        g['h'] = math.sqrt(g['lr'] / g['num_data'])
        g['noise_std'] = math.sqrt(2*(1 - g['momentum']) * g['temperature'])

    def _step_fn(self, group, p, state, calc_metrics=True, is_final=False):
        """if is_final, do not change parameters or momentum"""
        M_rsqrt = self._preconditioner_default(state, p)
        d = p.numel()

        # Update the momentum with the gradient
        if group['momentum'] > 0:
            momentum = state['momentum_buffer']
            if calc_metrics:
                # NOTE: the momentum is from the previous time step
                state['est_temperature'] = dot(momentum, momentum) / d
            if not is_final:
                momentum.mul_(group['momentum']).add_(p.grad, alpha=-group['hn']*M_rsqrt)
        else:
            if not is_final:
                momentum = p.grad.detach().mul(-group['hn']*M_rsqrt)
            if calc_metrics:
                # TODO: make the momentum be from the previous time step
                state['est_temperature'] = dot(momentum, momentum) / d

        if not is_final:
            # Add noise to momentum
            if group['temperature'] > 0:
                momentum.add_(torch.randn_like(momentum), alpha=group['noise_std'])

        if calc_metrics:
            # NOTE: p and p.grad are from the same time step
            state['est_config_temp'] = dot(p, p.grad) * (group['num_data']/d)

        if not is_final:
            # Take the gradient step
            p.add_(momentum, alpha=group['h']*M_rsqrt)

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

                precond[p] = square_avg.mean().item() + eps
                min_s = min(min_s, precond[p])

        for p, new_M in precond.items():
            # ^(1/2) to form the preconditioner,
            # ^(-1/2) because we want the preconditioner's inverse square root.
            self.state[p]['preconditioner'] = (new_M / min_s)**(-1/4)
