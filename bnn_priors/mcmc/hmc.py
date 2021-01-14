import torch
import math
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
import typing

from .sgld import dot
from .verlet_sgld import VerletSGLD


class HMC(VerletSGLD):
    """HMC with Verlet integration. Really `VerletSGLD` but with momentum=1 and
    temperature=1, and a different M-H acceptance probability.

    The user should call `sample_momentum` regularly to refresh the HMC momentum.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        num_data (int): the number of data points in this learning task
        raise_on_no_grad (bool): whether to complain if a parameter does not
                                 have a gradient
        raise_on_nan: whether to complain if a gradient is not all finite.
    """
    def __init__(self, params: Sequence[Union[torch.nn.Parameter, Dict]],
                 lr: float, num_data: int,
                 raise_on_no_grad: bool=True, raise_on_nan: bool=True):
        super().__init__(params, lr, num_data, 1., 1.,
                         raise_on_no_grad=raise_on_no_grad,
                         raise_on_nan=raise_on_nan)

    def _point_energy(self, group, p, state):
        return .5 * dot(state['momentum_buffer'], state['momentum_buffer'])

    def _update_group_fn(self, g):
        # Ensure momentum and temperature are correct at every step
        # No matter what modifications are done before `self.step`.
        super()._update_group_fn(g)
        assert g['momentum'] == 1. and g['temperature'] == 1.

    def _step_fn(self, group, p, state, is_initial=False, is_final=False,
                 save_state=False, calc_metrics=True):
        if save_state:
            self._save_state(group, p, state)

        M_rsqrt = self._preconditioner_default(state, p)
        momentum = state['momentum_buffer']

        if is_initial:
            mom_dot = dot(momentum, momentum)
            # Subtract initial kinetic energy from delta_energy
            state['delta_energy'] = -.5 * mom_dot
            if calc_metrics:
                state['est_temperature'] = mom_dot / p.numel()

        if calc_metrics:
            # Temperature diagnostics
            d = p.numel()
            if not is_final and not is_initial:
                state['est_temperature'] = dot(momentum, momentum) / d
            # NOTE: p and p.grad are from the same time step
            state['est_config_temp'] = dot(p, p.grad) * (group['num_data']/d)

        # Gradient step on the momentum
        grad_lr = -.5 * group['grad_v'] * group['bhn'] * M_rsqrt
        momentum.add_(p.grad, alpha=grad_lr)

        if is_final:
            if calc_metrics:
                # If it is the final step, p and p.grad correspond to the same time
                # step as the updated momentum
                state['est_temperature'] = dot(momentum, momentum) / p.numel()
        else:
            # Update the parameters:
            p.add_(momentum, alpha=group['bh']*M_rsqrt)

            # RMSProp moving average
            alpha = group['rmsprop_alpha']
            state['square_avg'].mul_(alpha).addcmul_(p.grad, p.grad, value=1 - alpha)
