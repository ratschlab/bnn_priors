import torch
import math
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
import typing

from .sgld import SGLD, dot

class VerletSGLD(SGLD):
    """SGLD with momentum, preconditioning and diagnostics from Wenzel et al. 2020.
    Uses Verlet integration instead of Euler symplectic integration.

    The contribution from the Verlet integration to the acceptance probability
    is neutral (multiply by 1), because it is perfectly time-reversible.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        num_data (int): the number of data points in this learning task
        momentum (float): momentum factor (default: 0)
        temperature (float): Temperature for tempering the posterior.
                             temperature=0 corresponds to SGD with momentum.
        raise_on_no_grad (bool): whether to complain if a parameter does not
                                 have a gradient
        raise_on_nan: whether to complain if a gradient is not all finite.
    """
    _preconditioner_pow = -1/4
    def _convert_momentum_buffer(self, momentum_buffer, old_M, new_M):
        pass  # No need in this class

    def _momentum_std(self, group, state, p) -> float:
        temperature = group['temperature']
        mom_decay = group['momentum']
        b = 2/(3 - mom_decay)
        # a = (1 + mom_decay)/(3 - mom_decay)
        # gamma = (1 - mom_decay)*math.sqrt(group['num_data'] / group['lr'])
        return math.sqrt(temperature * b)

    @torch.no_grad()
    def initial_step(self, closure: Optional[Callable[..., torch.Tensor]]=None):
        """The initial transition for the Verlet integrator.
        θ(n), m(n) -> θ(n+1), u(n+1)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) - gradient of parameters
        """
        return self._step_internal(closure, gradient_time_step=0.5)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[..., torch.Tensor]]=None):
        """An intermediate transition for the Verlet integrator.
        θ(n), u(n) -> θ(n+1), u(n+1)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) - gradient of parameters
        """
        return self._step_internal(closure)

    @torch.no_grad()
    def final_step(self, closure: Optional[Callable[..., torch.Tensor]]=None):
        """The final transition for the Verlet integrator
        θ(n), u(n) -> θ(n), m(n)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) - gradient of parameters
        """
        return self._step_internal(closure, gradient_time_step=0.5, is_final=True)

    def _step_fn(self, group, p, state, gradient_time_step=1.0, is_final=False):
        """An intermediate transition for the Verlet integrator.
        θ(n), u(n) -> θ(n+1), u(n+1)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) + dependent gaussian noise
        """
        mom_decay = group['momentum']
        temperature = group['temperature']
        num_data = group['num_data']

        hn = math.sqrt(group['lr'] * num_data)
        h = math.sqrt(group['lr'] / num_data)
        gamma = (1 - mom_decay)*math.sqrt(num_data / group['lr'])
        b = 2/(3 - mom_decay)
        a = (1 + mom_decay)/(3 - mom_decay)
        noise_std = math.sqrt(temperature*gamma/2)

        # Calculate m̄^(n+1/2)
        momentum = state['momentum_buffer']
        M_sqrt_repr = state['preconditioner']
        momentum.addcmul_(M_sqrt_repr, p.grad, value=-hn*gradient_time_step)

        if is_final:
            return  # Only take a half-gradient step on the momentum in the final step

        if noise_std > 0:
            noise = torch.randn_like(p).mul_(noise_std)
            momentum.add_(noise)

        # Update parameter
        p.addcmul_(momentum, M_sqrt_repr, value=b*h)

        # Temperature diagnostics
        d = p.numel()
        state['est_temperature'] = dot(momentum, momentum) * (b/d)
        state['est_config_temp'] = dot(p, p.grad) * (num_data/d)

        # RMSProp
        alpha = group['rmsprop_alpha']
        state['square_avg'].mul_(alpha).addcmul_(p.grad, p.grad, value=1 - alpha)

        # Second half step for the momentum
        if noise_std > 0:
            state['momentum_buffer'] = noise.add_(momentum, value=a)
        else:
            momentum.mul_(a)


class HMC(VerletSGLD):
    """HMC with Verlet integration. Really `VerletSGLD` but with momentum=1 and
    temperature=1.

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
        super().__init__(params, lr, num_data, 1., 1., raise_on_no_grad,
                         raise_on_nan)
