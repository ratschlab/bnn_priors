import torch
import math
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
import typing

from .sgld import SGLD

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
    @torch.no_grad()
    def initial_step(self, closure: Optional[Callable[..., torch.Tensor]]=None):
        """Verlet integration consists of:
        - half a step for the momentum
        - one full step for the parameters

        - Thereafter, run steps of symplectic Euler integration
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            mom_decay = group['momentum']
            temperature = group['temperature']
            num_data = group['num_data']
            hn = math.sqrt(group['lr'] * num_data)
            h = math.sqrt(group['lr'] / num_data)

            for p in group['params']:
                if p.grad is None:
                    if self.raise_on_no_grad:
                        raise RuntimeError(
                            f"No gradient for parameter with shape {p.shape}")
                    continue

                if self.raise_on_nan and not torch.isfinite(p.grad).all():
                    raise ValueError(
                        f"Gradient of shape {p.shape} is not finite: {p.grad}")

                # Update the momentum
                state = self.state[p]
                if mom_decay > 0:
                    momentum = self.momentum_buffer(p)
                    momentum.mul_(mom_decay).add_(p.grad, alpha=-hn)
                else:
                    momentum = p.grad.detach().mul(-hn)

                M = self.preconditioner(p)

                # Add noise to momentum
                if temperature > 0:
                    c = math.sqrt(2*(1 - mom_decay) * temperature * M)
                    momentum.add_(torch.randn_like(momentum), alpha=c)

                # Take the gradient step
                p.add_(momentum, alpha=h/M)

                # Temperature diagnostics
                d = p.numel()
                state['est_temperature'] = dot(momentum, momentum) / (M*d)
                state['est_config_temp'] = dot(p, p.grad) * (num_data/d)

        return loss
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            h = math.sqrt(group['lr'] / group['num_data'])
            for p in group['params']:
                momentum = self.momentum_buffer(p)
                M = self.preconditioner(p)

                # Half-step on the parameters
                p.add_(momentum, alpha=h/(2*M))

        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            mom_decay = group['momentum']
            temperature = group['temperature']
            num_data = group['num_data']
            hn = math.sqrt(group['lr'] * num_data)
            h = math.sqrt(group['lr'] / num_data)

            for p in group['params']:
                # Update the momentum
                state = self.state[p]
                momentum = state['momentum_buffer']
                M = state['preconditioner']

                if mom_decay > 0:
                    momentum.mul_(mom_decay).add_(p.grad, alpha=-hn)
                else:
                    torch.mul(p.grad.detach(), -hn, out=momentum)

                # Add noise to momentum
                if temperature > 0:
                    c = math.sqrt(2*(1 - mom_decay) * temperature * M)
                    momentum.add_(torch.randn_like(momentum), alpha=c)

                if step_i < num_steps-1:
                    # Full step on the parameters
                    p.add_(momentum, alpha=h/M)
                else:
                    # Half-step on the parameters in the final iteration
                    p.add_(momentum, alpha=h/(2*M))

                    # Temperature diagnostics
                    d = p.numel()
                    state['est_temperature'] = dot(momentum, momentum) / (M*d)
                    state['est_config_temp'] = dot(p, p.grad) * (num_data/d)
        return loss


class HMC(VerletSGLD):
    """HMC with Verlet integration. Really `VerletSGLD` but with momentum=1 and
    temperature=0.

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
        super().__init__(params, lr, num_data, 1., 0., raise_on_no_grad,
                         raise_on_nan)
