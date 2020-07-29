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
    @torch.no_grad()
    def initial_step(self, closure: Optional[Callable[..., torch.Tensor]]=None, save_state=True):
        """The initial transition for the Verlet integrator.
        θ(n), m(n) -> θ(n+1), u(n+1)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) - gradient of parameters
        """
        return self._step_internal(closure, is_initial=True, save_state=save_state)

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
        return self._step_internal(closure, is_final=True)

    def _update_group(self, g):
        g['a'] = g['momentum']
        g['b^2h^2'] = g['lr'] / g['num_data']
        g['bh'] = math.sqrt(g['b^2h^2'])
        g['bhn'] = math.sqrt(g['lr'] * g['num_data'])
        g['noise_std'] = math.sqrt((1 - g['a']**2) * g['temperature'])

    def _step_fn(self, group, p, state, gradient_time_step=1.0, is_initial=False, is_final=False, save_state=False):
        """An intermediate transition for the Verlet integrator.
        θ(n), u(n) -> θ(n+1), u(n+1)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) + dependent gaussian noise
        """
        M_rsqrt = self._preconditioner_default(state, p)

        # Parameters for initial or final or intermediate steps
        if is_initial:
            mom_decay = math.sqrt(group['a'])
            grad_v = 1.
            noise_std = math.sqrt((1 - group['a']) * group['temperature'])
        elif is_final:
            mom_decay = math.sqrt(group['a'])
            grad_v = mom_decay
            noise_std = math.sqrt((1 - group['a']) * group['temperature'])
        else:
            mom_decay = group['a']
            grad_v = 1 + group['a']
            noise_std = group['noise_std']

        # Gradient step on the momentum
        grad_lr = -.5 * grad_v * group['bhn'] * M_rsqrt
        if mom_decay > 0:
            momentum = state['momentum_buffer']
            if save_state:
                try:
                    state['prev_momentum_buffer'].copy_(momentum)
                except KeyError:
                    state['prev_momentum_buffer'] = momentum.to(device='cpu', copy=True)
            momentum.mul_(mom_decay).add_(p.grad, alpha=grad_lr)
        else:
            momentum = p.grad.detach().mul(grad_lr)

        # Add noise to the momentum
        if noise_std > 0:
            momentum.add_(torch.randn_like(p), alpha=noise_std)

        if save_state:
            try:
                state['prev_parameter'].copy_(p)
                state['prev_grad'].copy_(p.grad)
            except KeyError:
                state['prev_parameter'] = p.detach().to(device='cpu', copy=True)
                state['prev_grad'] = p.grad.detach().to(device='cpu', copy=True)
        # Update the parameters
        if not is_final:
            p.add_(momentum, alpha=group['bh']*M_rsqrt)

        # Temperature diagnostics
        d = p.numel()
        state['est_temperature'] = dot(momentum, momentum) / d
        state['est_config_temp'] = dot(p, p.grad) * (group['num_data']/d)

        # RMSProp moving average
        alpha = group['rmsprop_alpha']
        state['square_avg'].mul_(alpha).addcmul_(p.grad, p.grad, value=1 - alpha)


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

    def point_energy(self):
        kinetic_energy = 0.
        for p, state in self.state.items():
            mom = state['momentum_buffer']
            kinetic_energy += dot(mom, mom)
        return kinetic_energy / 2

    @torch.no_grad()
    def maybe_reject(self, delta_energy):
        num_data = self.param_groups[0]['num_data']

        reject = (torch.rand(()).item() > math.exp(-delta_energy))
        if reject:
            for p, state in self.state.items():
                p.data.copy_(state['prev_parameter'])
                p.grad.copy_(state['prev_grad'])
                state['momentum_buffer'].copy_(state['prev_momentum_buffer'])

        return reject




