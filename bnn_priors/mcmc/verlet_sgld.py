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
    def delta_energy(self, prev_potential: float, potential: float) -> float:
        "Calculates the difference in energy since the last `initial_step` and now."
        num_data = self.param_groups[0]['num_data']
        assert all(g['num_data'] == num_data for g in self.param_groups),\
            "unclear which `num_data` to use"
        delta_energy = 0.
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                point_energy = self._point_energy(group, p, state)
                delta_energy += state['delta_energy'] + point_energy

        if isinstance(potential, torch.Tensor):
            potential = potential.item()
        delta_energy += (potential - prev_potential) * num_data
        return delta_energy

    def _point_energy(self, group, p, state):
        M_rsqrt = self._preconditioner_default(state, p)
        curv = M_rsqrt**2 * group['num_data']**2 * group['b^2h^2'] / 8
        return curv * dot(p.grad, p.grad)

    @torch.no_grad()
    def maybe_reject(self, delta_energy: float) -> (bool, float):
        "Maybe reject the current parameters, based on the difference in energy"
        temperature = self.param_groups[0]['temperature']
        assert all(g['temperature'] == temperature for g in self.param_groups),\
            "unclear which `temperature` to use"

        if temperature == 0.0:
            return False, 0.  # Never reject

        # rand() > min(1., exp(-delta_energy / temperature))
        log_accept_prob = -delta_energy / temperature
        reject = (math.log(torch.rand(()).item()) > log_accept_prob)
        if reject:
            for p, state in self.state.items():
                p.data.copy_(state['prev_parameter'])
                p.grad.copy_(state['prev_grad'])
                try:
                    state['momentum_buffer'].copy_(state['prev_momentum_buffer'])
                except KeyError:
                    pass
        return reject, log_accept_prob

    def _save_state(self, group, p, state):
        try:
            state['prev_parameter'].copy_(p)
            state['prev_grad'].copy_(p.grad)
            if group['momentum'] > 0:
                state['prev_momentum_buffer'].copy_(state['momentum_buffer'])
        except KeyError:
            state['prev_parameter'] = p.detach().to(device='cpu', copy=True)
            state['prev_grad'] = p.grad.detach().to(device='cpu', copy=True)
            if group['momentum'] > 0:
                state['prev_momentum_buffer'] = state['momentum_buffer'].to(
                    device='cpu', copy=True)

    @torch.no_grad()
    def initial_step(self, closure: Optional[Callable[..., torch.Tensor]]=None,
                     save_state=True, calc_metrics=True):
        """The initial transition for the Verlet integrator.
        θ(n), m(n) -> θ(n+1), u(n+1)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) - gradient of parameters
        """
        # keep a `torch.optim.lr_scheduler` happy
        self._step_count = getattr(self, '_step_count', 0) + 1
        def update_group_fn(g):
            self._update_group_fn(g)
            a = g['momentum']
            g['mom_decay'] = math.sqrt(a)
            g['grad_v'] = 1.
            g['noise_std'] = math.sqrt((1 - a) * g['temperature'])
        return self._step_internal(update_group_fn, self._step_fn, closure,
                                   is_initial=True, save_state=save_state,
                                   calc_metrics=calc_metrics)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[..., torch.Tensor]]=None,
             calc_metrics=True):
        """An intermediate transition for the Verlet integrator.
        θ(n), u(n) -> θ(n+1), u(n+1)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) - gradient of parameters
        """
        return self._step_internal(self._update_group_fn, self._step_fn,
                                   closure, calc_metrics=calc_metrics)

    @torch.no_grad()
    def final_step(self, closure: Optional[Callable[..., torch.Tensor]]=None,
                   calc_metrics=True):
        """The final transition for the Verlet integrator
        θ(n), u(n) -> θ(n), m(n)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) - gradient of parameters
        """
        # keep a `torch.optim.lr_scheduler` happy
        self._step_count = getattr(self, '_step_count', 0) + 1
        def update_group_fn(g):
            self._update_group_fn(g)
            a = g['momentum']
            g['mom_decay'] = math.sqrt(a)
            g['grad_v'] = g['mom_decay']
            g['noise_std'] = math.sqrt((1 - a) * g['temperature'])
        return self._step_internal(update_group_fn, self._step_fn, closure,
                                   is_final=True, calc_metrics=calc_metrics)

    def _update_group_fn(self, g):
        g['b^2h^2'] = g['lr'] / g['num_data']
        g['bh'] = math.sqrt(g['b^2h^2'])
        g['bhn'] = math.sqrt(g['lr'] * g['num_data'])

        a = g['momentum']
        g['mom_decay'] = a
        g['grad_v'] = 1 + a
        g['noise_std'] = math.sqrt((1 - a**2) * g['temperature'])


    def _step_fn(self, group, p, state, is_initial=False, is_final=False,
                 save_state=False, calc_metrics=True):
        """An intermediate transition for the Verlet integrator.
        θ(n), u(n) -> θ(n+1), u(n+1)

        u(n) is not the momentum, rather, it is
        u(n) = sqrt(b)*m(n) + dependent gaussian noise
        """
        if save_state:
            self._save_state(group, p, state)
        M_rsqrt = self._preconditioner_default(state, p)

        # Gradient step on the new_momentum
        old_momentum = state['momentum_buffer']
        new_momentum = torch.randn_like(p).mul_(group['noise_std'])
        grad_lr = -.5 * group['grad_v'] * group['bhn'] * M_rsqrt
        new_momentum.add_(p.grad, alpha=grad_lr)
        if group['mom_decay'] > 0:
            new_momentum.add_(old_momentum, alpha=group['mom_decay'])

        # Calculate this steps's contribution to the energy difference
        c_gm = -.5 * group['bhn'] * M_rsqrt
        if is_initial:
            state['delta_energy'] = -self._point_energy(group, p, state)
        else:
            state['delta_energy'] += state['prev_new_momentum_delta']
            state['delta_energy'] += c_gm * dot(p.grad, old_momentum)
        state['prev_new_momentum_delta'] = c_gm * dot(p.grad, new_momentum)

        if calc_metrics:
            # Temperature diagnostics
            d = p.numel()
            if is_final:
                # If it is the final step, p and p.grad correspond to the same
                # time step as `new_momentum`
                state['est_temperature'] = dot(new_momentum, new_momentum) / d
            else:
                # the momentum is from the previous time step
                state['est_temperature'] = dot(old_momentum, old_momentum) / d
            # NOTE: p and p.grad are (and have to be) from the same time step
            state['est_config_temp'] = dot(p, p.grad) * (group['num_data']/d)

        state['momentum_buffer'] = new_momentum
        if not is_final:
            p.add_(new_momentum, alpha=group['bh']*M_rsqrt)

            # RMSProp moving average
            alpha = group['rmsprop_alpha']
            state['square_avg'].mul_(alpha).addcmul_(p.grad, p.grad, value=1 - alpha)
