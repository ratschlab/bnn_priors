import torch
import math
from scipy.stats import chi2
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
        raise_on_no_grad (bool): whether to complain if a parameter does not
                                 have a gradient
        raise_on_nan: whether to complain if a gradient is not all finite.
    """
    def __init__(self, params: Sequence[Union[torch.nn.Parameter, Dict]], lr: float,
                 num_data: int, momentum: float=0, temperature: float=1.,
                 raise_on_no_grad: bool=True, raise_on_nan: bool=False):
        assert lr >= 0 and num_data >= 0 and momentum >= 0 and temperature >= 0
        defaults = dict(lr=lr, num_data=num_data, momentum=momentum,
                        temperature=temperature)
        super(SGLD, self).__init__(params, defaults)
        self.raise_on_no_grad = raise_on_no_grad
        self.raise_on_nan = raise_on_nan

    def preconditioner(self, parameter):
        try:
            return self.state[parameter]['preconditioner']
        except KeyError:
            v = self.state[parameter]['preconditioner'] = 1.
            return v

    def momentum_buffer(self, parameter):
        try:
            return self.state[parameter]['momentum_buffer']
        except KeyError:
            return self._sample_momentum(parameter)

    def _sample_momentum(self, parameter):
        temperature = self.param_groups[0]['temperature']
        M = self.preconditioner(parameter)
        v = torch.randn_like(parameter).mul_(math.sqrt(M * temperature))
        self.state[parameter]['momentum_buffer'] = v
        return v

    def sample_momentum(self):
        for group in self.param_groups:
            for p in group['params']:
                self._sample_momentum(p)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[..., torch.Tensor]]=None,
             _momentum_step_multiplier=2):
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
                    c = math.sqrt(_momentum_step_multiplier*(1 - mom_decay) * temperature * M)
                    momentum.add_(torch.randn_like(momentum), alpha=c)

                # Take the gradient step
                p.add_(momentum, alpha=h/M)

                # Temperature diagnostics
                d = p.numel()
                state['est_temperature'] = dot(momentum, momentum) / (M*d)
                state['est_config_temp'] = dot(p, p.grad) * (num_data/d)

        return loss

    @torch.no_grad()
    def estimate_preconditioner(self, closure: Callable[..., torch.Tensor],
                                dataloader: Sequence[Tuple], eps: float=1e-7
                                ) -> typing.Dict[torch.nn.Parameter, float]:
        """Estimates the preconditioner for each parameter using the algorithm in
        Wenzel et al. 2020.

        Args:
            closure: calculates the minibatch loss, and stores its gradient
                      in the `.grad` attribute of the parameters.
            dataloader: a sequence of *args that represent minibatches, and are
                        used as arguments to the `closure`.
            eps: RMSProp regularization parameter

        Returns:
            precond: for each parameter (key), the resulting preconditioner (value).

        """
        precond = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                precond[p] = 0.

        K = len(dataloader)
        for args in dataloader:
            with torch.enable_grad():
                _loss = closure(args)

            for group in self.param_groups:
                for p in group['params']:
                    precond[p] += dot(p.grad, p.grad)

        min_s = math.inf
        for p, v in precond.items():
            precond[p] = math.sqrt(eps + v/(p.numel() * K))
            min_s = min(min_s, precond[p])

        for p, v in precond.items():
            state = self.state[p]
            old_M = state['preconditioner']
            new_M = v/min_s
            conversion = math.sqrt(new_M / old_M)
            state['momentum_buffer'].mul_(conversion)

            precond[p] = state['preconditioner'] = new_M
        return precond

    @torch.no_grad()
    def temperature_diagnostics(self, c=0.95):
        """Estimates the current temperature of each parameter
        and calculates their confidence interval.

        Args:
            c (float or ndarray ortensor): the target probability that the
                temperature is inside the confidence interval
        Returns:
            est_c (float): the empirical probability that the temperatures are
                           inside the confidence interval.
            d (dict): For each parameter (key), a tuple (temperature, lower_confidence, upper_confidence)
        """
        d = OrderedDict()
        est_c_sum = 0
        N = 0
        for group in self.param_groups:
            for p in group["params"]:
                df = p.numel()
                _a = group["temperature"] / df
                lower = _a * chi2.ppf((1-c)/2, df)
                upper = _a * chi2.ppf((1+c)/2, df)

                t = self.state[p]['est_temperature']

                est_c_sum += ((lower <= t) & (t <= upper))
                N += 1

                d[p] = (t, lower, upper)
        return est_c_sum/N, d
