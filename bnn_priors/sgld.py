import torch
import math
from scipy.stats import chi2
from collections import OrderedDict


class SGLD(torch.optim.Optimizer):
    """Implements SGLD with momentum and diagnostics from Wenzel et al. 2020.

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

    """
    def __init__(self, params, lr, num_data, momentum=0, temperature=1.,
                 raise_on_no_grad=True, raise_on_nan=True):
        assert lr >= 0 and num_data >= 0 and momentum >= 0 and temperature >= 0
        defaults = dict(lr=lr, num_data=num_data, momentum=momentum,
                        temperature=temperature)
        super(SGLD, self).__init__(params, defaults)
        self.raise_on_no_grad = raise_on_no_grad
        self.raise_on_nan = raise_on_nan

    @torch.no_grad()
    def step(self, closure=None):
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
                    if 'momentum_buffer' in state:
                        momentum = state['momentum_buffer']
                        momentum.mul_(mom_decay).add_(p.grad, alpha=-hn)
                    else:
                        momentum = state['momentum_buffer'] = (
                            p.grad.detach().mul(-hn))
                else:
                    momentum = p.grad.detach().mul(-hn)

                if 'preconditioner' in state:
                    M = state['preconditioner']
                else:
                    M = state['preconditioner'] = 1.

                # Add noise to momentum
                if temperature > 0:
                    c = math.sqrt(2*(1 - mom_decay) * temperature * M)
                    momentum.add_(torch.randn_like(momentum), alpha=c)

                # Take the gradient step
                p.add_(momentum, alpha=h/M)

                # Temperature diagnostics
                _m = momentum.view(-1)
                d = _m.size(0)
                state['est_temperature'] = (_m @ _m).item() / (M*d)
                state['est_config_temp'] = (
                    (p.view(-1) @ p.grad.view(-1)).item() * (num_data/d))

        return loss

    @torch.no_grad()
    def estimate_preconditioner(self, closure, K=32, eps=1e-7):
        """Estimates the preconditioner for each parameter using the algorithm in
        Wenzel et al. 2020.

        Args:
            closure (int -> IO float): calculates the k-th minibatch gradient.
                                       Should use a different minibatch every
                                       time it is called.
            K (int): the number of minibatches to use for estimating.
            eps (float): RMSProp regularization parameter

        Returns:
            precond (dict): for each parameter (key), the resulting
                            preconditioner (value).
        """
        precond = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                precond[p] = 0.

        for k in range(K):
            with torch.enable_grad():
                # TODO: what is closure supposed to do exactly? Because we don't use its output actually
                closure(k)

            for group in self.param_groups:
                for p in group['params']:
                    precond[p] += (p.grad.view(-1) @ p.grad.view(-1)).item()

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
