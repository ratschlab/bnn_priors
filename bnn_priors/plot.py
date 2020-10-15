import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from typing import Optional, Union, Tuple, Dict



def get_sizes(model: torch.nn.Module) -> Dict[str, int]:
    return {k: p.numel() for k, p in model.state_dict().items()}


def weighted_var_se(w, x):
    """Computes the variance of a weighted mean following Cochran 1977.

    Adapted from a stats.SE answer copyright Ming K, licensed under CC BY-SA 3.0.
    https://stats.stackexchange.com/a/33959/172619
    """
    n, = w.shape
    assert x.shape[-1] == n
    xWbar = (x @ w) / w.sum()
    wbar = w.mean()

    w__wbar = w-wbar
    wx__wbar_xWbar = w*x - wbar * xWbar[..., None]

    se = n/((n-1) * w.sum()**2)*(
        (wx__wbar_xWbar ** 2).sum(-1)
        - 2*xWbar  * (wx__wbar_xWbar @ w__wbar)
        + xWbar**2 * (w__wbar @ w__wbar))

    return xWbar, se


def ewma(array, alpha):
    """
    Exponential weighted moving average with decay alpha.

    Copyright snooper77 and William Miller, licensed under CC BY-SA 4.0
    https://stackoverflow.com/a/59199643/4521118
    """
    if alpha == 0.0:
        return array
    b = [1-alpha]
    a = [1, -alpha]
    zi = scipy.signal.lfiltic(b, a, array[0:1], [0])
    return scipy.signal.lfilter(b, a, array, zi=zi)[0]


def temperature_stderr(ax, metrics, temp_group, sizes: Dict[str, int], ewma_alpha:float=0.0,
                       mask:Union[slice, np.ndarray]=slice(None), label=None, legend=True,
                       line_kwargs={}, confidence_kwargs={}):
    """Plots the weighted mean and standard error of the various parameters'
    temperatures"""
    temperatures = metrics[temp_group]
    if label is None:
        label = temp_group

    mean = temperatures['all'][mask]
    keys = list(temperatures.keys())
    keys.remove('all')

    temps = np.stack([temperatures[k][mask] for k in keys], axis=1)
    # Weights: number of elements of each parameter
    weights = np.array([sizes[k] for k in keys], dtype=float)

    _mean, var_se = weighted_var_se(weights, temps)

    dist = MultivariateNormal(
        torch.from_numpy(mean), DiagLazyTensor(torch.from_numpy(var_se)))

    steps = metrics["steps"][mask]
    line, *_ = ax.plot(steps, metrics["temperature"][mask], linestyle='--', **line_kwargs)
    gp_posterior(ax, torch.from_numpy(steps), dist, ewma_alpha=ewma_alpha, color=line.get_color(), label=label, **confidence_kwargs)

    if legend:
        ax.legend()


def _gamma_confidence(sizes: Dict[str, int], c: Union[float, np.ndarray]=0.95) -> Dict[
        str, Tuple[np.ndarray, np.ndarray]]:
    """Calculates the confidence intervals for the kinetic temperature of the
    momentum of each parameter. Assumes the target temperature is 1.

    Arguments:
        c: the target confidence levels for the intervals
    Returns:
        "parameter -> (lower, upper)" dictionary of confidence intervals
        per parameters. `lower` and `upper` have the same shape as `c`.
    """
    d = {}
    for k, df in sizes.items():
        lower = scipy.stats.chi2.ppf((1-c)/2, df=df, scale=1/df)
        upper = scipy.stats.chi2.ppf((1+c)/2, df=df, scale=1/df)
        d[k] = (lower, upper)
    return d

def kinetic_temperature_intervals(
        ax, metrics, sizes: Dict[str, int], mask:Union[slice, np.ndarray]=slice(None),
        ewma_alpha:float=0.0, confidences=[0.05, 0.25, 0.50, 0.75, 0.95],
        label="confidence", legend=True, cmap=None, hline_kwargs={}, plot_kwargs={}):

    confidences = np.array(confidences)
    intervals = _gamma_confidence(sizes, confidences)
    temperature = metrics["temperature"][mask]

    keys = list(metrics["est_temperature"].keys())
    keys.remove("all")

    counts = None
    for k in keys:
        adjusted_temp = (metrics["est_temperature"][k][mask] / temperature)
        lower, upper = intervals[k]
        within_mask = np.logical_and(lower[:, None] <= adjusted_temp, adjusted_temp <= upper[:, None])
        if counts is None:
            counts = within_mask.astype(float)
        else:
            counts += within_mask.astype(float)
    counts /= len(keys)

    if cmap is None:
        cmap = plt.get_cmap('plasma')
    colors = cmap(confidences)

    steps = metrics["steps"][mask]
    for confidence, count, color in zip(confidences, counts, colors):
        line = ax.axhline(confidence, linestyle="--", linewidth=0.5, color=color, **hline_kwargs)
        count = ewma(count, ewma_alpha)
        kwargs = dict(linestyle="-", color=line.get_color(),
                       label=f"{label} {confidence:.2f}")
        kwargs.update(plot_kwargs)
        ax.plot(steps, count, **kwargs)

    if legend:
        ax.legend()


def metric(ax, metrics, name, mask:Union[slice, np.ndarray]=slice(None),
           ewma_alpha:float=0.0,
           legend=True, iqr_ylim=None, transform=(lambda x: x),
           plot_kwargs={}):
    val = ewma(transform(metrics[name][mask]), ewma_alpha)
    kwargs = dict(label=name)
    kwargs.update(plot_kwargs)
    ax.plot(metrics["steps"][mask], val, **kwargs)
    if legend:
        ax.legend()
    if iqr_ylim is not None:
        all_min, q25, median, q75, all_max = np.nanpercentile(val, (0, 25, 50, 75, 100))
        iqr = q75 - q25
        lower = max(median - iqr_ylim*iqr, all_min-0.05*iqr)
        upper = min(median + iqr_ylim*iqr, all_max+0.05*iqr)
        ax.set_ylim((lower, upper))


def vlines(ax, metrics, mask, plot_kwargs={}):
    x_vlines = metrics["steps"][mask]
    vlines = np.zeros((len(x_vlines), 2, 2))
    vlines[:, :, 0] = x_vlines[:, None]
    vlines[:, 0, 1] = 0.  # ymin
    vlines[:, 1, 1] = 1.  # ymax
    trans = ax.get_xaxis_transform(which='grid')
    kwargs = dict(color="red", linestyle="--", transform=trans)
    kwargs.update(plot_kwargs)
    ax.add_collection(LineCollection(vlines, **kwargs))


def n(t):
    try:
        return t.cpu().detach().numpy()
    except AttributeError:
        pass
    return t


def gp_posterior(ax, x: torch.Tensor, preds: MultivariateNormal,
                 ewma_alpha:float=0.0, label:Optional[str]=None, sort=True,
                 fill_alpha=0.05, **kwargs):
    x = x.view(-1)
    if sort:
        # i = x.argsort(dim=-2)[:, 0]
        i = x.argsort()
        if i.equal(torch.arange(i.size(0))):
            i = slice(None, None, None)
    else:
        i = slice(None, None, None)
    x = n(x[i])

    preds_mean = preds.mean.view(-1)
    mean = ewma(n(preds_mean[i]), ewma_alpha)
    line, *_ = ax.plot(x, mean, **kwargs)
    if label is not None:
        line.set_label(label)

    C = line.get_color()
    lower, upper = (p.view(-1) for p in preds.confidence_region())

    lower = ewma(n(lower[i]), ewma_alpha)
    upper = ewma(n(upper[i]), ewma_alpha)
    ax.fill_between(x, lower, upper, alpha=fill_alpha, color=C)
    ax.plot(x, lower, color=C, linewidth=0.5)
    ax.plot(x, upper, color=C, linewidth=0.5)
