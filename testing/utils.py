import functools
import torch


def requires_float64(fn):
    @functools.wraps(fn)
    def float64_fn(*args, **kwargs):
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_default_dtype(dtype)
    return float64_fn
