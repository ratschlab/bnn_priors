import torch
import math
import torch.distributions as td

from . import distributions
from .base import Prior
from numbers import Number


__all__ = ('MultivariateT',)


class MultivariateT(Prior):
    def __init__(self, shape, loc, scale_tril, df=3, event_dim=None,
                 permute=None):
        if event_dim is None:
            event_dim = len(shape)
        if permute is None:
            permuted_shape = shape
            permute = list(range(len(shape)))
        else:
            permuted_shape = torch.Size([shape[i] for i in permute])

        assert event_dim >= 1
        out_event_shape = torch.Size(permuted_shape)[len(permuted_shape)-event_dim:]
        assert len(out_event_shape) == event_dim
        batch_shape = torch.Size(permuted_shape)[:len(permuted_shape)-event_dim]

        if isinstance(scale_tril, Number) or isinstance(loc, Number):
            scale_tril = torch.ones([1, 1]) * scale_tril
            loc = torch.zeros([1]) + loc

        # broadcast loc and scale_tril, check size of last dimension
        correlation_size = td.MultivariateNormal(loc, scale_tril=scale_tril).event_shape[-1]

        if correlation_size == 1:
            if out_event_shape[-1] == 1:
                event_shape = out_event_shape
            else:
                event_shape = torch.Size([*out_event_shape, 1])
        else:
            # put dimensions of out_event_shape together until they are as large
            # as the covariance matrix / mean, i.e. `correlation_size`
            size = 1
            coincides = False
            # iterate starting from the end
            for i, size_i in reversed(list(enumerate(out_event_shape))):
                size *= size_i
                if size == correlation_size:
                    coincides = True
                    last_idx = i
                    break
            assert coincides
            event_shape = torch.Size([*out_event_shape[:last_idx], correlation_size])

        super().__init__(shape=shape, loc=loc, scale_tril=scale_tril, df=df,
                         event_shape=event_shape, out_event_shape=out_event_shape,
                         permute=permute, batch_shape=batch_shape)

    def _dist(self, loc, scale_tril, df, event_shape, **_kwargs):
        return distributions.MultivariateT(
            df=df, loc=loc, scale_tril=scale_tril, event_shape=event_shape)

    def _sample_value(self, shape: torch.Size):
        dist = self._dist_obj()
        x = dist.sample(sample_shape=self.batch_shape)
        # Make sure the tensor that goes into the nn.Parameter is contiguous.
        return x.view(self.batch_shape + self.out_event_shape).permute(*self.permute).contiguous()

    def log_prob(self) -> torch.Tensor:
        # We do the reshaping here in the Prior class because the pytorch API
        # for Transforms does not easily allow for an input and output of the
        # transform with a different number of dimensions, or a different shape.
        p = self.p.permute(*self.permute).view(self.batch_shape + self.event_shape)
        # p is not contiguous, but log_prob typically does not care.
        return self._dist_obj().log_prob(p).sum()
