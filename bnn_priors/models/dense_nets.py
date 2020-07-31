from .layers import Linear
from . import RegressionModel, RaoBRegressionModel
from .. import prior
from torch import nn, Tensor

__all__ = ('LinearNealNormal', 'DenseNet', 'RaoBDenseNet')

def LinearNealNormal(in_dim: int, out_dim: int, std_w: float, std_b: float) -> nn.Module:
    return Linear(prior.Normal((out_dim, in_dim), 0., std_w/in_dim**.5),
                  prior.Normal((out_dim,), 0., std_b))


def DenseNet(in_features: int, out_features: int, width: int, noise_std: float=1.) -> nn.Module:
    return RegressionModel(
        nn.Sequential(
            LinearNealNormal(in_features, width, 2**.5, 1.0),
            nn.ReLU(),
            LinearNealNormal(width, width, 2**.5, 1.0),
            nn.ReLU(),
            LinearNealNormal(width, out_features, 2**.5, 1.0)), noise_std)


def RaoBDenseNet(x_train: Tensor, y_train: Tensor, width: int, noise_std: float=1.) -> nn.Module:
    in_dim = x_train.size(-1)
    out_dim = y_train.size(-1)
    return RaoBRegressionModel(
        x_train, y_train, noise_std,
        last_layer_std=(2/width)**.5,
        net=nn.Sequential(
            LinearNealNormal(in_dim, width, 2**.5, 1.0),
            nn.ReLU(),
            LinearNealNormal(width, width, 2**.5, 1.0),
            nn.ReLU()))
