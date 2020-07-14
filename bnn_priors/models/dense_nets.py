from .layers import Linear
from . import RegressionModel, RaoBRegressionModel
from .. import prior
from torch import nn

__all__ = ('LinearNealNormal', 'DenseNet', 'RaoBDenseNet')

def LinearNealNormal(in_dim, out_dim, std_w, std_b):
    return Linear(prior.Normal((out_dim, in_dim), 0., std_w/in_dim**.5),
                  prior.Normal((out_dim,), 0., std_b))


def DenseNet(num_data, in_features, out_features, width, noise_std=1.):
    return RegressionModel(
        num_data, noise_std, nn.Sequential(
            LinearNealNormal(in_features, width, 2**.5, 1.0),
            nn.ReLU(),
            LinearNealNormal(width, width, 2**.5, 1.0),
            nn.ReLU(),
            LinearNealNormal(width, out_features, 2**.5, 1.0)))


def RaoBDenseNet(x_train, y_train, width, noise_std=1.):
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
