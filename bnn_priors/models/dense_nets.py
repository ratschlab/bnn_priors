from .layers import Linear
from .regression import GaussianUnivariateRegression, RaoBGaussianUnivariateRegression
from .. import prior
from torch import nn

__all__ = ('LinearNealNormal', 'DenseNet', 'RaoBDenseNet')

def LinearNealNormal(in_dim, out_dim, std_w, std_b):
    return Linear(prior.Normal((out_dim, in_dim), 0., std_w/in_dim**.5),
                  prior.Normal((out_dim,), 0., std_b))


def DenseNet(x_train, y_train, width, noise_std=1.):
    in_dim = x_train.size(-1)
    out_dim = y_train.size(-1)
    num_data = x_train.size(0)
    return GaussianUnivariateRegression(
        num_data, noise_std, [
            LinearNealNormal(in_dim, width, 2**.5, 1.0),
            nn.ReLU(),
            LinearNealNormal(width, width, 2**.5, 1.0),
            nn.ReLU(),
            LinearNealNormal(width, out_dim, 2**.5, 1.0)])


def RaoBDenseNet(x_train, y_train, width, noise_std=1.):
    in_dim = x_train.size(-1)
    out_dim = y_train.size(-1)
    return RaoBGaussianUnivariateRegression(
        x_train, y_train, noise_std,
        last_layer_std=(2/width)**.5,
        latent_fn_modules=[
            LinearNealNormal(in_dim, width, 2**.5, 1.0),
            nn.ReLU(),
            LinearNealNormal(width, width, 2**.5, 1.0),
            nn.ReLU()])
