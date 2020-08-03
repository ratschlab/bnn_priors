import torch
from torch import nn
import unittest
from torch.utils.data import DataLoader, TensorDataset

from bnn_priors import eff_dim

from .utils import requires_float64

def _assert_eigvecs_allclose(ref_vecs, vecs, **kwargs):
    "Eigenvectors can have flipped sign"
    for i in range(ref_vecs.shape[1]):
        assert (torch.allclose(ref_vecs[:, i], vecs[:, i], **kwargs)
                or torch.allclose(ref_vecs[:, i], -vecs[:, i], **kwargs))


class EffDimTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(832173)

    def test_hessian(self, Da=5, Db=4):
        "Compare `eff_dim.hessian` with manual calculation"
        A = torch.randn(Da, Da)
        dataloader = [(torch.randn(Db, Db),) for _ in range(3)]

        parameters = [torch.randn(Da, requires_grad=True),
                      torch.randn(Db, requires_grad=True)]
        def fn(B):
            v, w = parameters
            return v@A@v + w@B@w

        hess = eff_dim.hessian(parameters, fn, dataloader)

        assert torch.allclose(hess[:Da, :Da], (A+A.t()) * len(dataloader))

        assert torch.allclose(hess[Da:, Da:], sum(B+B.t() for (B,) in dataloader))
        assert torch.equal(hess[:Da, Da:], torch.zeros((Da, Db)))
        assert torch.equal(hess[Da:, :Da], torch.zeros((Db, Da)))

    def example_model_loss_data(self, D=4, N=10, linear_only=False):
        if linear_only:
            model = nn.Linear(D, 1, bias=True)
        else:
            model = nn.Sequential(
                nn.Linear(D, D, bias=True),
                nn.Tanh(),
                nn.Linear(D, 1, bias=True))
        model.requires_grad_(True)

        def squared_error(x, y):
            f = model(x)
            v = (f - y).squeeze(1)
            return v @ v

        loader = DataLoader(TensorDataset(torch.randn(N, D), torch.randn(N, 1)),
                            batch_size=5)
        return model, squared_error, loader

    def test_hess_vec_prod(self):
        """Compare `eff_dim.hess_vec_prod` with first computing the hessian matrix,
        then the HVP"""
        model, fn, loader = self.example_model_loss_data()

        hess = eff_dim.hessian(model.parameters(), fn, loader)
        vec_flat = torch.randn(hess.size(1))
        vec_all = eff_dim.unflatten_like(vec_flat, model.parameters())

        # Store the hvp in [p.grad for p in model.parameters()]
        eff_dim.hess_vec_prod(vec_all, model.parameters(), fn, loader)

        hvp_flat = hess @ vec_flat
        hvp_all = eff_dim.unflatten_like(hvp_flat, model.parameters())

        for param, hvp in zip(model.parameters(), hvp_all):
            assert torch.allclose(param.grad, hvp)

    @requires_float64
    def test_symeig_positive_lanczos(self, D=8):
        "Compare `eff_dim.symeig_lanczos` with `torch.symeig`"
        M = torch.randn(D, D)
        M = M@M.t()
        ref_vals, ref_vecs = torch.symeig(M, eigenvectors=True)

        # All values/vectors:
        vals, vecs = eff_dim.symeig_positive_lanczos(M, n_eigs=-1, vecs=True)
        assert torch.allclose(ref_vals, vals, rtol=1e-4)
        _assert_eigvecs_allclose(ref_vecs, vecs, rtol=1e-3)

        # Leading values/vectors: not exact, no test
        # E = 4
        # vals, vecs = eff_dim.symeig_positive_lanczos(M, n_eigs=E, vecs=True)
        # assert torch.allclose(ref_vals[E:], vals, rtol=1e-4)
        # _assert_eigvecs_allclose(ref_vecs[:, E:], vecs, rtol=1e-3)

    @requires_float64
    def test_hessian_eigs_positive_lanczos(self):
        "Compare `eff_dim.hessian_eigs_positive_lanczos` with hessian then `torch.symeig`"
        model, fn, loader = self.example_model_loss_data()
        M = eff_dim.hessian(model.parameters(), fn, loader)
        ref_vals, ref_vecs = torch.symeig(M, eigenvectors=True)
        num_negative = (ref_vals < 0).sum().item()

        # All values/vectors:
        vals, vecs = eff_dim.hessian_eigs_positive_lanczos(
            model.parameters(), fn, loader, n_eigs=-1, vecs=True)
        assert torch.allclose(ref_vals[num_negative:], vals[num_negative:])
        assert torch.allclose(torch.ones(()), vals[:num_negative])

        assert torch.allclose(torch.zeros(()), vecs[:, :num_negative])
        _assert_eigvecs_allclose(ref_vecs[:, num_negative:], vecs[:, num_negative:])


    def test_PD_hessian_eigs(self, D=4):
        "Like `test_hessian_eigs_pozitive_lanczos` but for a positive-definite Hessian"
        model, fn, loader = self.example_model_loss_data(linear_only=True)

        M = eff_dim.hessian(model.parameters(), fn, loader)
        ref_vals, ref_vecs = torch.symeig(M, eigenvectors=True)
        num_negative = (ref_vals < 0).sum().item()

        # All values/vectors:
        vals, vecs = eff_dim.hessian_eigs_positive_lanczos(
            model.parameters(), fn, loader, n_eigs=-1, vecs=True)
        assert torch.allclose(ref_vals[num_negative:], vals[num_negative:], rtol=1e-4)
        assert torch.allclose(torch.ones(()), vals[:num_negative])
        _assert_eigvecs_allclose(ref_vecs[:, num_negative:], vecs[:, num_negative:], rtol=1e-3)
