"""
Copyright 2020 Wesley Maddox, Greg Benton, and Andrew G. Wilson
(https://github.com/g-benton/hessian-eff-dim/blob/temp/hess/utils.py)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import numpy as np
from typing import Sequence, Optional, List, Callable, Any, Tuple
from torch import Tensor

from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag


def unflatten_like(vector: Tensor, seq: Sequence[Tensor]):
    """Takes a flat `Tensor` (`vector`) and unflattens it to a list of `Tensor`s
    shaped like the tensors in `seq`.
    """
    assert vector.dim() == 1
    out = []
    i = 0
    for t in seq:
        n = t.numel()
        out.append(vector[i:i+n].view(t.shape))
        i += n
    return out


def flatten(seq: Sequence[Tensor], out: Optional[Tensor]=None):
    "Concatenate `Tensor`s in `seq` into a flat vector"
    seq = list(seq)
    numel = sum(t.numel() for t in seq)
    if out is None:
        out = torch.empty(numel, dtype=seq[0].dtype, device=seq[0].device)
    assert out.dim() == 1 and out.size(0) == numel, (
        "`out` must be flat and of the right length")

    i = 0
    for t in seq:
        n = t.numel()
        out[i:i+n] = t.detach().view(-1)
        i += n
    return out


def flatten_grads(seq: Sequence[Tensor], out=None):
    "Concatenate the gradients of `seq` into a flat vector"
    return flatten((p.grad for p in seq), out=out)


def hess_vec_prod(vec: Sequence[Tensor], parameters: List[Tensor],
                  fn: Callable[..., Tensor], dataloader: Sequence[Tuple]):
    """Evaluate product of the Hessian of the loss function with a direction
    vector `vec`. (Hessian-vector product, HVP)

    On returning, the HVP is stored in `(p.grad for p in parameters)`

    Arguments:
        vec: a list of tensor with the same dimensions as `parameters`.
        parameters: the parameters to calculate the HVP of.
        fn: The function for which to calculate the HVP.
        dataloader: an iterator that gives the arguments to `fn` at every iteration. It is assumed that
          `fn(*[torch.cat(list(d[i] for d in dataloader)) for i in n_args])
          == sum(fn(*d) for d in dataloader)`
          This is true when len(dataloader) == 1, and for most losses in machine learning.
    """
    parameters = list(parameters)
    # Clear gradients
    for p in parameters:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    for args in dataloader:
        objective = fn(*args)
        assert objective.dim() == 0, "output of `fn` must be a scalar"
        grad = torch.autograd.grad(objective, inputs=parameters, create_graph=True)

        # Inner product of gradient with direction vector
        prod = sum(g.view(-1) @ v.view(-1) for (g, v) in zip(grad, vec))
        prod.backward()  # Store HVP in the `p.grad`s


def hessian(parameters: Sequence[Tensor], fn: Callable[..., Tensor],
            dataloader: Sequence[Tuple]) -> Tensor:
    """Evaluate the Hessian of `fn` with respect to `parameters`.

    Arguments:
        parameters: the parameters for the Hessian.
        fn: the function for the Hessian
        dataloader: an iterator that gives the arguments to `fn` at every iteration. It is assumed that
          `fn(*[torch.cat(list(d[i] for d in dataloader)) for i in n_args])
          == sum(fn(*d) for d in dataloader)`
          This is true when len(dataloader) == 1, and for most losses in machine learning.
    """
    parameters = list(parameters)
    numel = sum(p.numel() for p in parameters)
    hessian = torch.zeros((numel, numel), dtype=parameters[0].dtype, device='cpu')

    base_vec = torch.zeros(numel, device=parameters[0].device, dtype=parameters[0].dtype)

    for i in range(numel):
        base_vec.zero_()
        base_vec[i] = 1.
        hess_vec_prod(unflatten_like(base_vec, parameters),
                      parameters, fn, dataloader)

        flatten_grads(parameters, out=hessian[i, :])
    return hessian


def _eig_lanczos(mvmul_closure, n_eigs, vecs, dtype, device, numel):
    if n_eigs == -1:
        n_eigs = numel
    assert 1 <= n_eigs and n_eigs <= numel

    qmat, t_mat = lanczos_tridiag(mvmul_closure, n_eigs,
                                  dtype=dtype, device=device,
                                  matrix_shape=(numel, numel), num_init_vecs=1)
    vals, t_vals = lanczos_tridiag_to_diag(t_mat)
    return vals, (qmat @ t_vals if vecs else None)


def hessian_eigs_positive_lanczos(parameters: Sequence[Tensor],
                                  fn: Callable[..., Tensor], dataloader: Sequence[Tuple],
                                  n_eigs: int=-1, vecs: bool=True) -> Tuple[Tensor, Optional[Tensor]]:

    """Returns the leading `n_eigs` approximate eigenvalues and eigenvectors of
    the Hessian of `fn` with respect to `parameters`, without evaluating the
    full Hessian. If `n_eigs` is -1, the eigenvalues and eigenvectors are
    exact.

    If any of these eigenvalues is negative, it is returned as 1.0. Its
    corresponding eigenvector is all zeros.

    The output of this function is quite sensitive to floating point precision.
    For best results, use `float64`.

    Arguments:
        parameters: the parameters for the Hessian.
        fn: the function for the Hessian
        dataloader: an iterator that gives the arguments to `fn` at every iteration. It is assumed that
          `fn(*[torch.cat(list(d[i] for d in dataloader)) for i in n_args])
          == sum(fn(*d) for d in dataloader)`
          This is true when len(dataloader) == 1, and for most losses in machine learning.
        n_eigs (int): the number of eigenvalues and vectors to calculate. Use
          `-1` to calculate all of them.
        vecs: whether to calculate the eigenvectors

    Returns:
        An (eigenvalues, Q) tuple. The columns of `Q` are the eigenvectors.
        The eigenvalues are in ascending order.

    """
    parameters = list(parameters)
    numel = sum(p.numel() for p in parameters)

    def hvp(vector):
        assert vector.dim() == 2 and vector.size(1) == 1
        hess_vec_prod(unflatten_like(vector.view(-1), parameters),
                      parameters, fn, dataloader)
        return flatten_grads(parameters).unsqueeze(1)
    return _eig_lanczos(hvp, n_eigs, vecs, parameters[0].dtype,
                        parameters[0].device, numel)


def symeig_positive_lanczos(mat: Tensor, n_eigs: int=-1, vecs: bool=True
                            ) -> Tuple[Tensor, Optional[Tensor]]:

    """Returns the leading `n_eigs` approximate eigenvalues and eigenvectors of
    symmetrical matrix `mat`. If `n_eigs` is -1, the eigenvalues and
    eigenvectors are exact.

    If any of these eigenvalues is negative, it is returned as 1.0. Its
    corresponding eigenvector is all zeros.

    The output of this function is quite sensitive to floating point precision.
    For best results, use `float64`.

    Arguments:
        mat: the symmetric matrix for which to calculate the eigendecomposition
        n_eigs (int): the number of eigenvalues and vectors to calculate. Use
          `-1` to calculate all of them.
        vecs: whether to calculate the eigenvectors

    Returns:
        An (eigenvalues, Q) tuple. The columns of `Q` are the eigenvectors.
        The eigenvalues are in ascending order.

    """
    assert mat.dim() == 2 and mat.size(0) == mat.size(1)
    def mvmul_closure(vector):
        return mat @ vector
    return _eig_lanczos(mvmul_closure, n_eigs, vecs, mat.dtype, mat.device,
                        mat.size(0))

