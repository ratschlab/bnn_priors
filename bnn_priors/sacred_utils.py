import collections
import contextlib
import os
import pickle
from pathlib import Path

import numpy as np
import sacred
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, TensorDataset

import gpytorch

__all__ = ["load_sorted_dataset", "interlaced_argsort",
           "base_dir", "new_file"]

ingredient = sacred.Ingredient("i_SU")


@ingredient.config
def config():
    # GPytorch
    num_likelihood_samples = 20
    # Pytorch
    default_dtype = "float64"

    # Dataset loading
    dataset_base_path = "/scratch/ag919/datasets/"
    dataset_name = "CIFAR10"
    sorted_dataset_path = os.path.join(dataset_base_path, "interlaced_argsort")
    N_train = None
    N_test = None
    ZCA_transform = False
    ZCA_bias = 1e-5

    # Whether to take the "test" set from the end of the training set
    test_is_validation = True
    dataset_treatment = "train_random_balanced"
    train_idx_path = None


# GPytorch
@ingredient.pre_run_hook
def gpytorch_pre_run_hook(num_likelihood_samples, default_dtype, _seed):
    gpytorch.settings.num_likelihood_samples._set_value(num_likelihood_samples)
    # disable CG, it makes the eigenvalues of test Gaussian negative :(
    gpytorch.settings.max_cholesky_size._set_value(1000000)
    torch.set_default_dtype(getattr(torch, default_dtype))
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)

@ingredient.post_run_hook
def print_experiment(_run):
    print(f"This was run {_run._id}")



# File observer creation
def add_file_observer(experiment, default_dir="/home/ag919/rds/hpc-work/logs"):
    try:
        log_dir = Path(os.environ['LOG_DIR'])
    except KeyError:
        log_dir = Path(os.environ['HOME'])/"rds/hpc-work/logs"
    log_dir = log_dir/experiment.path
    experiment.observers.append(
        sacred.observers.FileStorageObserver(str(log_dir)))


# File handling
@ingredient.capture
def base_dir(_run, _log):
    try:
        return Path(_run.observers[0].dir)
    except IndexError:
        _log.warning("This run has no associated directory, using `/tmp`")
        return Path("/tmp")


@contextlib.contextmanager
def new_file(relative_path, mode="wb"):
    full_path = os.path.join(base_dir(), relative_path)
    with open(full_path, mode) as f:
        yield f


# Datasets and sorted data sets
@ingredient.capture
def load_dataset(dataset_name, dataset_base_path, ZCA_transform, additional_transforms=[]):
    dataset_base_path = Path(dataset_base_path)
    if dataset_name == "CIFAR10_ZCA_wrong":
        assert not ZCA_transform, "double ZCA"
        data = np.load(dataset_base_path/"CIFAR10_ZCA_wrong.npz")
        train = TensorDataset(*(torch.from_numpy(data[a]) for a in ("X", "y")))
        test = TensorDataset(*(torch.from_numpy(data[a]) for a in ("Xt", "yt")))
        return train, test

    elif dataset_name == "CIFAR10_ZCA_shankar_wrong":
        assert not ZCA_transform, "double ZCA"
        data = np.load(dataset_base_path/"cifar_10_zca_augmented_extra_zca_augment_en9fKkGMMg.npz")
        train = TensorDataset(
            torch.from_numpy(data["X_train"]).permute(0, 3, 1, 2),
            torch.from_numpy(data["y_train"]))
        test = TensorDataset(
            torch.from_numpy(data["X_test"]).permute(0, 3, 1, 2),
            torch.from_numpy(data["y_test"]))
        return train, test

    if dataset_name == "CIFAR10":
        dataset_base_path = dataset_base_path/dataset_name
    if len(additional_transforms) == 0:
        trans = torchvision.transforms.ToTensor()
    else:
        trans = torchvision.transforms.Compose([
            *additional_transforms,
            torchvision.transforms.ToTensor(),
        ])

    _dset = getattr(torchvision.datasets, dataset_name)
    train = _dset(dataset_base_path, train=True, download=True, transform=trans)
    test = _dset(dataset_base_path, train=False, transform=trans)
    return train, test


def interlaced_argsort(dset):
    y = torch.tensor(dset.targets)
    starting_for_class = [None] * len(dset.classes)
    argsort_y = torch.argsort(y)
    for i, idx in enumerate(argsort_y):
        if starting_for_class[y[idx]] is None:
            starting_for_class[y[idx]] = i

    for i in range(len(starting_for_class)-1):
        assert starting_for_class[i] < starting_for_class[i+1]
    assert starting_for_class[0] == 0

    init_starting = [a for a in starting_for_class] + [len(dset)]

    res = []
    while len(res) < len(dset):
        for i in range(len(starting_for_class)):
            if starting_for_class[i] < init_starting[i+1]:
                res.append(argsort_y[starting_for_class[i]].item())
                starting_for_class[i] += 1
    assert len(set(res)) == len(dset)
    assert len(res) == len(dset)
    return res


def whole_dset(dset):
    return next(iter(DataLoader(dset, batch_size=len(dset))))


@ingredient.capture
def _apply_zeromean(X, rgb_mean, do_zero_mean):
    if not do_zero_mean:
        return X
    rgb_mean = rgb_mean.to(X.device)
    X -= rgb_mean
    return X

def _apply_ZCA(X, W):
    orig_dtype = X.dtype
    shape = X.size()
    W = W.to(X.device)
    X = X.reshape((X.size(0), -1)).to(W.dtype) @ W
    return X.reshape(shape).to(orig_dtype)


def _norm_shankar(X):
    "Perform normalization like https://github.com/modestyachts/neural_kernels_code/blob/ef09d4441cfc901d7a845ffac88ddd4754d4602e/utils.py#L280"
    # Per-instance zero-mean
    X = X - X.mean((1, 2, 3), keepdims=True)
    # Normalize each instance
    sqnorm = X.pow(2).sum((1, 2, 3), keepdims=True)
    return X * sqnorm.add(1e-16).pow(-1/2)


@ingredient.capture
def do_transforms(train_set, test_set, ZCA_transform: bool, ZCA_bias: float):
    X, y = whole_dset(train_set)
    device = ('cuda' if torch.cuda.device_count() else 'cpu')
    X = X.to(device=device, dtype=torch.float64)
    if ZCA_transform:
        X = _norm_shankar(X)
        # The output of GPU and CPU SVD is different, so we do it in CPU and
        # then bring it back to `device`
        _, S, V = torch.svd(X.reshape((len(train_set), -1)).cpu())
        S = S.to(device)
        V = V.to(device)

        # X^T @ X  ==  (V * S^2) @ V.T
        S = S.pow(2).div(len(train_set)).add(ZCA_bias).sqrt()
        W = (V / S) @ V.t()
        X = _apply_ZCA(X, W)

    Xt, yt = whole_dset(test_set)
    Xt = Xt.to(device=device, dtype=torch.float64)
    if ZCA_transform:
        Xt = _norm_shankar(Xt)
        Xt = _apply_ZCA(Xt, W)

        W = W.cpu()
    else:
        W = None
    return TensorDataset(X.cpu(), y), TensorDataset(Xt.cpu(), yt), W


def class_balanced_train_idx(train_set, N_train, forbidden_indices=None):
    train_y = torch.tensor(train_set.targets)
    argsort_y = torch.argsort(train_y)
    N_classes = len(train_set.classes)

    if forbidden_indices is not None:
        assert isinstance(forbidden_indices, set)
        new_argsort_y = filter(lambda x: x.item() not in forbidden_indices,
                               argsort_y)
        new_argsort_y = torch.tensor(list(new_argsort_y), dtype=torch.int64)

        assert len(set(new_argsort_y.numpy()).intersection(forbidden_indices)) == 0
        assert len(new_argsort_y) + len(forbidden_indices) == len(argsort_y)
        argsort_y = new_argsort_y

    starting_for_class = [None] * N_classes
    for i, idx in enumerate(argsort_y):
        if starting_for_class[train_y[idx]] is None:
            starting_for_class[train_y[idx]] = i

    min_gap = len(train_set)
    for prev, nxt in zip(starting_for_class, starting_for_class[1:]):
        min_gap = min(min_gap, nxt-prev)
    assert min_gap >= N_train//N_classes, "Cannot draw balanced data set"
    train_idx_oneclass = torch.randperm(min_gap)[:N_train//N_classes]

    train_idx = torch.cat([argsort_y[train_idx_oneclass + start]
                            for start in starting_for_class])
    # Check that it is balanced
    count = collections.Counter(a.item() for a in train_y[train_idx])
    for label in range(N_classes):
        assert count[label] == N_train // N_classes, "new set not balanced"
    assert len(set(train_idx)) == N_train, "repeated indices"
    return train_idx


@ingredient.capture
def load_sorted_dataset(sorted_dataset_path, N_train, N_test, ZCA_transform, test_is_validation, ZCA_bias, dataset_treatment, _run, train_idx_path):
    train_set, test_set = load_dataset()
    if dataset_treatment == "no_treatment":
        return train_set, test_set
    elif dataset_treatment == "unsorted":
        return load_unsorted_dataset()
    elif dataset_treatment in ["sorted", "sorted_legacy"]:
        with _run.open_resource(os.path.join(sorted_dataset_path, "train.pkl"), "rb") as f:
            train_idx = pickle.load(f)
        with _run.open_resource(os.path.join(sorted_dataset_path, "test.pkl"), "rb") as f:
            test_idx = pickle.load(f)

        if test_is_validation:
            assert N_train+N_test <= len(train_set), "Train+validation sets too large"
            if dataset_treatment == "sorted_legacy":
                test_set = Subset(train_set, train_idx[-N_test-1:-1])
            else:
                test_set = Subset(train_set, train_idx[-N_test:])
        else:
            test_set = Subset(test_set, test_idx[:N_test])
        train_set = Subset(train_set, train_idx[:N_train])

    elif dataset_treatment == "train_random_balanced":
        if N_test != len(test_set):
            test_set = Subset(test_set, range(N_test))
        train_y = torch.tensor(train_set.targets)
        argsort_y = torch.argsort(train_y)
        N_classes = len(train_set.classes)

        starting_for_class = [None] * N_classes
        train_idx = class_balanced_train_idx(train_set, N_train)
        np.save(base_dir()/"train_idx.npy", train_idx.numpy())
        train_set = Subset(train_set, train_idx)

    elif dataset_treatment == "extend_train_random_balanced":
        old_train_idx = np.load(Path(train_idx_path)/"train_idx.npy")
        old_train_idx = torch.from_numpy(old_train_idx)
        train_idx = class_balanced_train_idx(
            train_set, N_train, forbidden_indices=set(old_train_idx.numpy()))
        np.save(base_dir()/"train_idx.npy", train_idx.numpy())
        np.save(base_dir()/"old_train_idx.npy", old_train_idx.numpy())
        _train_set = train_set
        train_set = Subset(_train_set, train_idx)
        test_set = Subset(_train_set, old_train_idx)

    elif dataset_treatment == "newtest_train_random_balanced":
        old_train_idx = np.load(Path(train_idx_path)/"train_idx.npy")
        old_train_idx = torch.from_numpy(old_train_idx)
        np.save(base_dir()/"train_idx.npy", old_train_idx.numpy())
        _train_set = train_set
        train_set = Subset(_train_set, old_train_idx)
        test_set = test_set

    elif dataset_treatment == "extend_load_train_idx":
        train_idx = np.load(Path(train_idx_path)/"train_idx.npy")
        old_train_idx = np.load(Path(train_idx_path)/"old_train_idx.npy")
        _train_set = train_set
        train_set = Subset(_train_set, torch.from_numpy(train_idx))
        test_set = Subset(_train_set, torch.from_numpy(old_train_idx))

    elif dataset_treatment == "load_train_idx":
        train_idx = np.load(Path(train_idx_path)/"train_idx.npy")
        train_set = Subset(train_set, torch.from_numpy(train_idx))
        if N_test != len(test_set):
            test_set = Subset(test_set, range(N_test))
    else:
        raise ValueError(dataset_treatment)

    train, test, _ = do_transforms(train_set, test_set)
    return train, test


@ingredient.capture
def load_unsorted_dataset(test_is_validation, N_train, N_test):
    train_set, test_set = load_dataset()
    if test_is_validation:
        assert N_train+N_test <= len(train_set), "Train+validation sets too large"
        test_set = Subset(train_set, range(-N_test, 0, 1))
    else:
        test_set = Subset(test_set, range(N_test))
    train_set = Subset(train_set, range(N_train))
    return train_set, test_set
