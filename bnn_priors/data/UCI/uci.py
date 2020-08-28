import os
import torch as t
import numpy as np
from torch.utils.data import TensorDataset

from bnn_priors.data import Dataset

__all__ = ('UCI',)


class UCI:
    """
    The usage is:
    ```
    uci = UCIDataset("protein", 3)
    ```
    e.g. normalized training dataset:
    ```
    uci.norm.train
    ```
    """
    def __init__(self, dataset, split, dtype='float32', device="cpu"):
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/{dataset}/'
        data = np.loadtxt(f'{dataset_dir}/data.txt').astype(getattr(np, dtype))
        index_features = np.loadtxt(f'{dataset_dir}/index_features.txt')
        index_target = np.loadtxt(f'{dataset_dir}/index_target.txt')
        X_unnorm = t.from_numpy(data[:, index_features.astype(int)])
        y_unnorm = t.from_numpy(data[:, index_target.astype(int):index_target.astype(int)+1])

        # split into train and test
        index_train = np.loadtxt(f'{dataset_dir}/index_train_{split}.txt').astype(int)
        index_test  = np.loadtxt(f'{dataset_dir}/index_test_{split}.txt').astype(int)

        # record unnormalized dataset
        self.unnorm = Dataset(X_unnorm, y_unnorm, index_train, index_test, device)

        # compute normalization constants based on training set
        self.X_std = t.std(self.unnorm.train_X, 0)
        self.X_std[self.X_std == 0] = 1. # ensure we don't divide by zero
        self.X_mean = t.mean(self.unnorm.train_X, 0)

        self.y_mean = t.mean(self.unnorm.train_y)
        self.y_std  = t.std(self.unnorm.train_y)

        X_norm = (self.unnorm.X - self.X_mean)/self.X_std
        y_norm = (self.unnorm.y - self.y_mean)/self.y_std

        self.norm = Dataset(X_norm, y_norm, index_train, index_test, device)

        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape   = self.unnorm.X.shape[1:]
        self.out_shape  = self.unnorm.y.shape[1:]

    def denormalize_y(self, y):
        return self.y_std * y + self.y_mean



