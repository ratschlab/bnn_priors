import torch as t
import numpy as np
from torch.utils.data import TensorDataset

class Dataset:
    """
    Represents the full dataset.  We will have two copies: one normalised, one unnormalized.
    """
    def __init__(self, X, y, index_train, index_test):
        self.X = X
        self.y = y

        self.train_X = self.X[index_train]
        self.train_y = self.y[index_train]
        self.test_X  = self.X[index_test]
        self.test_y  = self.y[index_test]

        self.train = TensorDataset(self.train_X, self.train_y)
        self.test  = TensorDataset(self.test_X,  self.test_y)

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
    def __init__(self, dataset, split, dtype='float32'):
        data = np.loadtxt('data/UCI/{}/data.txt'.format(dataset)).astype(getattr(np, dtype))
        index_features = np.loadtxt('data/UCI/{}/index_features.txt'.format(dataset))
        index_target = np.loadtxt('data/UCI/{}/index_target.txt'.format(dataset))
        X_unnorm = t.from_numpy(data[:, index_features.astype(int)])
        y_unnorm = t.from_numpy(data[:, index_target.astype(int):index_target.astype(int)+1])

        # split into train and test
        index_train = np.loadtxt('data/UCI/{}/index_train_{}.txt'.format(dataset, split)).astype(int)
        index_test  = np.loadtxt('data/UCI/{}/index_test_{}.txt'.format(dataset, split)).astype(int)

        # record unnormalized dataset
        self.unnorm = Dataset(X_unnorm, y_unnorm, index_train, index_test)

        # compute normalization constants based on training set
        self.X_std = t.std(self.unnorm.train_X, 0)
        self.X_std[self.X_std == 0] = 1. # ensure we don't divide by zero
        self.X_mean = t.mean(self.unnorm.train_X, 0)

        self.y_mean = t.mean(self.unnorm.train_y)
        self.y_std  = t.std(self.unnorm.train_y)

        X_norm = (self.unnorm.X - self.X_mean)/self.X_std
        y_norm = (self.unnorm.y - self.y_mean)/self.y_std

        self.norm = Dataset(X_norm, y_norm, index_train, index_test)

        self.num_train_set = self.unnorm.X.shape[0]
        self.in_features   = self.unnorm.X.shape[1]
        self.out_features  = self.unnorm.y.shape[1]

    def denormalize_y(self, y):
        return self.y_std * y + self.y_mean



