import os
import torch as t
import numpy as np
from torch.utils.data import TensorDataset

from bnn_priors.data import Dataset

__all__ = ('RandomData', 'Synthetic', 'RandomOODTestData')


class RandomData:
    """
    The usage is:
    ```
    data = RandomData(dim=64, n_points=2000)
    ```
    e.g. normalized training dataset:
    ```
    data.norm.train
    ```
    """
    def __init__(self, dim=20, n_points=2000, dtype='float32', device="cpu"):
        X_unnorm = t.from_numpy(np.random.uniform(low=-1., high=1., size=[n_points, dim]).astype(dtype))
        y_unnorm = t.from_numpy(np.random.uniform(low=-1., high=1., size=[n_points, 1]).astype(dtype))

        # split into train and test
        index_train = np.arange(n_points//2)
        index_test  = np.arange(n_points//2, n_points)

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


class RandomOODTestData(RandomData):
    def __init__(self, dim=20, n_points=2000, dtype='float32', device="cpu"):
        len_train = n_points//2
        X_unnorm = t.from_numpy(np.random.uniform(low=-1., high=1., size=[len_train, dim]).astype(dtype))
        y_unnorm = t.from_numpy(np.random.uniform(low=-1., high=1., size=[len_train, 1]).astype(dtype))

        X_test_unnorm = t.from_numpy(np.random.uniform(low=1., high=2., size=[n_points-len_train, dim]).astype(dtype))
        y_test_unnorm = t.from_numpy(np.random.uniform(low=1., high=2., size=[n_points-len_train, 1]).astype(dtype))

        X_unnorm = t.cat([X_unnorm, X_test_unnorm])
        y_unnorm = t.cat([y_unnorm, y_test_unnorm])

        index_train = np.arange(len_train)
        index_test  = np.arange(len_train, n_points)

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


    
class Synthetic:
    """
    The usage is:
    ```
    synth_data = Synthetic(dataset=data, model=net)
    ```
    e.g. normalized training dataset:
    ```
    synth_data.norm.train
    ```
    """
    def __init__(self, dataset, model, batch_size=None, dtype='float32', device="cpu"):
        if batch_size is None:
            new_y = model(dataset.norm.X).sample()
        else:
            dataloader_train = t.utils.data.DataLoader(dataset.norm.train, batch_size=batch_size)
            dataloader_test = t.utils.data.DataLoader(dataset.norm.test, batch_size=batch_size)
            batch_preds = []
            for dataloader in [dataloader_train, dataloader_test]:
                for batch_x, _ in dataloader:
                    batch_preds.append(model(batch_x).sample())
            new_y = t.cat(batch_preds)

        # split into train and test
        index_train = np.arange(len(dataset.norm.train_X))
        index_test  = np.arange(len(dataset.norm.train_X), len(dataset.norm.X))

        # record unnormalized dataset
        self.unnorm = Dataset(dataset.unnorm.X, new_y, index_train, index_test, device)
        self.norm = Dataset(dataset.norm.X, new_y, index_train, index_test, device)

        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape   = self.unnorm.X.shape[1:]
        self.out_shape  = self.unnorm.y.shape[1:]
