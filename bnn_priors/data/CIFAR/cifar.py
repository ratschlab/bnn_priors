import os
import torch as t
import torchvision
import numpy as np
from bnn_priors.data import Dataset


__all__ = ('CIFAR10',)


class CIFAR10:
    """
    The usage is:
    ```
    cifar10 = CIFAR10()
    ```
    e.g. normalized training dataset:
    ```
    cifar10.norm.train
    ```
    """
    def __init__(self, dtype='float32', device="cpu"):
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/cifar10/'
        
        # load data
        data_train = torchvision.datasets.CIFAR10(dataset_dir, download=True, train=True)
        data_test = torchvision.datasets.CIFAR10(dataset_dir, download=True, train=False)

        # get data into right shape and type
        X_unnorm = t.from_numpy(np.concatenate([data_train.data, data_test.data]).astype(dtype)).permute(0,3,1,2)
        y = t.from_numpy(np.concatenate([data_train.targets, data_test.targets]).astype('int'))
        # alternative version to yield one-hot vectors
        # y = t.from_numpy(np.eye(10)[np.concatenate([data_train.targets, data_test.targets])].astype(dtype))
        
        # train / test split
        index_train = np.arange(len(data_train))
        index_test = np.arange(len(data_train), len(data_train) + len(data_test))
        
        # create unnormalized data set
        self.unnorm = Dataset(X_unnorm, y, index_train, index_test, device)
        
        # compute normalization constants based on training set
        self.X_std = t.std(self.unnorm.train_X, (0, 2, 3), keepdims=True)
        self.X_mean = t.mean(self.unnorm.train_X, (0, 2, 3), keepdims=True)
        
        # create normalized data set
        X_norm = (self.unnorm.X - self.X_mean)/self.X_std
        self.norm = Dataset(X_norm, y, index_train, index_test, device)

        # save some data shapes
        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape   = self.unnorm.X.shape[1:]
        self.out_shape  = self.unnorm.y.shape[1:]
