import os
import torch as t
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


__all__ = ('Dataset', 'DatasetFromTorch')


class Dataset:
    """
    Represents the full dataset.  We will have two copies: one normalised, one unnormalized.
    """
    def __init__(self, X, y, index_train, index_test, device="cpu"):
        self.X = X.to(device)
        self.y = y.to(device)

        self.train_X = self.X[index_train]
        self.train_y = self.y[index_train]
        self.test_X  = self.X[index_test]
        self.test_y  = self.y[index_test]

        self.train = TensorDataset(self.train_X, self.train_y)
        self.test  = TensorDataset(self.test_X,  self.test_y)


def load_all(dset):
    loader = DataLoader(dset, batch_size=len(dset), shuffle=False)
    return next(iter(loader))


class DatasetFromTorch(Dataset):
    def __init__(self, train, test, device):
        self.train = train
        self.test = test
        self.train_X, self.train_y = (a.to(device) for a in load_all(train))
        self.test_X, self.test_y = (a.to(device) for a in load_all(test))
