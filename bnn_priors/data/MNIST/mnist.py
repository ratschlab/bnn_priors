import os
import torch as t
import torchvision
import numpy as np
from scipy import ndimage
from bnn_priors.data import Dataset


__all__ = ('MNIST','RotatedMNIST', 'FashionMNIST')


class MNIST:
    """
    The usage is:
    ```
    mnist = MNIST()
    ```
    e.g. normalized training dataset:
    ```
    mnist.norm.train
    ```
    """
    def __init__(self, dtype='float32', device="cpu", download=False):
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/mnist/'
        
        # load data
        data_train = torchvision.datasets.MNIST(dataset_dir, download=download, train=True)
        data_test = torchvision.datasets.MNIST(dataset_dir, download=download, train=False)

        # get data into right shape and type
        X_unnorm = t.from_numpy(np.concatenate([data_train.data, data_test.data]).astype(dtype)).reshape([-1, 784])
        y = t.from_numpy(np.concatenate([data_train.targets, data_test.targets]).astype('int'))
        
        # train / test split
        index_train = np.arange(len(data_train))
        index_test = np.arange(len(data_train), len(data_train) + len(data_test))
        
        # create unnormalized data set
        self.unnorm = Dataset(X_unnorm, y, index_train, index_test, device)
        
        # create normalized data set
        X_norm = self.unnorm.X / 255.
        self.norm = Dataset(X_norm, y, index_train, index_test, device)

        # save some data shapes
        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape   = self.unnorm.X.shape[1:]
        self.out_shape  = self.unnorm.y.shape[1:]
        
        
class RotatedMNIST:
    """
    The usage is:
    ```
    rot_mnist = RotatedMNIST()
    ```
    e.g. normalized training dataset:
    ```
    rot_mnist.norm.train
    ```
    """
    def __init__(self, dtype='float32', device="cpu", download=False):
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/mnist/'
        
        # load data
        data_train = torchvision.datasets.MNIST(dataset_dir, download=download, train=True)
        data_test = torchvision.datasets.MNIST(dataset_dir, download=download, train=False)
        
        # Rotate the images
        np.random.seed(1337)
        
        data_test_rot_small = np.zeros_like(data_test.data)
        labels_rot_small = np.zeros_like(data_test.targets)

        for i, img in enumerate(data_test.data):
            angle = np.random.randint(low=-45, high=45)
            img_rot = ndimage.rotate(img, angle, reshape=False)
            data_test_rot_small[i] = img_rot
            labels_rot_small[i] = data_test.targets[i]
            
        data_test_rot_large = np.zeros_like(data_test.data)
        labels_rot_large = np.zeros_like(data_test.targets)

        for i, img in enumerate(data_test.data):
            angle = np.random.randint(low=-90, high=90)
            img_rot = ndimage.rotate(img, angle, reshape=False)
            data_test_rot_large[i] = img_rot
            labels_rot_large[i] = data_test.targets[i]

        # get data into right shape and type
        X_unnorm = t.from_numpy(np.concatenate([data_train.data, data_test.data, data_test_rot_small,
                                                data_test_rot_large]).astype(dtype)).reshape([-1, 784])
        y = t.from_numpy(np.concatenate([data_train.targets, data_test.targets, labels_rot_small,
                                         labels_rot_large]).astype('int'))
        
        # train / test split
        index_train = np.arange(len(data_train))
        index_test = np.arange(len(data_train), len(data_train) + 3*len(data_test))
        
        # create unnormalized data set
        self.unnorm = Dataset(X_unnorm, y, index_train, index_test, device)
        
        # create normalized data set
        X_norm = self.unnorm.X / 255.
        self.norm = Dataset(X_norm, y, index_train, index_test, device)

        # save some data shapes
        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape   = self.unnorm.X.shape[1:]
        self.out_shape  = self.unnorm.y.shape[1:]

        
class FashionMNIST:
    """
    The usage is:
    ```
    fmnist = FashionMNIST()
    ```
    e.g. normalized training dataset:
    ```
    fmnist.norm.train
    ```
    """
    def __init__(self, dtype='float32', device="cpu", download=False):
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/mnist/'
        
        # load data
        data_train = torchvision.datasets.FashionMNIST(dataset_dir, download=download, train=True)
        data_test = torchvision.datasets.FashionMNIST(dataset_dir, download=download, train=False)

        # get data into right shape and type
        X_unnorm = t.from_numpy(np.concatenate([data_train.data, data_test.data]).astype(dtype)).reshape([-1, 784])
        y = t.from_numpy(np.concatenate([data_train.targets, data_test.targets]).astype('int'))
        
        # train / test split
        index_train = np.arange(len(data_train))
        index_test = np.arange(len(data_train), len(data_train) + len(data_test))
        
        # create unnormalized data set
        self.unnorm = Dataset(X_unnorm, y, index_train, index_test, device)
        
        # create normalized data set
        X_norm = self.unnorm.X / 255.
        self.norm = Dataset(X_norm, y, index_train, index_test, device)

        # save some data shapes
        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape   = self.unnorm.X.shape[1:]
        self.out_shape  = self.unnorm.y.shape[1:]
