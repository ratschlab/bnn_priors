import os
import torch as t
import torchvision
from torchvision import transforms
import numpy as np
from ..base import Dataset, DatasetFromTorch, load_all


__all__ = ('CIFAR10','CIFAR10_C', 'SVHN', 'CIFAR10Augmented', 'CIFAR10Small')


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
    def __init__(self, dtype='float32', device="cpu", download=False):
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/cifar10/'
        self.dtype = dtype
        self.device = device
        
        # load data
        data_train = torchvision.datasets.CIFAR10(dataset_dir, download=download, train=True)
        data_test = torchvision.datasets.CIFAR10(dataset_dir, download=download, train=False)

        self._save_datasets(data_train.data, data_test.data, data_train.targets, data_test.targets)

    def _save_datasets(self, inputs_train, inputs_test, labels_train, labels_test, permutation=(0,3,1,2)):
        # get data into right shape and type
        X_unnorm = t.from_numpy(np.concatenate([inputs_train, inputs_test]).astype(self.dtype)).permute(permutation)
        y = t.from_numpy(np.concatenate([labels_train, labels_test]).astype('int'))
        # alternative version to yield one-hot vectors
        # y = t.from_numpy(np.eye(10)[np.concatenate([data_train.targets, data_test.targets])].astype(dtype))
        
        # train / test split
        index_train = np.arange(len(inputs_train))
        index_test = np.arange(len(inputs_train), len(inputs_train) + len(inputs_test))
        
        # create unnormalized data set
        self.unnorm = Dataset(X_unnorm, y, index_train, index_test, self.device)
        
        # compute normalization constants based on training set
        self.X_std = t.std(self.unnorm.train_X, (0, 2, 3), keepdims=True)
        self.X_mean = t.mean(self.unnorm.train_X, (0, 2, 3), keepdims=True)
        
        # create normalized data set
        X_norm = (self.unnorm.X - self.X_mean)/self.X_std
        self.norm = Dataset(X_norm, y, index_train, index_test, self.device)

        # save some data shapes
        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape   = self.unnorm.X.shape[1:]
        self.out_shape  = self.unnorm.y.shape[1:]

        
class CIFAR10_C(CIFAR10):
    """
    The usage is:
    ```
    cifar10_c = CIFAR10_C()
    ```
    e.g. normalized training dataset:
    ```
    cifar10_c.norm.train
    ```
    The corrupted data has to be downloaded from
    https://zenodo.org/record/2535967
    """
    def __init__(self, corruption, dtype='float32', device="cpu", download=False):
        super().__init__(dtype, device)
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir_normal = f'{_ROOT}/cifar10/'
        dataset_dir_corrupted = f'{_ROOT}/cifar10-c/CIFAR-10-C'
        
        corruptions = ['fog',
             'jpeg_compression',
             'zoom_blur',
             'speckle_noise',
             'glass_blur',
             'spatter',
             'shot_noise',
             'defocus_blur',
             'elastic_transform',
             'gaussian_blur',
             'frost',
             'saturate',
             'brightness',
             'snow',
             'gaussian_noise',
             'motion_blur',
             'contrast',
             'impulse_noise',
             'pixelate']
        
        assert corruption in corruptions, f"Corruption {corruption} not found"
        
        # load data
        data_train = torchvision.datasets.CIFAR10(dataset_dir_normal,
                                                  download=download, train=True)
        data_test = np.load(os.path.join(dataset_dir_corrupted, f"{corruption}.npy"))
        labels = np.load(os.path.join(dataset_dir_corrupted, f"labels.npy"))

        self._save_datasets(data_train.data, data_test, data_train.targets, labels)

        
class SVHN(CIFAR10):
    """
    The usage is:
    ```
    svhn = SVHN()
    ```
    e.g. normalized training dataset:
    ```
    svhn.norm.train
    ```
    """
    def __init__(self, dtype='float32', device="cpu", download=False):
        super().__init__(dtype, device)
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/svhn/'
        
        # load data
        data_train = torchvision.datasets.SVHN(root=dataset_dir, download=download, split="train")
        data_test = torchvision.datasets.SVHN(root=dataset_dir, download=download, split="test")

        self._save_datasets(data_train.data, data_test.data, data_train.labels,
                            data_test.labels, permutation=(0,1,2,3))

class CIFAR10Augmented:
    def __init__(self, dtype='float32', device="cpu", download=False):
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/cifar10/'
        dtype = getattr(t, dtype)
        self.dtype = dtype
        self.device = device

        unnorm_train = torchvision.datasets.CIFAR10(
            dataset_dir, download=download, train=True, transform=transforms.ToTensor())
        unnorm_x, _ = load_all(unnorm_train)
        X_mean = unnorm_x.mean(dim=(0, 2, 3), keepdims=True)
        X_std = unnorm_x.std(dim=(0, 2, 3), keepdims=True)
        self.X_mean = X_mean
        self.X_std = X_std

        X_mean_tuple = tuple(a.item() for a in X_mean.view(-1))
        X_std_tuple = tuple(a.item() for a in X_std.view(-1))

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(X_mean_tuple, X_std_tuple),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(X_mean_tuple, X_std_tuple),
        ])
        data_train = torchvision.datasets.CIFAR10(dataset_dir, download=download, train=True, transform=transform_train)
        data_test = torchvision.datasets.CIFAR10(dataset_dir, download=download, train=False, transform=transform_test)

        self.norm = DatasetFromTorch(data_train, data_test, device=device)
        self.num_train_set = len(data_train)
        self.in_shape = t.Size([3, 32, 32])
        self.out_shape = t.Size([])


class CIFAR10Small(CIFAR10Augmented):
    def __init__(self, dtype='float32', device="cpu", download=False, subset_size=5000):
        super().__init__(dtype=dtype, device=device, download=download)
        # Dataset in order is not exactly balanced, but close enough.
        # for subset_size=5000 we have, {6: 519, 9: 498, 4: 519, 1: 460, 2: 519, 7: 486, 8: 520, 3: 486, 5: 488, 0: 505}
        self.norm.train.data = self.norm.train.data[:subset_size]
        self.norm.train.targets = self.norm.train.targets[:subset_size]
        self.norm.train_X = self.norm.train_X[:subset_size]
        self.norm.train_y = self.norm.train_y[:subset_size]
        self.num_train_set = subset_size
