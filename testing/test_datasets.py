import unittest
import torch

from bnn_priors.data import UCI, MNIST, RotatedMNIST, CIFAR10, CIFAR10_C


class _TestDataSet():
    def test_shape(self):
        assert self.data.norm.X.shape[-len(self.data.in_shape):]  == self.data.in_shape
        assert self.data.norm.X.shape[0] == self.data.norm.y.shape[0]
        assert self.data.norm.X.shape == self.shape
        assert self.data.norm.X.shape == self.data.unnorm.X.shape
        assert self.data.norm.y.shape == self.data.unnorm.y.shape
        
    def test_normalization(self):
        assert self.data.norm.X.mean().item() < 2.
        assert self.data.norm.X.mean().item() > -2.
        assert self.data.norm.X.std().item() < 2.
        
        
class TestMNIST(unittest.TestCase, _TestDataSet):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.data = MNIST()
        self.shape = torch.Size([70000, 784])
        
        
class TestRotatedMNIST(unittest.TestCase, _TestDataSet):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.data = RotatedMNIST()
        self.shape = torch.Size([90000, 784])
        
        
class TestCIFAR10(unittest.TestCase, _TestDataSet):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.data = CIFAR10()
        self.shape = torch.Size([60000, 3, 32, 32])
        
        
class TestCIFAR10_C(unittest.TestCase, _TestDataSet):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.data = CIFAR10_C(corruption="frost")
        self.shape = torch.Size([100000, 3, 32, 32])
        
        
class TestUCI(unittest.TestCase, _TestDataSet):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.data = UCI("boston", 0)
        self.shape = torch.Size([506, 13])
