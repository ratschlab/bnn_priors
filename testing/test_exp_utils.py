import unittest
import torch

from bnn_priors import exp_utils
from bnn_priors.data import UCI, MNIST, RotatedMNIST, CIFAR10, CIFAR10_C


class TestExpUtils(unittest.TestCase):
    def test_device(self):
        assert exp_utils.device("try_cuda") in [torch.device("cpu"), torch.device("cuda:0")]
        
    def test_data(self):
        device = exp_utils.device("try_cuda")
        assert isinstance(exp_utils.get_data("UCI_boston", device), UCI)
        assert isinstance(exp_utils.get_data("UCI_wine", device), UCI)
        assert isinstance(exp_utils.get_data("cifar10", device), CIFAR10)
        assert isinstance(exp_utils.get_data("cifar10c-frost", device), CIFAR10_C)
        assert isinstance(exp_utils.get_data("cifar10c-fog", device), CIFAR10_C)
        assert isinstance(exp_utils.get_data("mnist", device), MNIST)
        assert isinstance(exp_utils.get_data("rotated_mnist", device), RotatedMNIST)