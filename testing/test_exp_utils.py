import unittest
import torch
from tempfile import TemporaryDirectory
from pathlib import Path

from bnn_priors import exp_utils
from bnn_priors.data import UCI, MNIST, RotatedMNIST, CIFAR10, CIFAR10_C
import numpy as np
import h5py


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


class TestHDF5Saver(unittest.TestCase):
    def test_recorded_metrics(self):
        with TemporaryDirectory() as directory:
            fname = Path(directory)/"metrics_test.h5"

            with exp_utils.HDF5Metrics(fname, "w", chunk_size=13) as metrics:
                for step in range(-1, 100):
                    metrics.add_scalar("re_step", step, step)
                    if step==-1 or step%5 == 0:
                        metrics.add_scalar("step5", step//5, step)
                    if step==-1 or step%11 == 0:
                        metrics.add_scalar("step11", step//11, step)
                    if step==-1 or step%23 == 0:
                        metrics.add_scalar("step23", step//23, step)

                    if step%31 == 0:
                        metrics.flush()

            the_len = 104
            buf = np.zeros(the_len, dtype=np.int64)
            with h5py.File(fname, "r") as f:
                for k in f.keys():
                    assert len(f[k]) == the_len
                    if k != "timestamps":
                        assert np.all(f[k][101:] == -2**63)
                        assert f[k][0] == -1

                assert np.all( np.isnan(f["timestamps"][101:]))
                assert np.all(~np.isnan(f["timestamps"][:101]))

                assert np.array_equal(f["steps"][:101], np.arange(-1, 100))
                assert np.array_equal(f["steps"][:], f["re_step"][:])

                assert np.array_equal(f["step5"][1:101:5], np.arange(100//5))
                assert np.array_equal(f["step11"][1:101:11], np.arange(100//11 + 1))
                assert np.array_equal(f["step23"][1:101:23], np.arange(100//23 + 1))
