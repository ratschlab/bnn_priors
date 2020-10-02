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
                        metrics.add_scalar("step11", float(step//11), step)
                    if step==-1 or step%23 == 0:
                        metrics.add_scalar("step23", step//23, step)

                    if step%31 == 0:
                        metrics.flush()
                        with h5py.File(fname, "r", swmr=True) as read_metrics:
                            for k in read_metrics.keys():
                                assert len(read_metrics[k]) == step+2


            the_len = 101
            buf = np.zeros(the_len, dtype=np.int64)
            with h5py.File(fname, "r") as f:
                for k in f.keys():
                    assert len(f[k]) == the_len
                    if k != "timestamps":
                        assert f[k][0] == -1

                assert np.all(~np.isnan(f["timestamps"][:]))

                assert np.array_equal(f["steps"][:], np.arange(-1, 100))
                assert np.array_equal(f["steps"][:], f["re_step"][:])


                # -2**63 is the minimum possible int64. It is what numpy
                # converts NaNs to when assigning them to arrays of size larger than 1
                # (for some reason arrays with 1 element throw a ValueError)
                a = np.zeros([2], dtype=np.int64)
                a[:] = np.nan
                assert np.all(a[:] == -2**63), "will be true"

                assert np.array_equal(f["step5"][1::5], np.arange(100//5))
                for i in range(1, 5):
                    assert np.all(f["step5"][1+i::5] == -2**63)

                assert np.array_equal(f["step11"][1::11], np.arange(100//11 + 1).astype(np.float64))
                for i in range(1, 11):
                    assert np.all(np.isnan(f["step11"][1+i::11]))

                assert np.array_equal(f["step23"][1::23], np.arange(100//23 + 1))
                for i in range(1, 23):
                    assert np.all(f["step23"][1+i::23] == -2**63)
