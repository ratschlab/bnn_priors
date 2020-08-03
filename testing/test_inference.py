import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats
from pathlib import Path

from bnn_priors.models import DenseNet
from bnn_priors.inference import SGLDRunner


class SGLDRunnerTest(unittest.TestCase):
    def test_snelson_inference(self):
        data = np.load(Path(__file__).parent/"../bnn_priors/data/snelson.npz")

        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        x_train = torch.from_numpy(data['x_train']).unsqueeze(1).to(device=device, dtype=torch.get_default_dtype())
        y_train = torch.from_numpy(data['y_train']).unsqueeze(1).to(x_train)

        x_test = torch.from_numpy(data['x_test']).unsqueeze(1).to(x_train)

        model = DenseNet(x_train.size(-1), y_train.size(-1), 128, noise_std=0.5)
        model.to(x_train)
        if torch.cuda.is_available():
            model = model.cuda()   # Resample model with He initialization so SGLD works.
        model.train()

        dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=15)

        sgld = SGLDRunner(model=model, dataloader=dataloader,
                          epochs_per_cycle=3, warmup_epochs=1, sample_epochs=1,
                          learning_rate=5e-4, temperature=1., data_mult=1., momentum=0.9,
                          sampling_decay=True, cycles=4, precond_update=2)
        sgld.run(progressbar=False)

        assert sgld.metrics["loss"][0] > sgld.metrics["loss"][-1]
        assert sgld.metrics["lr"][0] > sgld.metrics["lr"][1]
        assert (sgld.metrics["preconditioner/net.0.weight_prior.p"][0]
                != sgld.metrics["preconditioner/net.0.weight_prior.p"][-1])
