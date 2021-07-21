import math
import h5py
import numpy as np
import time
from sklearn.metrics import average_precision_score, roc_auc_score
import torch as t
import bnn_priors.data
from bnn_priors.models import RaoBDenseNet, DenseNet, PreActResNet18, ThinPreActResNet18, PreActResNet34, ClassificationDenseNet, ResNet, ClassificationConvNet, CorrelatedClassificationConvNet, CorrelatedResNet
import bnn_priors.models
from bnn_priors.prior import LogNormal
from bnn_priors import prior
from bnn_priors.prior import get_prior
from bnn_priors.third_party.calibration_error import ece, ace, rmsce
import warnings
import sacred
from pathlib import Path

from typing import Dict, Iterable, Tuple


def device(device: str):
    if device == "try_cuda":
        if t.cuda.is_available():
            return t.device("cuda:0")
        else:
            return t.device("cpu")
    return t.device(device)


def get_data(data: str, device: t.device):
    if data[:3] == "UCI":
        uci_dataset = data.split("_")[1]
        assert uci_dataset in ["boston", "concrete", "energy", "kin8nm",
                               "naval", "power", "protein", "wine", "yacht"]
        # TODO: do we ever use a different split than 0?
        dataset = bnn_priors.data.UCI(uci_dataset, 0, device=device)
    elif data[:8] == "cifar10c":
        corruption = data.split("-")[1]
        dataset = bnn_priors.data.CIFAR10_C(corruption, device=device)
    elif data == "cifar10":
        dataset = bnn_priors.data.CIFAR10(device=device)
    elif data == "cifar10_augmented":
        dataset = bnn_priors.data.CIFAR10Augmented(device=device)
    elif data == "cifar10_small":
        dataset = bnn_priors.data.CIFAR10Small(device=device)
    elif data == "mnist":
        dataset = bnn_priors.data.MNIST(device=device)
    elif data == "rotated_mnist":
        dataset = bnn_priors.data.RotatedMNIST(device=device)
    elif data == "fashion_mnist":
        dataset = bnn_priors.data.FashionMNIST(device=device)
    elif data == "svhn":
        dataset = bnn_priors.data.SVHN(device=device)
    elif data == "random":
        dataset = bnn_priors.data.RandomData(device=device)
    elif data == "random_ood_test":
        dataset = bnn_priors.data.RandomOODTestData(device=device)
    else:
        raise ValueError(f"Unknown data='{data}'")
    return dataset


def he_initialize(model):
    for name, param in model.named_parameters():
        if "weight_prior.p" in name:
            t.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
        elif "bias_prior.p" in name:
            bound = 1 / math.sqrt(param.size(0))
            t.nn.init.uniform_(param.data, -bound, bound)


def he_zerobias_initialize(model):
    for name, param in model.named_parameters():
        if "weight_prior.p" in name:
            t.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
        elif "bias_prior.p" in name:
            t.nn.init.zeros_(param.data)

def he_uniform_initialize(model):
    for name, param in model.named_parameters():
        if "weight_prior.p" in name:
            if "conv" in name or "shortcut" in name or param.dim() == 4:
                t.nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))
            elif "linear" in name or param.dim() == 2:
                bound = 1 / math.sqrt(param.size(1))
                t.nn.init.uniform_(param.data, -bound, bound)
            else:
                raise NotImplementedError(name)
        elif "bias_prior.p" in name:
            if "conv" in name or "shortcut" in name:
                raise NotImplementedError(name)
            elif "linear" in name or param.dim() == 1:
                bound = 1 / math.sqrt(param.size(0))
                t.nn.init.uniform_(param.data, -bound, bound)
            else:
                raise NotImplementedError(name)


class DummyModule(t.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def get_model(x_train, y_train, model, width, depth, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm,
             weight_prior_params, bias_prior_params):
    if weight_prior in ["cauchy"]:
        # NOTE: Cauchy and anything with infinite variance should use this
        scaling_fn = lambda std, dim: std/dim
    else:
        scaling_fn = lambda std, dim: std/dim**0.5
    weight_prior = get_prior(weight_prior)
    bias_prior = get_prior(bias_prior)
    if model == "densenet":
        net = DenseNet(x_train.size(-1), y_train.size(-1), width, depth, noise_std=LogNormal((), -1., 0.2),
                        prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                        prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                      weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params).to(x_train)
    elif model == "raobdensenet":
        net = RaoBDenseNet(x_train, y_train, width, noise_std=LogNormal((), -1., 0.2)).to(x_train)
    elif model == "classificationdensenet":
        net = ClassificationDenseNet(x_train.size(-1), y_train.max()+1, width, depth, softmax_temp=1.,
                        prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                        prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                        weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params).to(x_train)
    elif model in ["classificationconvnet", "correlatedclassificationconvnet",
                   "datadrivengaussconv", "datadrivendoublegammaconv"]:
        if len(x_train.shape) == 4:
            in_channels = x_train.shape[1]
            img_height = x_train.shape[-2]
        else:
            in_channels = 1
            img_height = int(math.sqrt(x_train.shape[-1]))

        if model == "classificationconvnet":
            network_class = ClassificationConvNet
        elif model == "correlatedclassificationconvnet":
            network_class = CorrelatedClassificationConvNet
        elif model == "datadrivengaussconv":
            network_class = bnn_priors.models.DataDrivenGaussianClassificationConvNet
        elif model == "datadrivendoublegammaconv":
            network_class = bnn_priors.models.DataDrivenDoubleGammaClassificationConvNet
        else:
            raise ValueError("model="+model)
        net = network_class(in_channels, img_height, y_train.max()+1, width, depth, softmax_temp=1.,
                        prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                        prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                        weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params).to(x_train)
    elif model == "resnet18":
        net = PreActResNet18(prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                            bn=batchnorm, softmax_temp=1., weight_prior_params=weight_prior_params,
                            bias_prior_params=bias_prior_params).to(x_train)
    elif model == "vwidth_resnet18":
        net = PreActResNet18(width=width, prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                            bn=batchnorm, softmax_temp=1., weight_prior_params=weight_prior_params,
                            bias_prior_params=bias_prior_params).to(x_train)
    elif model == "datadriven_resnet18":
        net = bnn_priors.models.DataDrivenPreActResNet18(
            prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=None,
            bn=batchnorm, softmax_temp=1., weight_prior_params=weight_prior_params,
            bias_prior_params=bias_prior_params).to(x_train)
    elif model == "vwidth_datadriven_resnet18":
        net = bnn_priors.models.DataDrivenPreActResNet18(
            width=width,
            prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=None,
            bn=batchnorm, softmax_temp=1., weight_prior_params=weight_prior_params,
            bias_prior_params=bias_prior_params).to(x_train)
    elif model == "thin_resnet18":
        net = ThinPreActResNet18(prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                            bn=batchnorm, softmax_temp=1., weight_prior_params=weight_prior_params,
                            bias_prior_params=bias_prior_params).to(x_train)
    elif model == "resnet34":
        net = PreActResNet34(prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                            bn=batchnorm, softmax_temp=1., weight_prior_params=weight_prior_params,
                            bias_prior_params=bias_prior_params).to(x_train)
    elif model == "googleresnet":
        net = ResNet(prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale, depth=20,
                     prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                     bn=batchnorm, softmax_temp=1., weight_prior_params=weight_prior_params,
                     bias_prior_params=bias_prior_params).to(x_train)
    elif model == "datadriven_mvt_googleresnet":
        net = bnn_priors.models.DataDrivenMVTGoogleResNet(
            softmax_temp=1., depth=20, num_classes=10, bn=batchnorm).to(x_train)
    elif model == "decreasing_mvt_googleresnet":
        net = bnn_priors.models.DecreasingMVTGoogleResNet(
            prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
            depth=20, prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale,
            scaling_fn=scaling_fn, bn=batchnorm, softmax_temp=1.,
            weight_prior_params=weight_prior_params,
            bias_prior_params=bias_prior_params).to(x_train)
    elif model == "correlatedgoogleresnet":
        # TODO: currently doesn't pass a specific prior to the dense layer, so it uses the default (Normal) one
        net = CorrelatedResNet(prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale, depth=20,
                     prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                     bn=batchnorm, softmax_temp=1., weight_prior_params=weight_prior_params,
                    bias_prior_params=bias_prior_params).to(x_train)
    elif model == "linear":
        net = bnn_priors.models.LinearRegression(
            x_train.size(-1), y_train.size(-1), noise_std=0.5,
            prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
            weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params).to(x_train)
    elif model == "logistic":
        net = bnn_priors.models.LogisticRegression(
            x_train.size(-1), y_train.size(-1), softmax_temp=1.,
            prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
            weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params).to(x_train)
    elif model == "raob_linear":
        net = bnn_priors.models.RaoBLinearRegression(x_train, y_train, noise_std=0.5)
    elif model == "test_gaussian":
        net = bnn_priors.models.GaussianModel(N=1, D=100)
    else:
        raise ValueError("model="+model)

    if x_train.device != t.device("cpu"):
        # For some reason, this increases GPU utilization and decreases CPU
        # utilization. The end result is much faster.
        the_net = t.nn.DataParallel(net.net)
    else:
        the_net = DummyModule(net.net)
    del net.net
    net.net = the_net
    return net


def _n_samples_dict(samples):
    n_samples = min(len(v) for _, v in samples.items())

    if not all((len(v) == n_samples) for _, v in samples.items()):
        warnings.warn("Not all samples have the same length. "
                      "Setting n_samples to the minimum.")
    return n_samples

def sample_iter(samples):
    for i in range(_n_samples_dict(samples)):
        yield dict((k, v[i]) for k, v in samples.items())


def evaluate_model(model: bnn_priors.models.AbstractModel,
                   dataloader_test: Iterable[Tuple[t.Tensor, t.Tensor]],
                   samples: Dict[str, t.Tensor],
                   likelihood_eval: bool, accuracy_eval: bool, calibration_eval: bool):
    if hasattr(dataloader_test.dataset, "tensors"):
        labels = dataloader_test.dataset.tensors[1].cpu()
    elif hasattr(dataloader_test.dataset, "targets"):
        labels = t.tensor(dataloader_test.dataset.targets).cpu()
    else:
        raise ValueError("I cannot find the labels in the dataloader.")
    N, *possibly_D = labels.shape
    E = _n_samples_dict(samples)

    if len(possibly_D) == 0:
        n_classes_or_funs = labels.max().item() + 1
    else:
        n_classes_or_funs, = possibly_D

    lps = t.zeros((E, N), dtype=t.float64, device='cpu')
    acc_data = t.zeros((E, N, n_classes_or_funs), dtype=t.float64, device='cpu')

    device = next(iter(model.parameters())).device

    for sample_i, sample in enumerate(sample_iter(samples)):
        with t.no_grad():
            model.load_state_dict(sample)
            i = 0
            for batch_x, batch_y in dataloader_test:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                if isinstance(preds, t.distributions.Categorical):
                    acc_data_batch = preds.logits # B,n_classes
                    lps_batch = preds.log_prob(batch_y)
                elif isinstance(preds, t.distributions.Normal):
                    acc_data_batch = preds.mean  # B,n_latent
                    lps_batch = preds.log_prob(batch_y).sum(-1)
                else:
                    raise ValueError(f"unknown likelihood {type(preds)}")

                if calibration_eval:
                    if not isinstance(preds, t.distributions.Categorical):
                        raise ValueError("Cannot calculate calibration metrics "
                                         f"for predictions of type {type(preds)}")

                next_i = i+len(batch_x)
                lps[sample_i, i:next_i] = lps_batch.detach()
                acc_data[sample_i, i:next_i, :] = acc_data_batch.detach()
                i = next_i

    def _log_space_mean(tensor, dim):
        return tensor.logsumexp(dim) - math.log(tensor.size(dim))

    # lps: n_samples,len(dataset)
    lps_each_model = lps.mean(1)
    lp_ensemble = _log_space_mean(lps, 0).mean()

    # acc_data: n_samples,len(dataset), (n_classes or n_outputs)
    if isinstance(preds, t.distributions.Categorical):
        ensemble_preds = t.distributions.Categorical(logits=_log_space_mean(acc_data, 0))
        last_preds = t.distributions.Categorical(logits=acc_data[-1])

        assert t.allclose(ensemble_preds.log_prob(labels).mean(0),
                          lp_ensemble)
        assert t.allclose(last_preds.log_prob(labels).mean(0), lps_each_model[-1])

    elif isinstance(preds, t.distributions.Normal):
        ensemble_preds = t.distributions.Normal(acc_data.mean(0), t.ones_like(acc_data[0]))
        last_preds = t.distributions.Normal(acc_data[-1], ensemble_preds.scale)

    acc_or_mse_ensemble = model.acc_mse(ensemble_preds, labels).mean(0)
    acc_or_mse_last = model.acc_mse(last_preds, labels).mean(0)

    if calibration_eval:
        probs_mean = ensemble_preds.probs.numpy()
        eces = ece(labels, probs_mean)
        aces = ace(labels, probs_mean)
        rmsces = rmsce(labels, probs_mean)
    
    results = {}
    if likelihood_eval:
        results["lp_ensemble"] = lp_ensemble.item()
        results["lp_last"] = lps_each_model[-1].item()
    if accuracy_eval:
        results["acc_ensemble"] = acc_or_mse_ensemble.item()
        results["acc_last"] = acc_or_mse_last.item()
    if calibration_eval:
        results["ece"] = eces.mean().item()
        results["ace"] = aces.mean().item()
        results["rmsce"] = rmsces.mean().item()
    return results


def evaluate_ood(model, dataloader_train, dataloader_test, samples):
    loaders = {"train": dataloader_train, "eval": dataloader_test}
    probs = {"train": [], "eval": []}
    aurocs = []
    auprcs = []
    device = next(iter(model.parameters())).device

    for sample in sample_iter(samples):
        with t.no_grad():
            model.load_state_dict(sample)
            for dataset in ["train", "eval"]:
                probs_sample = []
                for batch_x, _ in loaders[dataset]:
                    batch_x = batch_x.to(device)
                    pred = model(batch_x)
                    # shape: len(batch) x n_classes
                    probs_sample.append(pred.probs.cpu().numpy())
                # shape: len(dataset) x n_classes
                probs_sample = np.concatenate(probs_sample, axis=0)
                probs[dataset].append(probs_sample)

    for dataset in ["train", "eval"]:
        # axis=0 -> over samples of the model
        probs[dataset] = np.mean(probs[dataset], axis=0)
        # axis=-1 -> over class probabilities
        probs[dataset] = np.max(probs[dataset], axis=-1)
        
    labels = np.concatenate([np.ones_like(probs["train"]), np.zeros_like(probs["eval"])])
    probs_cat = np.concatenate([probs["train"], probs["eval"]])
    auroc = roc_auc_score(labels, probs_cat)
    auprc = average_precision_score(labels, probs_cat)
    
    results = {}

    results["auroc"] = float(auroc)
    results["auprc"] = float(auprc)
    
    return results


def evaluate_marglik(model, train_samples, eval_samples):
    log_priors = []

    n_samples = _n_samples_dict(train_samples)
    assert n_samples == _n_samples_dict(eval_samples)
    for train_sample, eval_sample in zip(sample_iter(train_samples),
                                         sample_iter(eval_samples)):
        # Ideally we would only use eval_sample, but the (possibly hierarchical)
        # model has more parameters than eval_sample. This way we start with
        # parameters in `train_sample` and overwrite them with all parameters
        # that are in `eval_sample`
        sampled_state_dict = {**train_sample, **eval_sample}
        with t.no_grad():
            model.load_state_dict(sampled_state_dict)
            log_prior = model.log_prior().item()
            log_priors.append(log_prior)
        
    log_priors = t.tensor(log_priors)
    
    results = {}
    results["simple_logmarglik"] = log_priors.logsumexp(dim=0).item() - math.log(n_samples)
    results["mean_loglik"] = log_priors.mean().item()
    results["simple_marglik"] = log_priors.exp().mean().item()
    return results


class HDF5ModelSaver:
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.chunk_size = 1
        self.__i = 0
        self.__init_dsets = True

    def __enter__(self):
        self.f = h5py.File(self.path, self.mode, libver="latest",
                           rdcc_nbytes=0)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()  # Make sure to save all in-memory cache before closing
        self.f.close()

    def add_state_dict(self, state_dict, step):
        d = {k: v.cpu().detach().unsqueeze(0).numpy()
             for (k, v) in state_dict.items()}
        d["steps"] = np.array([step], dtype=np.int64)
        d["timestamps"] = np.array([time.time()], dtype=np.float64)
        self._extend_dict(d)

    def _extend_dict(self, d):
        length = self._assign_dict_current(d)
        self.__i += length

    def _assign_dict_current(self, d) -> int:
        """
       Assigns all data from dictionary `d` to the positions
        `self.__i:self.__i+self.chunk_size` in the HDF5 file.
        """
        if self.__init_dsets:
            for k, v in d.items():
                self._create_dset(k, v.shape[1:], v.dtype)
            # Set the h5 file to Single Reader Multiple Writer (SWMR) mode,
            # the metrics can be read during the run.
            # No more datasets can be created after that.
            # This will prevent any new keys from being added using `add_scalar`.
            self.f.swmr_mode = True
            self.__init_dsets = False

        i = self.__i
        length = None
        for k, value in d.items():
            if length is None:
                length = len(value)
            else:
                assert length == len(value), "lengths unequal"

            dset = self.f[k]
            if i + length >= len(dset):
                dset.resize(i + length, axis=0)
            dset[i:i+length] = value
        assert length is not None
        return length

    def _create_dset(self, name, shape, dtype):
        # int64 stores NaN as -2**63
        assert dtype in [np.float32, np.float64, np.int64], "accepts NaNs"

        return self.f.create_dataset(
            name, dtype=dtype,
            shape=(0,                *shape),
            chunks=(self.chunk_size, *shape),
            maxshape=(None,          *shape),
            fletcher32=True, fillvalue=np.nan)

    def flush(self):
        self.f.flush()

    def load_samples(self, idx=slice(None), keep_steps=True):
        try:
            self.f.flush()
        except:
            pass
        return load_samples(self.path, idx=idx, keep_steps=keep_steps)


class HDF5Metrics(HDF5ModelSaver):
    def __init__(self, path, mode, chunk_size=8*1024):
        super().__init__(path, mode)
        self.chunk_size = chunk_size
        self._step = -2**63
        self._cache = {}
        self._chunk_i = -1
        self.last_flush = time.time()

    def add_scalar(self, name, value, step, dtype=None):
        if step > self._step:
            self._chunk_i += 1
            if self._chunk_i >= self.chunk_size:
                self._extend_dict(self._cache)
                self._chunk_i = 0
                self._scrub_cache()

            self._step = step
            self._append("steps", step, np.int64)
            self._append("timestamps", time.time(), np.float64)

        elif step < self._step:
            raise ValueError(f"step went backwards ({self._step} -> {step})")
        self._append(name, value, np.float64)

    def _scrub_cache(self):
        for v in self._cache.values():
            v[:] = np.nan

    def _append(self, name, value, dtype):
        try:
            arr = self._cache[name]
        except KeyError:
            arr = self._cache[name] = np.empty(self.chunk_size, dtype=dtype)
            arr[:] = np.nan
        arr[self._chunk_i] = value

    def flush(self, every_s=0):
        "flush every `every_s` seconds"
        if self._chunk_i < 0:
            return  # Nothing to flush

        now = time.time()
        if every_s <= 0 or now - self.last_flush > every_s:
            self.last_flush = now
            trimmed_cache = {k: v[:self._chunk_i+1] for k, v in self._cache.items()}
            self._assign_dict_current(trimmed_cache)
            self.f.flush()


def load_samples(path, idx=slice(None), keep_steps=True):
    try:
        with h5py.File(path, "r", swmr=True) as f:
            if keep_steps:
                return {k: t.from_numpy(np.asarray(v[idx]))
                        for k, v in f.items()}
            else:
                return {k: t.from_numpy(np.asarray(v[idx]))
                        for k, v in f.items()
                        if k not in ["steps", "timestamps"]}
    except OSError:
        samples = t.load(path)
        return {k: v[idx] for k, v in samples.items()}


def sneaky_artifact(_run, name):
    """modifed `artifact_event` from `sacred.observers.FileStorageObserver`
    Returns path to the name.
    """
    obs = _run.observers[0]
    assert isinstance(obs, sacred.observers.FileStorageObserver)
    obs.run_entry["artifacts"].append(name)
    obs.save_json(obs.run_entry, "run.json")
    return Path(obs.dir)/name


def reject_samples_(samples, metrics_file):
    try:
        rejected_arr = metrics_file["acceptance/rejected"][is_sample]
    except KeyError:
        return samples
    is_sample = (metrics_file["acceptance/is_sample"][:] == 1)
    assert np.all((rejected_arr == 0) | (rejected_arr == 1))
    metrics_steps = metrics_file["steps"][is_sample]
    rejected = {int(s): bool(r) for (s, r) in zip(metrics_steps, rejected_arr)}

    for i in range(_n_samples_dict(samples)):
        sample_step = samples["steps"][i].item()
        if rejected[sample_step]:
            for k in samples.keys():
                samples[k][i] = samples[k][i-1]
    return samples
