import math
import h5py
import numpy as np
import time
from sklearn.metrics import average_precision_score, roc_auc_score
import torch as t
from bnn_priors.data import UCI, CIFAR10, CIFAR10_C, MNIST, RotatedMNIST, FashionMNIST, SVHN, RandomData
from bnn_priors.models import RaoBDenseNet, DenseNet, PreActResNet18, PreActResNet34, ClassificationDenseNet, ResNet
from bnn_priors.prior import LogNormal
from bnn_priors import prior
from bnn_priors.prior import get_prior
from bnn_priors.third_party.calibration_error import ece, ace, rmsce
import warnings


def device(device):
    if device == "try_cuda":
        if t.cuda.is_available():
            return t.device("cuda:0")
        else:
            return t.device("cpu")
    return t.device(device)


def get_data(data, device):
    assert (data[:3] == "UCI" or data[:7] == "cifar10" or data[-5:] == "mnist"
            or data in ["svhn", "random"]), f"Unknown data set {data}"
    if data[:3] == "UCI":
        uci_dataset = data.split("_")[1]
        assert uci_dataset in ["boston", "concrete", "energy", "kin8nm",
                               "naval", "power", "protein", "wine", "yacht"]
        # TODO: do we ever use a different split than 0?
        dataset = UCI(uci_dataset, 0, device=device)
    elif data[:8] == "cifar10c":
        corruption = data.split("-")[1]
        dataset = CIFAR10_C(corruption, device=device)
    elif data == "cifar10":
        dataset = CIFAR10(device=device)
    elif data == "mnist":
        dataset = MNIST(device=device)
    elif data == "rotated_mnist":
        dataset = RotatedMNIST(device=device)
    elif data == "fashion_mnist":
        dataset = FashionMNIST(device=device)
    elif data == "svhn":
        dataset = SVHN(device=device)
    elif data == "random":
        dataset = RandomData(device=device)
    return dataset


def he_initialize(model):
    for name, param in model.named_parameters():
        if "weight_prior.p" in name:
            t.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
        elif "bias_prior.p" in name:
            t.nn.init.zeros_(param.data)


def get_model(x_train, y_train, model, width, depth, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm,
             weight_prior_params, bias_prior_params):
    assert model in ["densenet", "raobdensenet", "resnet18", "resnet34",
                     "classificationdensenet", "test_gaussian", "googleresnet"]
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
    elif model == "resnet18":
        net = PreActResNet18(prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
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
    elif model == "test_gaussian":
        from testing.test_sgld import GaussianModel
        net = GaussianModel(N=1, D=100)
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


def evaluate_model(model, dataloader_test, samples, eval_data, likelihood_eval,
                   accuracy_eval, calibration_eval):
    lps = []
    accs = []
    probs = []

    for sample in sample_iter(samples):
        with t.no_grad():
            model.load_state_dict(sample)
            lps_sample = []
            accs_sample = []
            probs_sample = []
            for batch_x, batch_y in dataloader_test:
                pred = model(batch_x)
                lps_batch = pred.log_prob(batch_y)
                if "cifar10" in eval_data or "mnist" in eval_data:
                    # shape: batch_size
                    accs_batch = (t.argmax(pred.probs, dim=1) == batch_y).float()
                else:
                    accs_batch = (pred.mean - batch_y)**2.  # shape: batch_size
                if calibration_eval:
                    probs_batch = pred.probs
                else:
                    probs_batch = t.tensor([])
                lps_sample.extend(list(lps_batch.cpu().numpy()))
                accs_sample.extend(list(accs_batch.cpu().numpy()))
                probs_sample.extend(list(probs_batch.cpu().numpy()))
            lps.append(lps_sample)
            accs.append(accs_sample)
            probs.append(probs_sample)
            
    lps = t.tensor(lps, dtype=t.float64)  # n_samples,len(dataset)
    lps = lps.logsumexp(1) - math.log(lps.size(1))
    accs = t.tensor(accs, dtype=t.float64)
    accs = accs.mean(dim=1)
    
    if calibration_eval:
        labels = dataloader_test.dataset.tensors[1].cpu().numpy()
        probs_mean = t.tensor(probs).mean(dim=0)
        eces = ece(labels, probs_mean)
        aces = ace(labels, probs_mean)
        rmsces = rmsce(labels, probs_mean)
    
    results = {}
    if likelihood_eval:
        results["lp_ensemble"] = lps.logsumexp(0).item() - math.log(len(lps))
        # lps.exp().square().mean(0).sqrt().log() / math.sqrt(len(lps))
        results["lp_ensemble_std"] = lps.mul(2.).logsumexp(0).div(2.).item()
        results["lp_ensemble_stderr"] = results["lp_ensemble_std"] / math.sqrt(len(lps))
        results["lp_mean"] =  lps.mean().item()
        results["lp_std"] =  lps.std().item()
        results["lp_stderr"] = lps.std().item() / math.sqrt(len(lps))
        results["lp_last"] = lps[-1].item()
    if accuracy_eval:
        results["acc_mean"] = accs.mean().item()
        results["acc_std"] =  accs.std().item()
        results["acc_stderr"] = accs.std().item() / math.sqrt(len(accs))
        results["acc_last"] = accs[-1].item()
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

    for sample in sample_iter(samples):
        with t.no_grad():
            model.load_state_dict(sample)
            for dataset in ["train", "eval"]:
                probs_sample = []
                for batch_x, batch_y in loaders[dataset]:
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


class HDF5Metrics:
    def __init__(self, path, mode, chunk_size=1024, chunks_in_cache=8*1024):
        self.path = path
        self.mode = mode
        # defaults: 1 KB chunks, 8 MB cache.
        self.chunk_size = chunk_size
        self.chunks_in_cache = chunks_in_cache
        self._i = -1
        self._step = -1
        self.last_flush = time.time()

    def __enter__(self):
        self.f = h5py.File(self.path, self.mode, libver="latest",
                           rdcc_nbytes=self.chunks_in_cache * self.chunk_size)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()

    def add_scalar(self, name, value, step, dtype=None):
        if step > self._step:
            self._step = step
            self._i += 1
            self._append("steps", step, dtype=np.int64)
            self._append("timestamps", time.time(), dtype=np.float64)
            if self._i == 1:
                # Set the h5 file to Single Reader Multiple Writer (SWMR) mode,
                # the metrics can be read during the run.
                # No more datasets can be created after that.
                # This will prevent any new keys from being added using `add_scalar`.
                self.f.swmr_mode = True

        elif step < self._step:
            raise ValueError(f"step went backwards ({self._step} -> {step})")
        self._append(name, value, dtype=dtype)

    def _append(self, name, value, dtype=None):
        try:
            dset = self.f[name]
        except KeyError:
            if isinstance(value, np.ndarray):
                if dtype is None:
                    dtype = value.dtype
                else:
                    assert dtype == value.dtype
                shape = value.shape
            elif isinstance(value, t.Tensor):
                np_dtype = np.dtype(str(value.dtype).lstrip("torch."))
                if dtype is None:
                    dtype = np_dtype
                else:
                    assert dtype == np_dtype
                shape = tuple(value.size())
            else:
                if dtype is None:
                    dtype = type(value)
                shape = ()
            dset = self._create_metric(name, shape, dtype)

        assert isinstance(dset, h5py.Dataset)
        if self._i >= len(dset):
            # If they are all `_append`ed to in every iteration,
            # `dset`s sizes always are within `self.chunk_size` of each other
            dset.resize(self._i+self.chunk_size, axis=0)
        dset[self._i] = value

    def _create_metric(self, name, shape, dtype):
        return self.f.create_dataset(
            name, dtype=dtype,
            shape=(0,                *shape),
            chunks=(self.chunk_size, *shape),
            maxshape=(None,          *shape),
            fletcher32=True, fillvalue=np.nan)

    def flush(self, every_s=0):
        "flush every `every_s` seconds"
        now = time.time()
        if now - self.last_flush > every_s:
            self.last_flush = now
        return self.f.flush()


class HDF5ModelSaver(HDF5Metrics):
    def __init__(self, path, mode):
        super().__init__(path, mode, chunk_size=1, chunks_in_cache=0)

    def add_state_dict(self, state_dict, step):
        for k, v in state_dict.items():
            self.add_scalar(k, v.cpu().detach().numpy(), step, dtype=None)

    def load_samples(self):
        try:
            self.flush()
        except:
            pass
        return load_samples(self.path)


def load_samples(path, idx=slice(None)):
    try:
        with h5py.File(path, "r", swmr=True) as f:
            return {k: t.from_numpy(v[idx])
                    for k, v in f.items()
                    if k not in ["steps", "timestamps"]}
    except OSError:
        samples = t.load(path)
        return {k: v[idx] for k, v in samples.items()}
