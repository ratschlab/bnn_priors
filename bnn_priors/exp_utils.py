import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch as t
from bnn_priors.data import UCI, CIFAR10, CIFAR10_C, MNIST, RotatedMNIST, FashionMNIST, SVHN, RandomData
from bnn_priors.models import RaoBDenseNet, DenseNet, PreActResNet18, PreActResNet34, ClassificationDenseNet
from bnn_priors.prior import LogNormal
from bnn_priors import prior
from bnn_priors.prior import get_prior
from bnn_priors.third_party.calibration_error import ece, ace, rmsce


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
    assert model in ["densenet", "raobdensenet", "resnet18", "resnet34", "classificationdensenet", "test_gaussian"]
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
    elif model == "test_gaussian":
        from testing.test_sgld import GaussianModel
        net = GaussianModel(N=1, D=100)
    return net


def evaluate_model(model, dataloader_test, samples, n_samples,
                   eval_data, likelihood_eval, accuracy_eval, calibration_eval):
    lps = []
    accs = []
    probs = []

    for i in range(n_samples):
        sample = dict((k, v[i]) for k, v in samples.items())
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


def evaluate_ood(model, dataloader_train, dataloader_test, samples, n_samples):

    loaders = {"train": dataloader_train, "eval": dataloader_test}
    probs = {"train": [], "eval": []}
    aurocs = []
    auprcs = []

    for i in range(n_samples):
        sample = dict((k, v[i]) for k, v in samples.items())
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


def evaluate_marglik(model, train_samples, eval_samples, n_samples):
    log_priors = []

    for i in range(n_samples):
        train_sample = dict((k, v[i]) for k, v in train_samples.items())
        eval_sample = dict((k, v[i]) for k, v in eval_samples.items())
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
