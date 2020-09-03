import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch as t
from bnn_priors.data import UCI, CIFAR10, CIFAR10_C, MNIST, RotatedMNIST, FashionMNIST
from bnn_priors.models import RaoBDenseNet, DenseNet, PreActResNet18, PreActResNet34, ClassificationDenseNet
from bnn_priors.prior import LogNormal
from bnn_priors import prior
from bnn_priors.third_party.calibration_error import ece, ace, rmsce


def device(device):
    if device == "try_cuda":
        if t.cuda.is_available():
            return t.device("cuda:0")
        else:
            return t.device("cpu")
    return t.device(device)


def get_data(data, device):
    assert data[:3] == "UCI" or data[:7] == "cifar10" or data[-5:] == "mnist", f"Unknown data set {data}"
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
    return dataset


def get_prior(prior_name):
    if prior_name == "mixture":
        return prior.Mixture
    else:
        return prior.get_prior(prior_name)


def get_model(x_train, y_train, model, width, weight_prior, weight_loc,
             weight_scale, bias_prior, bias_loc, bias_scale, batchnorm,
             weight_prior_params, bias_prior_params):
    assert model in ["densenet", "raobdensenet", "resnet18", "resnet34", "classificationdensenet"]
    if weight_prior in ["cauchy"]:
        #TODO: which other distributions should use this? Laplace?
        scaling_fn = lambda std, dim: std/dim
    else:
        scaling_fn = lambda std, dim: std/dim**0.5
    weight_prior = get_prior(weight_prior)
    bias_prior = get_prior(bias_prior)
    if model == "densenet":
        net = DenseNet(x_train.size(-1), y_train.size(-1), width, noise_std=LogNormal((), -1., 0.2),
                        prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                        prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                      weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params).to(x_train)
    elif model == "raobdensenet":
        net = RaoBDenseNet(x_train, y_train, width, noise_std=LogNormal((), -1., 0.2)).to(x_train)
    elif model == "classificationdensenet":
        net = ClassificationDenseNet(x_train.size(-1), y_train.max()+1, width, softmax_temp=1.,
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
    return net


def evaluate_model(model, dataloader_test, samples, bn_params, n_samples,
                   eval_data, likelihood_eval, accuracy_eval, calibration_eval):
    lps = []
    accs = []
    probs = []

    for i in range(n_samples):
        sample = dict((k, v[i]) for k, v in samples.items())
        sampled_state_dict = {**sample, **bn_params}
        with t.no_grad():
            # TODO: get model.using_params() to work with batchnorm params
            model.load_state_dict(sampled_state_dict)
            lps_sample = []
            accs_sample = []
            probs_sample = []
            for batch_x, batch_y in dataloader_test:
                pred = model(batch_x)
                lps_batch = pred.log_prob(batch_y)
                if eval_data[:7] == "cifar10" or eval_data[-5:] == "mnist":
                    accs_batch = (t.argmax(pred.probs, dim=1) == batch_y).float()
                else:
                    accs_batch = (pred.mean - batch_y)**2.
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
            
    lps = t.tensor(lps, dtype=t.float64)
    lps = lps.logsumexp(0) - math.log(n_samples)
    accs = t.tensor(accs, dtype=t.float64)
    accs = accs.mean(dim=1)
    
    if calibration_eval:
        labels = dataloader_test.dataset.tensors[1].cpu().numpy()
        eces = t.tensor([ece(labels, probs_sample)
                         for probs_sample in probs], dtype=t.float64)
        aces = t.tensor([ace(labels, probs_sample)
                         for probs_sample in probs], dtype=t.float64)
        rmsces = t.tensor([rmsce(labels, probs_sample)
                           for probs_sample in probs], dtype=t.float64)
    
    results = {}
    if likelihood_eval:
        results["lp_mean"] = lps.mean().item()
        results["lp_std"] =  lps.std().item()
        results["lp_stderr"] = lps.std().item() / math.sqrt(len(lps))
    if accuracy_eval:
        results["acc_mean"] = accs.mean().item()
        results["acc_std"] =  accs.std().item()
        results["acc_stderr"] = accs.std().item() / math.sqrt(len(accs))
    if calibration_eval:
        results["ece"] = eces.mean().item()
        results["ace"] = aces.mean().item()
        results["rmsce"] = rmsces.mean().item()
        
    return results


def evaluate_ood(model, dataloader_train, dataloader_test, samples, bn_params, n_samples):

    loaders = {"train": dataloader_train, "eval": dataloader_test}
    probs = {"train": [], "eval": []}
    aurocs = []
    auprcs = []

    for i in range(n_samples):
        sample = dict((k, v[i]) for k, v in samples.items())
        sampled_state_dict = {**sample, **bn_params}
        with t.no_grad():
            # TODO: get model.using_params() to work with batchnorm params
            model.load_state_dict(sampled_state_dict)
            for dataset in ["train", "eval"]:
                probs_sample = []
                for batch_x, batch_y in loaders[dataset]:
                    pred = model(batch_x)
                    probs_batch, _ = pred.probs.max(dim=1)
                    probs_sample.extend(list(probs_batch.cpu().numpy()))
                probs[dataset].extend(probs_sample)
            
    
        labels = np.concatenate([np.ones_like(probs["train"]), np.zeros_like(probs["eval"])])
        probs_cat = np.concatenate([probs["train"], probs["eval"]])
        aurocs.append(roc_auc_score(labels, probs_cat))
        auprcs.append(average_precision_score(labels, probs_cat))
        
    aurocs = t.tensor(aurocs)
    auprcs = t.tensor(auprcs)
    
    results = {}

    results["auroc_mean"] = aurocs.mean().item()
    results["auroc_std"] =  aurocs.std().item()
    results["auroc_stderr"] = aurocs.std().item() / math.sqrt(len(aurocs))
    
    results["auprc_mean"] = auprcs.mean().item()
    results["auprc_std"] =  auprcs.std().item()
    results["auprc_stderr"] = auprcs.std().item() / math.sqrt(len(auprcs))
    
    return results