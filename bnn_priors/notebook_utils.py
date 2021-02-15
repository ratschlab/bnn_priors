from pathlib import Path
import os
import pandas as pd
import json

def flatten(in_dict):
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, dict):
            v_flat = flatten(v)
            out_dict.update({k+"."+v_k: v_v for v_k, v_v in v_flat.items()})
        else:
            out_dict[k] = v
    return out_dict

def collect_runs(base_log_dir, metrics_must_exist=True):
    base_log_dir = Path(base_log_dir)
    series_list = []

    for run in os.listdir(base_log_dir):
        if run in ["_sources", "jugdir"]:
            continue

        if metrics_must_exist and not os.path.exists(base_log_dir/run/"metrics.h5"):
            continue

        try:
            with open(base_log_dir/run/"run.json") as f:
                s2 = pd.Series(flatten(json.load(f)))
            with open(base_log_dir/run/"config.json") as f:
                s1 = pd.Series(flatten(json.load(f)))
        except FileNotFoundError:
            continue

        s2["the_dir"] = base_log_dir/run
        series_list.append(pd.concat([s1, s2]))

    return pd.DataFrame(series_list)


def unique_cols(df, blacklist=set([
     'heartbeat',
     'log_dir',
     'run_id',
     'host.cpu',
     'host.gpus.driver_version',
     'host.hostname',
     'host.python_version',
     'start_time',
     'status',
     'stop_time',
     'the_dir',
     'result.acc_last',
     'result.acc_mean',
     'result.acc_std',
     'result.acc_stderr',
     'result.lp_ensemble',
     'result.lp_ensemble_std',
     'result.lp_ensemble_stderr',
     'result.lp_last',
     'result.lp_mean',
     'result.lp_std',
     'result.lp_stderr'])):

    different_cols = []
    for col in df:
        if col in blacklist or col.startswith("meta.options") or col.startswith("result."):
            continue
        try:
            if len(df[col].unique()) > 1:
                different_cols.append(col)

        except TypeError:
            pass
    return different_cols

def json_load(path):
    with open(path, "r") as f:
        return json.load(f)

def json_dump(obj, path):
    with open(path, "w") as f:
        return json.dump(obj, f)
