#!/bin/bash

exp_dir="../results/200830_mnist"

config_files=($(ls $exp_dir/*/config.json))

for conf_file in ${config_files[@]}
do
    bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file eval_data=fashion_mnist ood_eval=True"
done
