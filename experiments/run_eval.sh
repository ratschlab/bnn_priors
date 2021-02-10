#!/bin/bash

exp_dir="../results/my_exp_cifar"
calibration_data="cifar10c-gaussian_blur"
ood_data="svhn"
skip=50

config_files=($(ls $exp_dir/*/config.json))

for conf_file in ${config_files[@]}
do
    python eval_bnn.py with config_file=$conf_file skip_first=$skip
    python eval_bnn.py with config_file=$conf_file eval_data=$calibration_data calibration_eval=True skip_first=$skip
    python eval_bnn.py with config_file=$conf_file eval_data=$ood_data ood_eval=True skip_first=$skip
done
