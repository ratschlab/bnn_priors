#!/bin/bash

exp_dir="../results/200831_cifar10"

config_files=($(ls $exp_dir/*/config.json))
corruptions=( fog jpeg_compression zoom_blur speckle_noise glass_blur spatter shot_noise defocus_blur elastic_transform gaussian_blur frost saturate brightness snow gaussian_noise motion_blur contrast impulse_noise pixelate )

for conf_file in ${config_files[@]}
do
   
    bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file"
    for corruption in ${corruptions[@]}
    do
        bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file eval_data=cifar10c-$corruption calibration_eval=True ood_eval=True"
    done
done
