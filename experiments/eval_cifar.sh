#!/bin/bash

exp_dir="../results/201126_cifar"
eval_samples=10
skip=30

config_files=($(ls $exp_dir/*/config.json))
# corruptions=( fog jpeg_compression zoom_blur speckle_noise glass_blur spatter shot_noise defocus_blur elastic_transform gaussian_blur frost saturate brightness snow gaussian_noise motion_blur contrast impulse_noise pixelate )
corruptions=( gaussian_blur pixelate )

for conf_file in ${config_files[@]}
do
   
    bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file skip_first=$skip"
    bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file ood_eval=True eval_data=svhn skip_first=$skip"
    for corruption in ${corruptions[@]}
    do
        bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file eval_data=cifar10c-$corruption calibration_eval=True ood_eval=True skip_first=$skip"
    done
    # bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file marglik_eval=True eval_samples=$exp_dir/$eval_samples/samples.pt skip_first=$skip"
done
