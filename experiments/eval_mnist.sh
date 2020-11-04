#!/bin/bash

exp_dir="../results/201102_mnist_cnn"
eval_samples=10
skip=30

config_files=($(ls $exp_dir/*/config.json))

for conf_file in ${config_files[@]}
do
    bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file skip_first=$skip"

    bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file eval_data=rotated_mnist calibration_eval=True skip_first=$skip"

    bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file eval_data=fashion_mnist ood_eval=True skip_first=$skip"

    # bsub -n 2 -W 2:00 -J "bnn" -sp 40 -g /vincent/analysis -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python eval_bnn.py with config_file=$conf_file marglik_eval=True eval_samples=$exp_dir/$eval_samples/samples.pt skip_first=$skip"
done
