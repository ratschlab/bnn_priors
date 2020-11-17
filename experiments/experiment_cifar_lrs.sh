#!/bin/bash

logdir='../results/201117_cifar'

priors=( gaussian convcorrnormal )
temps=( 0.01 1.0 )
model=googleresnet  # alternative: resnet18_thin
learning_rates=(0.1 0.03162277660168379 0.01 0.0031622776601683794 0.001 0.00031622776601683794 0.0001 "3.1622776601683795e-05")
batch_size=125
sampling_decay="flat"
load_samples="helo"

# momentum = e^(-gamma * h)
# lr = h^2 * num_data = h^2 * 50000
# log momentum = -gamma*h
# gamma = -(log momentum) / sqrt(lr/50000)
#
# Baseline settings:
# lr = 0.1
# momentum = 0.98
# -> gamma = 14.28
#
# [math.exp(-gamma * math.sqrt(lr / 50000)) for lr in 10**np.arange(-1, -5, -0.5)]
momenta=(0.98 0.988703473184311 0.9936317070771853 0.9964138398421518 0.9979817686415696 0.998864563375984 0.9993613383309885 0.9996408039418584)

for prior in "${priors[@]}"; do
    for temp in "${temps[@]}"; do
        for i in $(seq 0 "$(( "${#learning_rates[@]}" - 1))"); do
            command_common="source activate bnn; python train_bnn.py with weight_prior=$prior data=cifar10_augmented model=$model depth=20 width=16 warmup=9 burnin=0 skip=1 n_samples=100 cycles=100 temperature=$temp sampling_decay=$sampling_decay lr=${learning_rates[i]} init_method=he load_samples=$load_samples batch_size=$batch_size save_samples=True log_dir=$logdir batchnorm=True"
            bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=32000,ngpus_excl_p=1]" "$command_common inference=VerletSGLDReject momentum=${momenta[i]}"
            bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=32000,ngpus_excl_p=1]" "$command_common inference=HMCReject momentum=1.0"
        done
    done
done
