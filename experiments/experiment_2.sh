#!/bin/bash

priors=( gaussian uniform laplace student-t cauchy improper gaussian_gamma gaussian_uniform horseshoe )
inference=( SGLD )
scales=( 0.14 1.41 14.1 )
temps=( 0.0 0.1 1.0 )

for prior in "${priors[@]}"
do
    for scale in "${scales[@]}"
    do
        for temp in "${temps[@]}"
        do

           bsub -n 2 -W 24:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=cifar10 inference=SGLD model=resnet18 warmup=30 burnin=10 skip=1 n_samples=100 lr=0.01 weight_scale=$scale cycles=20 batch_size=64 temperature=$temp save_samples=True"
        done
    done
done
