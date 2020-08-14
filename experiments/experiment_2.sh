#!/bin/bash

priors=( gaussian uniform laplace student-t lognormal cauchy )
inference=( SGLD )
scales=( 0.70 1.41 2.83 )
temps=( 1.0 0.1 0.01 )

for prior in "${priors[@]}"
do
    for scale in "${scales[@]}"
    do
        for temp in "${temps[@]}"
        do

           bsub -n 2 -W 4:00 -J "bnn" -sp 40 -g /vincent/experiments/bnn -R "rusage[mem=4000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=cifar10 inference=SGLD model=resnet18 warmup=20 burnin=10 skip=1 n_samples=20 weight_scale=$scale cycles=5 batch_size=64 temperature=$temp"
        done
    done
done
