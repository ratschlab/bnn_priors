#!/bin/bash

logdir='../results/200907_cifar_nobatchnorm'
priors=( gaussian uniform laplace student-t cauchy improper gaussian_gamma gaussian_uniform horseshoe laplace_gamma laplace_uniform student-t_gamma student-t_uniform mixture )
scales=( 0.7 1.41 2.41 )
temps=( 0.0 0.1 1.0 2.0 )

for prior in "${priors[@]}"
do
    for scale in "${scales[@]}"
    do
        for temp in "${temps[@]}"
        do

           bsub -n 2 -W 48:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=16000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=cifar10 inference=SGLD model=resnet18 warmup=30 burnin=10 skip=1 n_samples=100 lr=0.01 weight_scale=$scale cycles=20 batch_size=128 temperature=$temp save_samples=True log_dir=$logdir batchnorm=False"
        done
    done
done
