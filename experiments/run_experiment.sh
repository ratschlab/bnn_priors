#!/bin/bash

logdir='../results/my_exp_cifar'
data="cifar10"
model="googleresnet"

priors=( improper gaussian laplace student-t convcorrnormal )
scales=( 1.41 )
temps=( 0.001 0.01 0.03 0.1 0.3 1.0 )
lr=0.01
num_cycles=60
lengthscales=( 0.5 1.0 )


for i in {1..5} # 5 replicates for the error bars
do
    for prior in "${priors[@]}"
    do
        for scale in "${scales[@]}"
        do
            for temp in "${temps[@]}"
            do
                if [[ $prior == "convcorrnormal" ]]; then
                    for lengthscale in "${lengthscales[@]}"
                    do
                        python train_bnn.py with weight_prior=$prior data=$data inference=VerletSGLDReject model=correlated$model warmup=45 burnin=0 skip=1 n_samples=300 lr=$lr momentum=0.994 weight_scale=$scale cycles=$num_cycles batch_size=128 temperature=$temp save_samples=True progressbar=False log_dir=$logdir batchnorm=True weight_prior_params.lengthscale=$lengthscale
                    done
                else
                    python train_bnn.py with weight_prior=$prior data=$data inference=VerletSGLDReject model=$model warmup=45 burnin=0 skip=1 n_samples=300 lr=$lr momentum=0.994 weight_scale=$scale cycles=$num_cycles batch_size=128 temperature=$temp save_samples=True progressbar=False log_dir=$logdir batchnorm=True
                fi
            done
        done
    done
done
