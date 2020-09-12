#!/bin/bash

logdir='../results/200912_synthetic'
priors_data=( gaussian laplace student-t cauchy gennorm )
priors_model=( gaussian laplace student-t cauchy gennorm mixture )
scales=( 1.41 )
temps=( 0.0 1.0 )

for prior_data in "${priors_data[@]}"
do
    for prior_model in "${priors_model[@]}"
    do
        for scale in "${scales[@]}"
        do
            for temp in "${temps[@]}"
            do
                bsub -n 2 -W 24:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior_model data=synthetic.random.$prior_data inference=SGLD model=densenet width=50 warmup=1000 burnin=200 skip=20 n_samples=100 lr=1e-3 weight_scale=$scale cycles=20 temperature=$temp save_samples=True log_dir=$logdir"
            done
        done
    done
done
