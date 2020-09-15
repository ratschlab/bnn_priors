#!/bin/bash

logdir='../results/200910_uci_test'
priors=( gaussian uniform laplace student-t cauchy improper gaussian_gamma gaussian_uniform horseshoe laplace_gamma laplace_uniform student-t_gamma student-t_uniform mixture )
datasets=( boston wine energy naval concrete kin8nm power yacht protein )
inference=( SGLD )  # add HMC if needed
scales=( 0.71 1.41 2.82 )
temps=( 0.0 0.1 1.0 )

for prior in "${priors[@]}"
do
    for scale in "${scales[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for inf in "${inference[@]}"
            do
                for temp in "${temps[@]}"
                do
                    #bsub -n 2 -W 24:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=4000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=UCI_$dataset inference=$inf warmup=5000 burnin=1000 weight_scale=$scale cycles=20 n_samples=100 skip=100 temperature=$temp log_dir=$logdir"
                    bsub -n 2 -W 48:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=4000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=UCI_$dataset inference=$inf warmup=500 burnin=100 weight_scale=$scale cycles=1 n_samples=50 save_samples=True skip=10 lr=1e-4 temperature=$temp log_dir=$logdir"
                done
            done
        done
    done
done
