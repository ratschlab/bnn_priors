#!/bin/bash

logdir='../results/200904_mnist'
priors=( gaussian uniform laplace student-t cauchy improper gaussian_gamma gaussian_uniform horseshoe laplace_gamma laplace_uniform student-t_gamma student-t_uniform mixture )
scales=( 0.7 1.41 2.41 )
temps=( 0.0 0.1 1.0 )
mixtures=( "g_l_s_c" "gg_lg_sg_h" "g_l_s_c_gg_lg_sg_h" )
dfs=( 2 3 4 )

for prior in "${priors[@]}"
do
    for scale in "${scales[@]}"
    do
        for temp in "${temps[@]}"
        do
            if [ $prior = "student-t" ]; then
                for df in "${dfs[@]}"
                do
                    bsub -n 2 -W 24:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=mnist inference=SGLD model=classificationdensenet width=100 warmup=30 burnin=10 skip=1 n_samples=100 lr=0.01 weight_scale=$scale cycles=20 batch_size=64 temperature=$temp save_samples=True log_dir=$logdir weight_prior_params={"'"'"df"'"'":$df}"
                done
            elif [ $prior = "mixture" ]; then
                for mix in "${mixtures[@]}"
                do
                    bsub -n 2 -W 24:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=mnist inference=SGLD model=classificationdensenet width=100 warmup=30 burnin=10 skip=1 n_samples=100 lr=0.01 weight_scale=$scale cycles=20 batch_size=64 temperature=$temp save_samples=True log_dir=$logdir weight_prior_params={"'"'"components"'"'":"'"'"$mix"'"'"}"
                done
            else
                bsub -n 2 -W 24:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=mnist inference=SGLD model=classificationdensenet width=100 warmup=30 burnin=10 skip=1 n_samples=100 lr=0.01 weight_scale=$scale cycles=20 batch_size=64 temperature=$temp save_samples=True log_dir=$logdir"
            fi
        done
    done
done
