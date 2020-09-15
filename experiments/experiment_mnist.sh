#!/bin/bash

logdir='../results/200911_mnist'
priors=( gaussian laplace student-t cauchy gennorm improper gaussian_empirical horseshoe laplace_empirical student-t_empirical gennorm_empirical mixture scale_mixture scale_mixture_empirical )
scales=( 1.41 ) # 1.41 2.82
temps=( 0.0 1.0 ) # 0.0 0.1 1.0
mixtures=( "g_l_s_c_gn" "ge_le_se_gne" )
dfs=( 2 3 5 )

for prior in "${priors[@]}"
do
    for scale in "${scales[@]}"
    do
        for temp in "${temps[@]}"
        do
            if [ $prior = "student-t" ]; then
                for df in "${dfs[@]}"
                do
                    bsub -n 2 -W 24:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=mnist inference=SGLD model=classificationdensenet width=100 warmup=30 burnin=10 skip=1 n_samples=100 lr=0.01 weight_scale=$scale cycles=20 batch_size=128 temperature=$temp save_samples=True log_dir=$logdir weight_prior_params={"'"'"df"'"'":$df}"
                done
            elif [ $prior = "mixture" ]; then
                for mix in "${mixtures[@]}"
                do
                    bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=16000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=mnist inference=SGLD model=classificationdensenet width=100 warmup=30 burnin=10 skip=1 n_samples=100 lr=0.01 weight_scale=$scale cycles=20 batch_size=128 temperature=$temp save_samples=True log_dir=$logdir weight_prior_params={"'"'"components"'"'":"'"'"$mix"'"'"}"
                done
            else
                bsub -n 2 -W 24:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=mnist inference=SGLD model=classificationdensenet width=100 warmup=30 burnin=10 skip=1 n_samples=100 lr=0.01 weight_scale=$scale cycles=20 batch_size=128 temperature=$temp save_samples=True log_dir=$logdir"
            fi
        done
    done
done
