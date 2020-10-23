#!/bin/bash

logdir='../results/201002_synthetic'
priors_data=( gaussian laplace student-t cauchy gennorm )
priors_model=( gaussian laplace student-t cauchy gennorm mixture improper )
scales=( 1.41 )
temps=( 0.1 1.0 )

for i in {1..20}
do
    for prior_data in "${priors_data[@]}"
    do
        for prior_model in "${priors_model[@]}"
        do
            for scale in "${scales[@]}"
            do
                for temp in "${temps[@]}"
                do
                    #bsub -n 2 -W 24:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior_model data=synthetic.mnist.$prior_data inference=SGLD model=classificationdensenet width=100 warmup=40 burnin=10 skip=1 n_samples=200 lr=1e-3 weight_scale=$scale cycles=40 temperature=$temp batch_size=256 save_samples=True log_dir=$logdir"
                    #bsub -n 2 -W 4:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior_model data=synthetic.random.$prior_data inference=VerletSGLD model=densenet width=50 depth=1 warmup=1000 burnin=1000 skip=20 n_samples=100 lr=1e-3 weight_scale=$scale cycles=20 temperature=$temp save_samples=True log_dir=$logdir"
                    bsub -n 2 -W 4:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=8000]" "source activate bnn; python train_bnn.py with weight_prior=$prior_model data=synthetic.random.$prior_data inference=VerletSGLD model=linear warmup=2000 burnin=2000 skip=20 n_samples=100 lr=1e-2 weight_scale=$scale cycles=20 temperature=$temp save_samples=True log_dir=$logdir"
                done
            done
        done
    done
done
