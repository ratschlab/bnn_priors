#!/usr/bin/env bash
set -euo pipefail

logdir='/scratches/huygens/ag919/logs/200930_linear'
# priors_data=( gaussian )
#priors_data=( gaussian laplace student-t cauchy gennorm )
priors_model=( gaussian laplace student-t cauchy gennorm mixture improper )
scales=( 1.41 )
temps=( 1.0 )

# prior_data comes from environment
CUDA_VISIBLE_DEVICES=""

for aaa in 1 2 3
do
    for inference in VerletSGLDReject HMCReject
    do
        for prior_model in "${priors_model[@]}"
        do
            for scale in "${scales[@]}"
            do
                for temp in "${temps[@]}"
                do
                    python train_bnn.py with weight_prior=$prior_model data=synthetic.random.$prior_data inference=$inference model=linear burnin=0 warmup=10 \
                        metrics_skip=999 sampling_decay=flat precond_update=None reject_samples=True device=cpu \
                        skip=1 n_samples=1000 lr=0.01 weight_scale=$scale cycles=1000 temperature=$temp save_samples=True log_dir=$logdir
                done
            done
        done
    done
done

# workon py37; cd Programacio/BNN-priors/experiments; prior_data=laplace CUDA_VISIBLE_DEVICES="" bash experiment_synthetic.sh
