#!/bin/bash

logdir='../results/201126_cifar'
#priors=( gaussian laplace gennorm student-t cauchy improper gaussian_empirical horseshoe laplace_empirical student-t_empirical gennorm_empirical mixture scale_mixture scale_mixture_empirical )
#priors=( gaussian laplace gennorm student-t cauchy improper mixture scale_mixture )
#priors=( gaussian improper convcorrnormal convcorrnormal_fitted_ls )
priors=( improper gaussian laplace student-t convcorrnormal )
scales=( 1.41 ) # 1.41 2.82
temps=( 0.001 0.01 0.05 0.1 0.5 1.0 ) # 0.0 0.1 1.0
mixtures=( "g_l_s_c_gn" "ge_le_se_gne" )
dfs=( 3 )
lr=0.01
lens="../bnn_priors/models/cifar10_fitted_lengthscales.pkl.gz"
model=googleresnet

for prior in "${priors[@]}"
do
    for scale in "${scales[@]}"
    do
        for temp in "${temps[@]}"
        do
            if [ $prior = "student-t" ]; then
                for df in "${dfs[@]}"
                do
                    bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=32000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=cifar10 inference=VerletSGLD model=$model warmup=30 burnin=15 skip=1 n_samples=100 lr=$lr momentum=0.98 weight_scale=$scale cycles=20 batch_size=128 temperature=$temp save_samples=True log_dir=$logdir batchnorm=True" # weight_prior_params.df=$df"
                done
            elif [ $prior = "mixture" ]; then
                for mix in "${mixtures[@]}"
                do
                    bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=64000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=cifar10 inference=VerletSGLD model=$model warmup=30 burnin=15 skip=1 n_samples=100 lr=$lr momentum=0.98 weight_scale=$scale cycles=20 batch_size=128 temperature=$temp save_samples=True log_dir=$logdir batchnorm=True weight_prior_params={"'"'"components"'"'":"'"'"$mix"'"'"}"
                done
            elif [[ $prior == "convcorrnormal" ]]; then
                bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=64000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=cifar10 inference=VerletSGLD model=$model warmup=30 burnin=15 skip=1 n_samples=100 lr=$lr momentum=0.98 weight_scale=$scale cycles=20 batch_size=128 temperature=$temp save_samples=True log_dir=$logdir batchnorm=True"
            elif [[ $prior == "convcorrnormal_fitted_ls" ]]; then
                bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=64000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=cifar10 inference=VerletSGLD model=$model warmup=30 burnin=15 skip=1 n_samples=100 lr=$lr momentum=0.98 weight_scale=$scale cycles=20 batch_size=128 temperature=$temp save_samples=True log_dir=$logdir batchnorm=True weight_prior_params.lengthscale_dict_file=$lens"
            else
                bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=32000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=cifar10 inference=VerletSGLD model=$model warmup=30 burnin=15 skip=1 n_samples=100 lr=$lr momentum=0.98 weight_scale=$scale cycles=20 batch_size=128 temperature=$temp save_samples=True log_dir=$logdir batchnorm=True"
            fi
        done
    done
done
# SGD run for comparison
bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=32000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=improper data=cifar10 inference=VerletSGLD model=$model warmup=30 burnin=15 skip=1 n_samples=100 lr=$lr momentum=0.98 weight_scale=1.41 cycles=20 batch_size=128 temperature=0.0 save_samples=True log_dir=$logdir batchnorm=True"
# VerletSGLD run for comparison
# bsub -n 2 -W 120:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=32000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=gaussian data=cifar10 inference=VerletSGLD model=$model warmup=30 burnin=15 skip=1 n_samples=100 lr=$lr momentum=0.98 weight_scale=1.41 cycles=20 batch_size=128 temperature=1.0 save_samples=True log_dir=$logdir batchnorm=True"
