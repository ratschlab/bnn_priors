#!/bin/bash

priors=( gaussian uniform laplace student-t lognormal cauchy )
datasets=( boston wine energy naval concrete kin8nm power yacht protein )
inference=( SGLD HMC )
scales=( 0.70 1.41 2.83 )

for prior in "${priors[@]}"
do
    for scale in "${scales[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for inf in "${inference[@]}"
            do
                bsub -n 2 -W 4:00 -J "bnn" -sp 40 -g /vincent/experiments/bnn -R "rusage[mem=4000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=UCI_$dataset inference=$inf warmup=5000 burnin=5000 weight_scale=$scale cycles=10"
            done
        done
    done
done
