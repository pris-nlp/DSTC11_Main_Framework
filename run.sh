#!/usr/bin bash


for s in 0
do
    python run_experiment.py \
        --data_root_dir ./dstc11/development \
        --experiment_root_dir results/DKT \
        --seed $s \

done
