#!/usr/bin bash


for s in 42 0 1 2 3 4
do
    python run_experiment.py \
        --data_root_dir ./dstc11/development \
        --experiment_root_dir results/DKT \
        --seed $s \

done
