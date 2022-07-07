#!/usr/bin bash


for s in 30 31 32 33 34
do
    python run_experiments.py \
        --data_root_dir ./dstc11/development \
        --experiment_root_dir results/DKT \
        --seed $s \

done
