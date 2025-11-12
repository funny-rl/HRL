#!/bin/bash

cd ../
set -x

MODEL=$1
USE_WANDB=$2

export HYDRA_FULL_ERROR=1

if [ -z "$MODEL" ]; then # if MODEL is not provided, set default
    MODEL="vanilla_ddpg"
fi
if [ -z "$USE_WANDB" ]; then # if USE_WANDB is not provided, set default
    USE_WANDB=false
fi

for SEED in 0 1 2 3 4
do
    python main.py \
        seed=$SEED \
        env_name=InvertedDoublePendulum-v5 \
        algos/model=$MODEL \
        use_wandb=$USE_WANDB \
        total_training_steps=50000
done

MountainCarContinuous-v0