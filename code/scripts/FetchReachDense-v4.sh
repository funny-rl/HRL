#!/bin/bash

cd ../
set -x

MODEL=$1
USE_WANDB=$2
use_step_rate=$3
group_name=$4

export HYDRA_FULL_ERROR=1

if [ -z "$MODEL" ]; then # if MODEL is not provided, set default
    MODEL="vanilla_ddpg"
fi
if [ -z "$USE_WANDB" ]; then # if USE_WANDB is not provided, set default
    USE_WANDB=false
fi
if [ -z "$use_step_rate" ]; then # if use_step_rate is not provided, set default
    use_step_rate=false
fi
if [ -z "$group_name" ]; then # if group_name is not provided, set default
    group_name="default_group"
fi

for SEED in 0 1 2 3 4
do
    python main.py \
        seed=$SEED \
        env_name=FetchReachDense-v4 \
        algos/model=$MODEL \
        algos.use_step_rate=$use_step_rate \
        use_wandb=$USE_WANDB \
        total_training_steps=100000 \
        debug=true \
        group_name=$group_name
done
