#!/bin/bash

cd ../
set -x

MODEL=$1
USE_WANDB=$2
use_step_rate=$3
use_interpolation=$4



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

for SEED in 0 1 2 3 4
do
    python main.py \
        seed=$SEED \
        env_name=FetchReach-v4 \
        algos/model=$MODEL \
        algos.use_step_rate=$use_step_rate \
        algos.model.use_interpolation=$use_interpolation \
        use_wandb=$USE_WANDB \
        total_training_steps=100000 \
        debug=false \
        group_name="${MODEL}_nonsteprate_q_bias_estimate"
done
