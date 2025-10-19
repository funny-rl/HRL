#!/bin/bash

cd ../

python main.py \
    algos.model.name=vanilla_ddpg \
    env_name=Pendulum-v1 \
    use_wandb=true \