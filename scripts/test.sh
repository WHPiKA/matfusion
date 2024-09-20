#!/bin/bash
T=$(date +%Y-%m-%d-%H-%M-%S)

cd ..


python eval.py \
    --dataset datasets/real_test_materialgan.yml \
    --checkpoint checkpoints/flash_v1_jax \
    --output results/flash_v1_on_materialgan \
    2>&1 | tee scripts/logs/test.$T

cd scripts
