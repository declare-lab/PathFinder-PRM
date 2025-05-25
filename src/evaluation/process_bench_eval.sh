#!/bin/bash

cd ProcessBench/code

python run_eval_finegrained_prm_v3.py \
    --model_path declare-lab/PathFinder-PRM-7B \
    --output_dir ./outputs/PathFinder-PRM-7B