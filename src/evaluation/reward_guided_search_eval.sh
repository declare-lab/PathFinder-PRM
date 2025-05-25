#!/bin/bash
cd reward_guided_search/

dataset_list=("amc23" "aime24" "math" "college_math" "minerva_math" "olympiadbench")

checkpoint="declare-lab/PathFinder-PRM-7B"

# Performing Search and Generating Responses
for dataset in "${dataset_list[@]}"; do
    echo "Dataset: $dataset"
    CUDA_VISIBLE_DEVICES=0,1 python Greedy-Seach-Zero-Shot.py \
        --policy_model_path Qwen/Qwen2.5-7B-Instruct \
        --reward_model_path $checkpoint \
        --data $dataset \
        --output_dir ./outputs/PathFinder-PRM-7B/$dataset 
done


# Evaluating Responses

python eval_results.py \
    --results_dir ./outputs/PathFinder-PRM-7B/