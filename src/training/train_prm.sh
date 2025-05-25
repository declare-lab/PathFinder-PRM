#!/bin/bash

export TORCH_CPP_LOG_LEVEL=WARNING 
export NCCL_DEBUG=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 25679 --config_file training_configs/deepspeed_zero2.yaml \
train_prm.py \
training_configs/prm_train.yaml