#!/bin/bash

input_dir="/root/autodl-tmp/VISTA-AbdomenAtlasDemo"
input_dir=$1        
second_stage=false
second_stage=$2     
num_gpus=1
num_gpus=$3
single_gpu=1
echo "input_dir: $input_dir"
echo "use second_stage: $second_stage"
echo "num_gpus: $num_gpus"


if [ $num_gpus -eq $single_gpu ]; then
CUDA_VISIBLE_DEVICES=0 python -m monai.bundle run \
    --config_file="['configs/inference.json', 'configs/batch_inference.json']" \
    --input_dir=$input_dir \
    --output_postfix="step1_117"
fi

if [ $num_gpus -gt $single_gpu ]; then
torchrun --nnodes=1 --nproc_per_node=$num_gpus -m monai.bundle run \
    --config_file="['configs/inference.json', 'configs/batch_inference.json', 'configs/mgpu_inference.json']" \
    --input_dir=$input_dir \
    --output_postfix="step1_117"
fi

# if $second_stage; then
#     echo "inference 2nd time for other classes"
#     CUDA_VISIBLE_DEVICES=0 python -m monai.bundle run \
#         --config_file="['configs/inference.json', 'configs/batch_inference.json']" \
#         --input_dir=$input_dir \
#         --everything_labels="[23,24,25,26,27,128,132]" \
#         --output_postfix="step2_7"
# fi 

    
