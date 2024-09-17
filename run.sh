#!/bin/bash

input_dir="/root/autodl-tmp/VISTA-AbdomenAtlasDemo"
input_dir=$1        
second_stage=false
second_stage=$2     
echo "input_dir: $input_dir"
echo "use second_stage: $second_stage"

folder_list=()

for dir in $(find $input_dir -type d -name "BDMAP_*" | sort -V); do
    folder_name=$(basename "$dir")  
    folder_list+=("$folder_name")  
done

echo "Found folders: ${folder_list[@]}"

for folder in "${folder_list[@]}"; do
    echo "Processing folder: $folder"

    CUDA_VISIBLE_DEVICES=0 python -m monai.bundle run \
        --config_file="['configs/inference.json', 'configs/batch_inference.json']" \
        --input_dir=$input_dir \
        --volume_name=$folder \
        --output_postfix="step1_117"

    if $second_stage; then
        echo "inference 2nd time for other classes"
        CUDA_VISIBLE_DEVICES=0 python -m monai.bundle run \
            --config_file="['configs/inference.json', 'configs/batch_inference.json']" \
            --input_dir=$input_dir \
            --volume_name=$folder \
            --everything_labels="[23,24,25,26,27,128,132]" \
            --output_postfix="step2_7"
    fi 

    
    python labels2onehot.py $folder $second_stage   # seperate 117 (and 7) labels to binary maps

    rm "./eval/$folder/ct_step1_117.nii.gz"
    if $second_stage; then
        rm rm "./eval/$folder/ct_step2_7.nii.gz"
    fi

done
