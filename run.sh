#!/bin/bash

# >>>>>>>>>>>>>> Command Line Inputs >>>>>>>>>>>>>>
input_dir=$1    # "/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro"
num_gpus=$2
# <<<<<<<<<<<<<< Command Line Inputs <<<<<<<<<<<<<<


# >>>>>>>>>>>>>> Log Info >>>>>>>>>>>>>>
echo "input_dir: $input_dir"
echo "num_gpus: $num_gpus"
single_gpu=1
# <<<<<<<<<<<<<< Log Info <<<<<<<<<<<<<<


# >>>>>>>>>>>>>> Tunable parameters >>>>>>>>>>>>>>
input_suffix="ct.nii.gz"
input_list="\$labels2onehot.build_input_list(@input_dir,""@input_suffix,""@output_dir)"
export VISTA3D_OUTPUT_DIR="./eval"
# <<<<<<<<<<<<<< Tunable parameters <<<<<<<<<<<<<<


# >>>>>>>>>>>>>> SINGLE GPU inference >>>>>>>>>>>>>>
if [ $num_gpus -eq $single_gpu ]; then
    echo "single GPU inference..."
    python -m monai.bundle run \
        --config_file="['configs/inference.json', 'configs/batch_inference.json', 'configs/mgpu_inference.json']" \
        --input_dir=$input_dir \
        --input_suffix=$input_suffix \
        --input_list=$input_list \
        --output_dir=$VISTA3D_OUTPUT_DIR \
        --output_postfix="step1_117"
fi
# <<<<<<<<<<<<<< MULTI GPU inference <<<<<<<<<<<<<<

# >>>>>>>>>>>>>> MULTI GPU inference >>>>>>>>>>>>>>
if [ $num_gpus -gt $single_gpu ]; then
    echo "multi GPU inference..."
    torchrun --nnodes=1 --nproc_per_node=$num_gpus -m monai.bundle run \
        --config_file="['configs/inference.json', 'configs/batch_inference.json', 'configs/mgpu_inference.json']" \
        --input_dir=$input_dir \
        --input_suffix=$input_suffix \
        --input_list=$input_list \
        --output_dir=$VISTA3D_OUTPUT_DIR \
        --output_postfix="step1_117"
fi
# <<<<<<<<<<<<<< MULTI GPU inference <<<<<<<<<<<<<<
