#!/bin/bash

# conda create --name vista3d_bundle python=3.9 -y

# conda activate vista3d_bundle

# pip install "monai[fire]"
tar -xzvf monai_weekly-1.4.dev2436.tar.gz
cd monai_weekly-1.4.dev2436
pip install .
cd ../
rm -rf monai_weekly-1.4.dev2436
pip install wheel
pip install fire
pip install nibabel
pip install pytorch-ignite