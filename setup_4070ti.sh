#!/bin/bash

# Create new environment for RTX 4070 Ti (Requires CUDA 11.8+)
conda create -n OpenOccupancy-4070 python=3.8 -y
source activate OpenOccupancy-4070

# Install PyTorch 2.0.1 with CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install openmim
mim install mmcv-full==1.7.1
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6

# Install other dependencies
pip install timm
pip install open3d
pip install PyMCubes
pip install spconv-cu118
pip install fvcore
pip install setuptools==59.5.0
pip install yapf==0.40.1

# Install OpenOccupancy
python setup.py develop

echo "Environment OpenOccupancy-4070 created. Please activate it and try running the evaluation again."
