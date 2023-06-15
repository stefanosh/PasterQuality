#!/bin/bash

#### Usage: For TabNet only https://github.com/Yura52/tabular-dl-revisiting-models#32-tensorflow-environment
# TODO: Several errors, mostly warnings, occur when tuning tabnet. Check if they are important based on tf and CUDA, or if they can be ignored.

#### First part, must be done separately from second part due to conda activate problems
PASTER_ROOT_DIR="/home/stefanos/pasteurAIzer"
PROJECT_DIR="/home/stefanos/pasteurAIzer/src/experiments/revisiting_models"

# Download and move the cod to the project directory
# git clone https://github.com/Yura52/tabular-dl-revisiting-models $PROJECT_DIR
cd $PROJECT_DIR

# conda create -n revisiting_models_tf python=3.7.10

# Does not work for some reason, must be done separately before the subsequent installs
# conda init bash

# eval "$(conda shell.bash hook)"
# conda activate revisiting_models_tf


### Sencond part
# Initial versions
# conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.0 numpy=1.19.2 -c pytorch -y

# Paster version to work with our GPUs and CUDA version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8  numpy=1.19.2 -c pytorch -c nvidia

conda install cudnn=7.6.5 -c anaconda -y
pip install tensorflow-gpu==1.14
pip install -r requirements.txt
conda install nodejs -y
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Due to a bug in the latest version of protobuf, we need to downgrade it
pip install protobuf==3.19.6

# if the following commands do not succeed, update conda
conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
conda env config vars set PROJECT_DIR=${PROJECT_DIR}
conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
conda env config vars set CUDA_HOME=${CONDA_PREFIX}
conda env config vars set CUDA_ROOT=${CONDA_PREFIX}

# If you are going to use CUDA all the time, you can save the environment variable in the Conda environment:
conda env config vars set CUDA_VISIBLE_DEVICES="0"

cd $PASTER_ROOT_DIR
python -m pip install -e .

# Does not work
# conda deactivate
# conda activate revisiting_models_tf

cd $PROJECT_DIR