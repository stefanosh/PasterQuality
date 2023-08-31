#!/bin/bash

PASTER_ROOT_DIR=/home/stefanos/PasterQuality/src/experiments/revisiting_models
PROJECT_DIR=/home/stefanos/PasterQuality/src/experiments/revisiting_models
# Download and move the cod to the project directory
# git clone https://github.com/Yura52/tabular-dl-revisiting-models $PROJECT_DIR
cd $PROJECT_DIR

# conda create -n revisiting_models python=3.8.8

# Does not work for some reason, must be done separately before the subsequent installs
# conda init bash
# conda activate revisiting_models


# Initial versions
# conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1.243 numpy=1.19.2 -c pytorch -y

# Paster version to work with our GPUs and CUDA version
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8  numpy=1.19.2 -c pytorch -c -y nvidia

# conda install cudnn=7.6.5 -c anaconda -y
# pip install -r requirements.txt
# conda install nodejs -y
# jupyter labextension install @jupyter-widgets/jupyterlab-manager

# if the following commands do not succeed, update conda
conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
conda env config vars set PROJECT_DIR=${PROJECT_DIR}
conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
conda env config vars set CUDA_HOME=${CONDA_PREFIX}
conda env config vars set CUDA_ROOT=${CONDA_PREFIX}

# If you are going to use CUDA all the time, you can save the environment variable in the Conda environment:
conda env config vars set CUDA_VISIBLE_DEVICES="0,1"

# cd $PASTER_ROOT_DIR
# python -m pip install -e .

# conda deactivate
# conda activate test

# cd $PROJECT_DIR
