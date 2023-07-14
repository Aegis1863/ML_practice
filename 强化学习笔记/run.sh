#!/bin/bash
module load anaconda/2022.10
module load cuda/11.8

source activate pytorch_310
export PYTHONUNBUFFERED=1

python 07_DQN.py