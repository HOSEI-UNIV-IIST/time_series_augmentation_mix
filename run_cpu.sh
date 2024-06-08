#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo USED GPUs=$CUDA_VISIBLE_DEVICES
pwd
source activate time_serie_torch

# Default value for SCRIPT_DIR
DEFAULT_SCRIPT_DIR="/home/23r9802_chen/messou/TimeSeriesForecasting"

# Attempt to determine the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if SCRIPT_DIR contains "/TimeSeriesForecasting"
if [[ "$SCRIPT_DIR" != *"/TimeSeriesForecasting"* ]]; then
    echo "Error: The script directory does not contain '/TimeSeriesForecasting'. Using the default directory."
    SCRIPT_DIR="$DEFAULT_SCRIPT_DIR"
fi

# Change directory to the script directory
cd "$SCRIPT_DIR" || { echo "Error: Unable to change directory to $SCRIPT_DIR"; exit 1; }

# Pass the model_group and electricity parameters to the Python script
python ./src/main.py --model_group 0 --dataset 'apartment'
