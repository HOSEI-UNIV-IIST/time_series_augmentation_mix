#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo USED GPUs=$CUDA_VISIBLE_DEVICES
pwd
source activate time_series_augmix

# Default value for SCRIPT_DIR
DEFAULT_SCRIPT_DIR="/home/23r8105_messou/tsa/time_series_augmentation_mix"

# Attempt to determine the directory of the script
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

# Check if SCRIPT_DIR contains "/time_series_augmentation_mix"
if [[ "$SCRIPT_DIR" != *"/time_series_augmentation_mix"* ]]; then
    echo "Error: The script directory ($SCRIPT_DIR) does not contain '/time_series_augmentation_mix'. Using the default directory."
    SCRIPT_DIR="$DEFAULT_SCRIPT_DIR"
fi

# Change directory to the script directory
cd "$SCRIPT_DIR" || { echo "Error: Unable to change directory to $SCRIPT_DIR"; exit 1; }

# Read the datasets from constant.py
dataset_file="$SCRIPT_DIR/utils/constant.py"

if [[ ! -f "$dataset_file" ]]; then
    echo "Error: Dataset file $dataset_file not found"
    exit 1
fi

# Extract the dataset list from constant.py
datasets=$(python3 -c "
import ast
with open('$dataset_file', 'r') as f:
    tree = ast.parse(f.read(), filename='$dataset_file')
    datasets = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == 'ucr_data':
            datasets = ast.literal_eval(node.value)
    print(' '.join(datasets))
")

if [[ -z "$datasets" ]]; then
    echo "Error: No datasets found in $dataset_file"
    exit 1
fi

# Define the augmentation techniques directly in the bash file
aug_tech=('--original' '--jitter' '--scaling' '--rotation' '--permutation' '--randompermutation' '--magwarp' '--timewarp' '--windowslice' '--windowwarp')

# Loop over each dataset and augmentation technique and run the Python script
for ratio in $(seq 1 4); do
  for dataset in $datasets; do
      for aug in "${aug_tech[@]}"; do
          python3 main.py --gpus=4 --data_dir=data/UCR --dataset="$dataset" --preset_files --ucr --normalize_input --train --save $aug=True --augmentation_ratio=$ratio --model=vgg
      done
  done
done
