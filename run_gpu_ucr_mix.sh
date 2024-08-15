#!/bin/bash
#SBATCH --gres=gpu:1
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
aug_tech_mix=(
    # Sequential Magnitude Methods (Uniq and Multi)
    'sequential_magnitude_uniq1' 'sequential_magnitude_uniq2' 'sequential_magnitude_uniq3' 'sequential_magnitude_uniq4'
    'sequential_magnitude_multi1' 'sequential_magnitude_multi2' 'sequential_magnitude_multi3' 'sequential_magnitude_multi4'
    # Sequential Time Methods (Uniq and Multi)
    'sequential_time_uniq1' 'sequential_time_uniq2' 'sequential_time_uniq3' 'sequential_time_uniq4'
    'sequential_time_multi1' 'sequential_time_multi2' 'sequential_time_multi3' 'sequential_time_multi4'
    # Sequential Combined Methods
    'sequential_combined1' 'sequential_combined2' 'sequential_combined3' 'sequential_combined4'
    'sequential_combined5' 'sequential_combined6' 'sequential_combined7' 'sequential_combined8'
    'sequential_combined9' 'sequential_combined10' 'sequential_combined11' 'sequential_combined12'
    # Parallel Magnitude Methods
    'parallel_magnitude_uniq_block1' 'parallel_magnitude_uniq_block2'
    'parallel_magnitude_uniq_block3' 'parallel_magnitude_uniq_block4'
    #'parallel_magnitude_uniq_mixed1' 'parallel_magnitude_uniq_mixed2'
    #'parallel_magnitude_uniq_mixed3' 'parallel_magnitude_uniq_mixed4'
    'parallel_magnitude_multi_block1' 'parallel_magnitude_multi_block2'
    'parallel_magnitude_multi_block3' 'parallel_magnitude_multi_block4'
    #'parallel_magnitude_multi_mixed1' 'parallel_magnitude_multi_mixed2'
    #'parallel_magnitude_multi_mixed3' 'parallel_magnitude_multi_mixed4'
    # Parallel Time Methods
    'parallel_time_uniq_block1' 'parallel_time_uniq_block2'
    'parallel_time_uniq_block3' 'parallel_time_uniq_block4'
    #'parallel_time_uniq_mixed1' 'parallel_time_uniq_mixed2'
    #'parallel_time_uniq_mixed3' 'parallel_time_uniq_mixed4'
    'parallel_time_multi_block1' 'parallel_time_multi_block2'
    'parallel_time_multi_block3' 'parallel_time_multi_block4'
    #'parallel_time_multi_mixed1' 'parallel_time_multi_mixed2'
    #'parallel_time_multi_mixed3' 'parallel_time_multi_mixed4'
    # Parallel Combined Methods
    'parallel_combined1' 'parallel_combined2' 'parallel_combined3'
    'parallel_combined4' 'parallel_combined5' 'parallel_combined6'
    'parallel_combined7' 'parallel_combined8' 'parallel_combined9'
    'parallel_combined10' 'parallel_combined11' 'parallel_combined12'
)

# Loop over each ratio, dataset, and augmentation technique and run the Python script
for ratio in $(seq 1); do
  for dataset in $datasets; do
    for aug in "${aug_tech_mix[@]}"; do
      python3 main.py --gpus=1 --data_dir=data/UCR --dataset="$dataset" --preset_files --ucr --normalize_input --train --save --augmentation_method="$aug" --augmentation_ratio=$ratio --optimizer=adam --model=fcnn
    done
  done
done
