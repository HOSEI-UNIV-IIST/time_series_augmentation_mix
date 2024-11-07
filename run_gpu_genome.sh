#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo "USED GPUs: $CUDA_VISIBLE_DEVICES"
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

# Dataset and augmentation methods
gnome_data=('GAS_PANTHER_EDUCATION_MOHAMMAD'
'ELECTRICITY_MOUSE_HEALTH_ESTELA' 'HOTWATER_FOX_LODGING_ALANA'
'SOLAR_BOBCAT_EDUCATION_ALISSA' 'SOLAR_BOBCAT_EDUCATION_COLEMAN' 'WATER_PANTHER_LODGING_CORA'
'ELECTRICITY_MOUSE_SCIENCE_MICHEAL' 'HOTWATER_ROBIN_EDUCATION_MARGARITO' 'WATER_WOLF_EDUCATION_URSULA'
'ELECTRICITY_GLOBAL_REACTIVE_POWER' 'GAS_PANTHER_LODGING_DEAN'
)

aug_tech_mix=(
    'sequential_combined4' 'sequential_combined5' 'sequential_combined7' 'sequential_combined12'
    #'parallel_combined3' 'parallel_combined4' 'parallel_combined10' 'parallel_combined12'
)

# Default parameters for Python script
GPUS=4
MODEL="cnn"
OPTIMIZER="adam"
INTERPRET_METHOD="lime"
NORMALIZE_INPUT=true  # Enable normalization
TRAIN=true            # Enable training
TUNE=true             # Enable tuning
SAVE=true             # Enable saving

# Debugging output for parameter values
echo "Configuration:"
echo "  GPUS: $GPUS"
echo "  MODEL: $MODEL"
echo "  OPTIMIZER: $OPTIMIZER"
echo "  INTERPRET_METHOD: $INTERPRET_METHOD"
echo "  NORMALIZE_INPUT: $NORMALIZE_INPUT"
echo "  TRAIN: $TRAIN"
echo "  TUNE: $TUNE"
echo "  SAVE: $SAVE"
echo "  gnome_data: ${gnome_data[@]}"
echo "  aug_tech_mix: ${aug_tech_mix[@]}"

# Loop through datasets and augmentation methods
for ratio in $(seq 1); do
  for dataset in "${gnome_data[@]}"; do
    for aug in "${aug_tech_mix[@]}"; do
      echo "Running dataset: $dataset, augmentation: $aug, ratio: $ratio"

      # Construct the command
      cmd="python3 main.py --gpus=$GPUS --dataset=$dataset --preset_files --augmentation_method=$aug --augmentation_ratio=$ratio --optimizer=$OPTIMIZER --model=$MODEL --interpret_method=$INTERPRET_METHOD"

      # Conditionally add flags for store_true parameters
      [[ "$NORMALIZE_INPUT" == true ]] && cmd+=" --normalize_input"
      [[ "$TRAIN" == true ]] && cmd+=" --train"
      [[ "$TUNE" == true ]] && cmd+=" --tune"
      [[ "$SAVE" == true ]] && cmd+=" --save"

      # Execute the command
      echo "Executing: $cmd"
      eval $cmd
    done
  done
done

echo "Script completed successfully."
