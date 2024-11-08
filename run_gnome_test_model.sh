#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo "USED GPUs: $CUDA_VISIBLE_DEVICES"
pwd
source activate time_series_augmix

# Load environment variables from .env file
set -a
source .env
set +a

# Attempt to determine the directory of the script or use the default
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
if [[ "$SCRIPT_DIR" != *"/time_series_augmentation_mix"* ]]; then
    echo "Error: The script directory ($SCRIPT_DIR) does not contain '/time_series_augmentation_mix'. Using the default directory."
    SCRIPT_DIR="$DEFAULT_SCRIPT_DIR"
fi

# Change directory to the script directory
cd "$SCRIPT_DIR" || { echo "Error: Unable to change directory to $SCRIPT_DIR"; exit 1; }

# Debugging output for parameter values
echo "Configuration:"
echo "  GPUS: $GPUS"
echo "  MODEL: $MODEL"
echo "  OPTIMIZER: $OPTIMIZER"
echo "  INTERPRET_METHOD: $INTERPRET_METHOD"
echo "  INTERPRET: $INTERPRET"
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
      cmd="python3 main.py --gpus=$GPUS --dataset=$dataset --preset_files --augmentation_method=$aug --augmentation_ratio=$ratio --optimizer=$OPTIMIZER --model="cnn_attention_bilstm" --interpret --interpret_method=$INTERPRET_METHOD"

      # Conditionally add flags for store_true parameters
      [[ "$INTERPRET" == true ]] && cmd+=" --interpret"
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
