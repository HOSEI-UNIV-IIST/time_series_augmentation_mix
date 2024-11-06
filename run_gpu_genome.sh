#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo "USED GPUs: $CUDA_VISIBLE_DEVICES"
pwd
source activate time_series_augmix

# Load configuration from YAML file
CONFIG_FILE="config/run_params.yaml"

# Read device type
device=$(yq e '.device' $CONFIG_FILE)

# General settings from the YAML file
GPUS=$(yq e '.general.gpus' $CONFIG_FILE)
MODEL=$(yq e '.general.model' $CONFIG_FILE)
OPTIMIZER=$(yq e '.general.optimizer' $CONFIG_FILE)
INTERPRET_METHOD=$(yq e '.general.interpret_method' $CONFIG_FILE)
NORMALIZE_INPUT=$(yq e '.general.normalization' $CONFIG_FILE)
TRAIN=$(yq e '.general.train' $CONFIG_FILE)
TUNE=$(yq e '.general.tune' $CONFIG_FILE)
SAVE=$(yq e '.general.save' $CONFIG_FILE)

# Load datasets and augmentation methods
gnome_data=($(yq e '.datasets[]' $CONFIG_FILE))
aug_tech_mix=($(yq e '.augmentation_methods[]' $CONFIG_FILE))

# Adjust GPU settings based on device type
if [[ "$device" == "gpu" ]]; then
  device_flag="--gpus=$GPUS"
  echo "Running on GPU with $GPUS GPU(s)"
else
  device_flag="--device=cpu"
  echo "Running on CPU"
fi

# Loop through datasets and augmentation methods
for ratio in $(seq 1); do
  for dataset in "${gnome_data[@]}"; do
    for aug in "${aug_tech_mix[@]}"; do
      python3 main.py $device_flag --dataset="ELECTRICITY_MOUSE_HEALTH_ESTELA" --preset_files \
                      --normalize_input="$NORMALIZE_INPUT" \
                      --augmentation_method="$aug" --augmentation_ratio=$ratio \
                      --optimizer="$OPTIMIZER" --train="$TRAIN" --tune="$TUNE" \
                      --save="$SAVE" --model="$MODEL" --interpret_method="$INTERPRET_METHOD"
    done
  done
done
