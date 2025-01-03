#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/11/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import argparse


def argument_parser():
    global args
    parser = argparse.ArgumentParser(description='Runs augmentation model.')
    # General settings
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use")
    parser.add_argument('--dataset', type=str, default='CBF', help='Name of dataset to test (required, ex: unipen1a)')
    parser.add_argument('--tune', default=False, action="store_true", help="Hyperparameters Tuner?")
    parser.add_argument('--train', default=False, action="store_true", help="Train?")
    parser.add_argument('--interpret', default=False, action='store_true',
                        help="Flag to perform interpretation after evaluation.")
    parser.add_argument('--interpret_method', type=str, default="shap", choices=["shap", "lime"],
                        help="Choose interpretation method (shap or lime).")
    parser.add_argument('--save', default=True, action="store_true", help="Save to disk?")
    parser.add_argument('--extension', type=str, default='txt', help="Dataset file extension")
    # Augmentation
    parser.add_argument('--read_augmented', default=False, action="store_true",
                        help="Read existing augmented data from disk?")
    parser.add_argument('--augmentation_ratio', type=int, default=1, help="How many times to augment")
    parser.add_argument('--num_augmentations', type=int, default=3, help="Number of random augmentations to apply")
    parser.add_argument('--seed', type=int, default=20240609, help="Randomization seed")
    parser.add_argument('--original', type=bool, default=False, help="Original dataset without augmentation")
    parser.add_argument('--jitter', type=bool, default=False, help="Jitter preset augmentation")
    parser.add_argument('--scaling', type=bool, default=False, help="Scaling preset augmentation")
    parser.add_argument('--permutation', type=bool, default=False, help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', type=bool, default=False,
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', type=bool, default=False, help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', type=bool, default=False, help="Time warp preset augmentation")
    parser.add_argument('--windowslice', type=bool, default=False, help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', type=bool, default=False, help="Window warp preset augmentation")
    parser.add_argument('--rotation', type=bool, default=False, help="Rotation preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    # File settings
    parser.add_argument('--preset_files', default=False, action="store_true", help="Use preset files")
    parser.add_argument('--ucr', default=False, action="store_true", help="Use UCR 2015")
    parser.add_argument('--ucr2018', default=False, action="store_true", help="Use UCR 2018")
    parser.add_argument('--data_dir', type=str, default="data/meters/preprocessed", help="Data dir")
    parser.add_argument('--train_data_file', type=str, default="", help="Train data file")
    parser.add_argument('--train_labels_file', type=str, default="", help="Train label file")
    parser.add_argument('--test_data_file', type=str, default="", help="Test data file")
    parser.add_argument('--test_labels_file', type=str, default="", help="Test label file")
    parser.add_argument('--test_split', type=int, default=0, help="Test split")
    parser.add_argument('--weight_dir', type=str, default="weights", help="Weight path")
    parser.add_argument('--log_dir', type=str, default="logs", help="Log path")
    parser.add_argument('--output_dir', type=str, default="output", help="Output path")
    parser.add_argument('--normalize_input', default=True, action="store_true",
                        help="Normalize between [-1,1] or [0,1]")
    parser.add_argument('--normalize_input_positive', default=True, action="store_true", help="Normalize between [0,1]")
    parser.add_argument('--delimiter', type=str, default=" ", help="Delimiter")
    # Network settings
    parser.add_argument('--optimizer', type=str, default="adam", help="Which optimizer")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
    parser.add_argument('--validation_split', type=int, default=0, help="Size of validation set")
    parser.add_argument('--n_trials', type=int, default=500, help="Number of Maximum number of trials as search space")
    parser.add_argument('--iterations', type=int, default=10000, help="Number of iterations")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--verbose', type=int, default=2, help="Verbose")

    parser.add_argument('--model', type=str, default="gru",
                        choices=[
                            "cnn", "lstm", "gru", "cnn_lstm", "cnn_gru", "bigru_cnn_bigru", "bilstm_cnn_bilstm",
                            "cnn_attention_bigru", "cnn_attention_bilstm"
                        ],
                        help="Set model name")
    parser.add_argument('--augmentation_method', type=str, default="simple",
                        choices=[
                            'simple',
                            # Sequential Magnitude Methods (Uniq and Multi)
                            'sequential_magnitude_uniq1', 'sequential_magnitude_uniq2', 'sequential_magnitude_uniq3',
                            'sequential_magnitude_uniq4',
                            'sequential_magnitude_multi1', 'sequential_magnitude_multi2', 'sequential_magnitude_multi3',
                            'sequential_magnitude_multi4',
                            # Sequential Time Methods (Uniq and Multi)
                            'sequential_time_uniq1', 'sequential_time_uniq2', 'sequential_time_uniq3',
                            'sequential_time_uniq4',
                            'sequential_time_multi1', 'sequential_time_multi2', 'sequential_time_multi3',
                            'sequential_time_multi4',
                            # Sequential Combined Methods
                            'sequential_combined1', 'sequential_combined2', 'sequential_combined3',
                            'sequential_combined4', 'sequential_combined5', 'sequential_combined6',
                            'sequential_combined7', 'sequential_combined8', 'sequential_combined9',
                            'sequential_combined10', 'sequential_combined11', 'sequential_combined12',
                            # Parallel Magnitude Methods (Unique Block and Mixed)
                            'parallel_magnitude_uniq1', 'parallel_magnitude_uniq2',
                            'parallel_magnitude_uniq3', 'parallel_magnitude_uniq4',
                            'parallel_magnitude_multi1', 'parallel_magnitude_multi2',
                            'parallel_magnitude_multi3', 'parallel_magnitude_multi4',
                            # Parallel Time Methods (Unique Block and Mixed)
                            'parallel_time_uniq1', 'parallel_time_uniq2',
                            'parallel_time_uniq3', 'parallel_time_uniq4',
                            'parallel_time_multi1', 'parallel_time_multi2',
                            'parallel_time_multi3', 'parallel_time_multi4',
                            # Parallel Combined Methods
                            'parallel_combined1', 'parallel_combined2', 'parallel_combined3',
                            'parallel_combined4', 'parallel_combined5', 'parallel_combined6',
                            'parallel_combined7', 'parallel_combined8', 'parallel_combined9',
                            'parallel_combined10', 'parallel_combined11', 'parallel_combined12'
                        ],
                        help="Augmentation method to apply")
    args = parser.parse_args()
    return args
