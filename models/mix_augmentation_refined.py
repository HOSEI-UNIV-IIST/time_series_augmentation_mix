#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/23/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""
import re

import numpy as np



import re
import numpy as np


def run_augmentation_refined(x, y, args):
    """
    Orchestrates the augmentation process based on the method specified in args.
    :param x: Input data
    :param y: Labels
    :param args: Arguments containing augmentation settings
    :return: Augmented data, augmented labels, and augmentation tags
    """

    augmentation_tags = args.extra_tag
    ratio = args.augmentation_ratio * 4 if 'sequential' in args.augmentation_method else args.augmentation_ratio

    if ratio > 0:
        initial_tags = f"{args.augmentation_ratio}"

        # Determine the augmentation method to apply
        if 'sequential' in args.augmentation_method:
            x_temp, y_temp, temp_tags = augment_sequential(x, y, args)
        elif 'parallel' in args.augmentation_method:
            x_temp, y_temp, temp_tags = augment_parallel(x, y, args)
        elif args.augmentation_method == 'simple':
            x_temp, temp_tags = augment_data_simple(x, args)
            y_temp = y  # For simple methods, y remains the same
        else:
            raise ValueError(f"Unknown augmentation method: {args.augmentation_method}")

        # Ensure that the feature dimension matches before concatenation
        if x.shape[1] != x_temp.shape[1]:
            raise ValueError(
                f"Mismatch in the feature dimension: x.shape[1]={x.shape[1]}, x_temp.shape[1]={x_temp.shape[1]}")

        # Augmented data is directly concatenated with original data
        x_aug = np.concatenate((x, x_temp), axis=0)
        y_aug = np.concatenate((y, y_temp), axis=0)

        print(f"x_temp.shape: {x_temp.shape}, y_temp.shape: {y_temp.shape}")
        print(f"x_aug.shape: {x_aug.shape}, y_aug.shape: {y_aug.shape}")
        print(f"Augmentation done: {augmentation_tags + temp_tags}")

        # Update augmentation tags
        augmentation_tags = f"{args.extra_tag}_{temp_tags}" if args.extra_tag else temp_tags
    else:
        x_aug, y_aug = x, y  # No augmentation, just return original data
        augmentation_tags = args.extra_tag

    return x_aug, y_aug, augmentation_tags


def augment_sequential(x, y, args):
    method_num = re.search(r'\d+', args.augmentation_method).group()
    if 'magnitude' in args.augmentation_method:
        if 'uniq' in args.augmentation_method:
            func_name = f'ads_magnitude_uniq{method_num}'
        elif 'multi' in args.augmentation_method:
            func_name = f'ads_magnitude_multi{method_num}'
    elif 'time' in args.augmentation_method:
        if 'uniq' in args.augmentation_method:
            func_name = f'ads_time_uniq{method_num}'
        elif 'multi' in args.augmentation_method:
            func_name = f'ads_time_multi{method_num}'
    elif 'combined' in args.augmentation_method:
        func_name = f'ads_sequential_combined{method_num}'
    else:
        raise ValueError(f"Unknown sequential augmentation method: {args.augmentation_method}")

    if func_name in globals():
        return globals()[func_name](x, y, args.augmentation_ratio)
    else:
        raise ValueError(f"Function {func_name} not found for augmentation method {args.augmentation_method}")


def augment_parallel(x, y, args):
    method_num = re.search(r'\d+', args.augmentation_method).group()
    if 'magnitude' in args.augmentation_method:
        if 'uniq' in args.augmentation_method:
            func_name = f'adp_magnitude_uniq{method_num}'
        elif 'multi' in args.augmentation_method:
            func_name = f'adp_magnitude_multi{method_num}'
        else:
            raise ValueError(f"Unknown parallel magnitude method: {args.augmentation_method}")
    elif 'time' in args.augmentation_method:
        if 'uniq' in args.augmentation_method:
            func_name = f'adp_time_uniq{method_num}'
        elif 'multi' in args.augmentation_method:
            func_name = f'adp_time_multi{method_num}'
        else:
            raise ValueError(f"Unknown parallel time method: {args.augmentation_method}")
    elif 'combined' in args.augmentation_method:
        func_name = f'adp_parallel_combined{method_num}'
    else:
        raise ValueError(f"Unknown parallel augmentation method: {args.augmentation_method}")

    if func_name in globals():
        return globals()[func_name](x, y, args.augmentation_ratio)
    else:
        raise ValueError(f"Function {func_name} not found for augmentation method {args.augmentation_method}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║             Simple Augmentation Methods                          ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
def augment_data_simple(x, args):
    import utils.augmentation as aug
    augmentation_tags = ""
    if args.jitter:
        x = aug.jitter(x)
        augmentation_tags += "_jitter"
    elif args.scaling:
        x = aug.scaling(x)
        augmentation_tags += "_scaling"
    elif args.rotation:
        x = aug.rotation(x)
        augmentation_tags += "_rotation"
    elif args.permutation:
        x = aug.permutation(x)
        augmentation_tags += "_permutation"
    elif args.randompermutation:
        x = aug.permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    elif args.magwarp:
        x = aug.magnitude_warp(x)
        augmentation_tags += "_magwarp"
    elif args.timewarp:
        x = aug.time_warp(x)
        augmentation_tags += "_timewarp"
    elif args.windowslice:
        x = aug.window_slice(x)
        augmentation_tags += "_windowslice"
    elif args.windowwarp:
        x = aug.window_warp(x)
        augmentation_tags += "_windowwarp"
    return x, augmentation_tags


# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║             Sequential Augmentation Methods                      ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# ————————————————————————————————————————————————————————
# MAGNITUDE  —  UNIQUE
# ————————————————————————————————————————————————————————
def ads_magnitude_uniq1(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.jitter(x_combined)

    augmentation_tags = f"ads_magnitude_uniq1"
    return x_combined, y_combined, augmentation_tags


def ads_magnitude_uniq2(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.rotation(x_combined)

    augmentation_tags = f"ads_magnitude_uniq2"
    return x_combined, y_combined, augmentation_tags


def ads_magnitude_uniq3(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.scaling(x_combined)
    x_combined = aug.scaling(x_combined)
    x_combined = aug.scaling(x_combined)
    x_combined = aug.scaling(x_combined)

    augmentation_tags = f"ads_magnitude_uniq3"
    return x_combined, y_combined, augmentation_tags


def ads_magnitude_uniq4(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.magnitude_warp(x_combined)
    x_combined = aug.magnitude_warp(x_combined)
    x_combined = aug.magnitude_warp(x_combined)
    x_combined = aug.magnitude_warp(x_combined)

    augmentation_tags = f"ads_magnitude_uniq4"
    return x_combined, y_combined, augmentation_tags


# ————————————————————————————————————————————————————————
# MAGNITUDE  —  MULTIPLE
# ————————————————————————————————————————————————————————
def ads_magnitude_multi1(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.scaling(x_combined)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.magnitude_warp(x_combined)
    x_combined = aug.jitter(x_combined)

    augmentation_tags = f"ads_magnitude_multi1"
    return x_combined, y_combined, augmentation_tags


def ads_magnitude_multi2(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.magnitude_warp(x_combined)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.scaling(x_combined)

    augmentation_tags = f"ads_magnitude_multi2"
    return x_combined, y_combined, augmentation_tags


def ads_magnitude_multi3(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.magnitude_warp(x_combined)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.scaling(x_combined)
    x_combined = aug.rotation(x_combined)

    augmentation_tags = f"ads_magnitude_multi3"
    return x_combined, y_combined, augmentation_tags


def ads_magnitude_multi4(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.scaling(x_combined)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.magnitude_warp(x_combined)

    augmentation_tags = f"ads_magnitude_multi4"
    return x_combined, y_combined, augmentation_tags


# ————————————————————————————————————————————————————————
# TIME  —  UNIQUE
# ————————————————————————————————————————————————————————
def ads_time_uniq1(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.permutation(x_combined)
    x_combined = aug.permutation(x_combined)
    x_combined = aug.permutation(x_combined)
    x_combined = aug.permutation(x_combined)

    augmentation_tags = f"ads_time_uniq1"
    return x_combined, y_combined, augmentation_tags


def ads_time_uniq2(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.window_slice(x_combined)

    augmentation_tags = f"ads_time_uniq2"
    return x_combined, y_combined, augmentation_tags


def ads_time_uniq3(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.time_warp(x_combined)

    augmentation_tags = f"ads_time_uniq3"
    return x_combined, y_combined, augmentation_tags


def ads_time_uniq4(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.window_warp(x_combined)

    augmentation_tags = f"ads_time_uniq4"
    return x_combined, y_combined, augmentation_tags


# ————————————————————————————————————————————————————————
# TIME  —  MULTIPLE
# ————————————————————————————————————————————————————————
def ads_time_multi1(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.permutation(x_combined)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.window_warp(x_combined)

    augmentation_tags = f"ads_time_multi1"
    return x_combined, y_combined, augmentation_tags


def ads_time_multi2(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.permutation(x_combined)

    augmentation_tags = f"ads_time_multi2"
    return x_combined, y_combined, augmentation_tags


def ads_time_multi3(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.permutation(x_combined)
    x_combined = aug.window_slice(x_combined)

    augmentation_tags = f"ads_time_multi3"
    return x_combined, y_combined, augmentation_tags


def ads_time_multi4(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.permutation(x_combined)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.time_warp(x_combined)

    augmentation_tags = f"ads_time_multi4"
    return x_combined, y_combined, augmentation_tags


# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║             Parallel Augmentation Methods                        ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ————————————————————————————————————————————————————————
# MAGNITUDE  —  UNIQUE
# ————————————————————————————————————————————————————————
def adp_magnitude_uniq1(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    # Ensure the dataset is evenly divided into 4 parts
    x_split, y_split = partition.divide_dataset(x, y, 4)

    # Apply the specific augmentation to one split
    x_jitter = aug.jitter(x_split[0].copy())

    # Concatenate the augmented and non-augmented splits, tiled by ratio
    x_combined = np.concatenate([np.tile(x_jitter, (ratio, 1)), x_split[1], x_split[2], x_split[3]], axis=0)
    y_combined = np.concatenate([np.tile(y_split[0], ratio), y_split[1], y_split[2], y_split[3]], axis=0)

    augmentation_tags = f"adp_magnitude_uniq1"
    return x_combined, y_combined, augmentation_tags


def adp_magnitude_uniq2(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)
    x_rotation = aug.rotation(x_split[1].copy())

    x_combined = np.concatenate([x_split[0], np.tile(x_rotation, (ratio, 1)), x_split[2], x_split[3]], axis=0)
    y_combined = np.concatenate([y_split[0], np.tile(y_split[1], ratio), y_split[2], y_split[3]], axis=0)

    augmentation_tags = f"adp_magnitude_uniq2"
    return x_combined, y_combined, augmentation_tags


def adp_magnitude_uniq3(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)
    x_scaling = aug.scaling(x_split[2].copy())

    x_combined = np.concatenate([x_split[0], x_split[1], np.tile(x_scaling, (ratio, 1)), x_split[3]], axis=0)
    y_combined = np.concatenate([y_split[0], y_split[1], np.tile(y_split[2], ratio), y_split[3]], axis=0)

    augmentation_tags = f"adp_magnitude_uniq3"
    return x_combined, y_combined, augmentation_tags


def adp_magnitude_uniq4(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)
    x_magnitude_warp = aug.magnitude_warp(x_split[3].copy())

    x_combined = np.concatenate([x_split[0], x_split[1], x_split[2], np.tile(x_magnitude_warp, (ratio, 1))], axis=0)
    y_combined = np.concatenate([y_split[0], y_split[1], y_split[2], np.tile(y_split[3], ratio)], axis=0)

    augmentation_tags = f"adp_magnitude_uniq4"
    return x_combined, y_combined, augmentation_tags

# ————————————————————————————————————————————————————————
# MAGNITUDE  —  MULTIPLE
# ————————————————————————————————————————————————————————
def adp_magnitude_multi1(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)

    # Apply different augmentations to each split
    x_scaling = aug.scaling(x_split[0].copy())
    x_rotation = aug.rotation(x_split[1].copy())
    x_magnitude_warp = aug.magnitude_warp(x_split[2].copy())
    x_jitter = aug.jitter(x_split[3].copy())

    x_combined = np.concatenate([np.tile(x_scaling, (ratio, 1)),
                                 np.tile(x_rotation, (ratio, 1)),
                                 np.tile(x_magnitude_warp, (ratio, 1)),
                                 np.tile(x_jitter, (ratio, 1))], axis=0)

    y_combined = np.concatenate([np.tile(y_split[i], ratio) for i in range(4)], axis=0)

    augmentation_tags = f"adp_magnitude_multi1"
    return x_combined, y_combined, augmentation_tags


def adp_magnitude_multi2(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_rotation = aug.rotation(x_split[0].copy())
    x_magnitude_warp = aug.magnitude_warp(x_split[1].copy())
    x_jitter = aug.jitter(x_split[2].copy())
    x_scaling = aug.scaling(x_split[3].copy())

    x_combined = np.concatenate([np.tile(x_rotation, (ratio, 1)),
                                 np.tile(x_magnitude_warp, (ratio, 1)),
                                 np.tile(x_jitter, (ratio, 1)),
                                 np.tile(x_scaling, (ratio, 1))], axis=0)

    y_combined = np.concatenate([np.tile(y_split[i], ratio) for i in range(4)], axis=0)

    augmentation_tags = f"adp_magnitude_multi2"
    return x_combined, y_combined, augmentation_tags


def adp_magnitude_multi3(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_magnitude_warp = aug.magnitude_warp(x_split[0].copy())
    x_jitter = aug.jitter(x_split[1].copy())
    x_scaling = aug.scaling(x_split[2].copy())
    x_rotation = aug.rotation(x_split[3].copy())

    x_combined = np.concatenate([np.tile(x_magnitude_warp, (ratio, 1)),
                                 np.tile(x_jitter, (ratio, 1)),
                                 np.tile(x_scaling, (ratio, 1)),
                                 np.tile(x_rotation, (ratio, 1))], axis=0)

    y_combined = np.concatenate([np.tile(y_split[i], ratio) for i in range(4)], axis=0)

    augmentation_tags = f"adp_magnitude_multi3"
    return x_combined, y_combined, augmentation_tags


def adp_magnitude_multi4(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_jitter = aug.jitter(x_split[0].copy())
    x_scaling = aug.scaling(x_split[1].copy())
    x_rotation = aug.rotation(x_split[2].copy())
    x_magnitude_warp = aug.magnitude_warp(x_split[3].copy())

    x_combined = np.concatenate([np.tile(x_jitter, (ratio, 1)),
                                 np.tile(x_scaling, (ratio, 1)),
                                 np.tile(x_rotation, (ratio, 1)),
                                 np.tile(x_magnitude_warp, (ratio, 1))], axis=0)

    y_combined = np.concatenate([np.tile(y_split[i], ratio) for i in range(4)], axis=0)

    augmentation_tags = f"adp_magnitude_multi4"
    return x_combined, y_combined, augmentation_tags


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ————————————————————————————————————————————————————————
# TIME  —  UNIQUE
# ————————————————————————————————————————————————————————
def adp_time_uniq1(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    # Split the data into 4 parts
    x_split, y_split = partition.divide_dataset(x, y, 4)
    # Apply the augmentation to the first part
    x_permutation = aug.permutation(x_split[0].copy())

    # Concatenate the augmented and non-augmented splits
    x_combined = np.concatenate([np.tile(x_permutation, (ratio, 1)), x_split[1], x_split[2], x_split[3]], axis=0)
    y_combined = np.concatenate([np.tile(y_split[0], ratio), y_split[1], y_split[2], y_split[3]], axis=0)

    augmentation_tags = f"adp_time_uniq1"
    return x_combined, y_combined, augmentation_tags


def adp_time_uniq2(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)
    x_window_slice = aug.window_slice(x_split[1].copy())

    x_combined = np.concatenate([x_split[0], np.tile(x_window_slice, (ratio, 1)), x_split[2], x_split[3]], axis=0)
    y_combined = np.concatenate([y_split[0], np.tile(y_split[1], ratio), y_split[2], y_split[3]], axis=0)

    augmentation_tags = f"adp_time_uniq2"
    return x_combined, y_combined, augmentation_tags


def adp_time_uniq3(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)
    x_time_warp = aug.time_warp(x_split[2].copy())

    x_combined = np.concatenate([x_split[0], x_split[1], np.tile(x_time_warp, (ratio, 1)), x_split[3]], axis=0)
    y_combined = np.concatenate([y_split[0], y_split[1], np.tile(y_split[2], ratio), y_split[3]], axis=0)

    augmentation_tags = f"adp_time_uniq3"
    return x_combined, y_combined, augmentation_tags


def adp_time_uniq4(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)
    x_window_warp = aug.window_warp(x_split[3].copy())

    x_combined = np.concatenate([x_split[0], x_split[1], x_split[2], np.tile(x_window_warp, (ratio, 1))], axis=0)
    y_combined = np.concatenate([y_split[0], y_split[1], y_split[2], np.tile(y_split[3], ratio)], axis=0)

    augmentation_tags = f"adp_time_uniq4"
    return x_combined, y_combined, augmentation_tags


# ————————————————————————————————————————————————————————
# TIME  —  MULTIPLE
# ————————————————————————————————————————————————————————
def adp_time_multi1(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)

    # Apply different augmentations to each split
    x_permutation = aug.permutation(x_split[0].copy())
    x_window_slice = aug.window_slice(x_split[1].copy())
    x_time_warp = aug.time_warp(x_split[2].copy())
    x_window_warp = aug.window_warp(x_split[3].copy())

    x_combined = np.concatenate([np.tile(x_permutation, (ratio, 1)),
                                 np.tile(x_window_slice, (ratio, 1)),
                                 np.tile(x_time_warp, (ratio, 1)),
                                 np.tile(x_window_warp, (ratio, 1))], axis=0)

    y_combined = np.concatenate([np.tile(y_split[i], ratio) for i in range(4)], axis=0)

    augmentation_tags = f"adp_time_multi1"
    return x_combined, y_combined, augmentation_tags


def adp_time_multi2(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_window_slice = aug.window_slice(x_split[0].copy())
    x_time_warp = aug.time_warp(x_split[1].copy())
    x_window_warp = aug.window_warp(x_split[2].copy())
    x_permutation = aug.permutation(x_split[3].copy())

    x_combined = np.concatenate([np.tile(x_window_slice, (ratio, 1)),
                                 np.tile(x_time_warp, (ratio, 1)),
                                 np.tile(x_window_warp, (ratio, 1)),
                                 np.tile(x_permutation, (ratio, 1))], axis=0)

    y_combined = np.concatenate([np.tile(y_split[i], ratio) for i in range(4)], axis=0)

    augmentation_tags = f"adp_time_multi2"
    return x_combined, y_combined, augmentation_tags


def adp_time_multi3(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_time_warp = aug.time_warp(x_split[0].copy())
    x_window_warp = aug.window_warp(x_split[1].copy())
    x_permutation = aug.permutation(x_split[2].copy())
    x_window_slice = aug.window_slice(x_split[3].copy())

    x_combined = np.concatenate([np.tile(x_time_warp, (ratio, 1)),
                                 np.tile(x_window_warp, (ratio, 1)),
                                 np.tile(x_permutation, (ratio, 1)),
                                 np.tile(x_window_slice, (ratio, 1))], axis=0)

    y_combined = np.concatenate([np.tile(y_split[i], ratio) for i in range(4)], axis=0)

    augmentation_tags = f"adp_time_multi3"
    return x_combined, y_combined, augmentation_tags


def adp_time_multi4(x, y, ratio=1):
    import utils.data_partitioning as partition
    import utils.augmentation as aug

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_window_warp = aug.window_warp(x_split[0].copy())
    x_permutation = aug.permutation(x_split[1].copy())
    x_window_slice = aug.window_slice(x_split[2].copy())
    x_time_warp = aug.time_warp(x_split[3].copy())

    x_combined = np.concatenate([np.tile(x_window_warp, (ratio, 1)),
                                 np.tile(x_permutation, (ratio, 1)),
                                 np.tile(x_window_slice, (ratio, 1)),
                                 np.tile(x_time_warp, (ratio, 1))], axis=0)

    y_combined = np.concatenate([np.tile(y_split[i], ratio) for i in range(4)], axis=0)

    augmentation_tags = f"adp_time_multi4"
    return x_combined, y_combined, augmentation_tags



# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║             Sequential Combined Augmentation Methods             ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def ads_sequential_combined1(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    # Duplicate and concatenate the data before applying augmentations
    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Permutation, Rotation, Time Warping, Scaling
    x_combined = aug.permutation(x_combined)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.scaling(x_combined)

    augmentation_tags = f"ads_sequential_combined1"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined2(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Permutation, Jittering, Time Warping, Magnitude Warping
    x_combined = aug.permutation(x_combined)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.magnitude_warp(x_combined)

    augmentation_tags = f"ads_sequential_combined2"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined3(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Permutation, Scaling, Window Slicing, Rotation
    x_combined = aug.permutation(x_combined)
    x_combined = aug.scaling(x_combined)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.rotation(x_combined)

    augmentation_tags = f"ads_sequential_combined3"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined4(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Permutation, Magnitude Warping, Window Slicing, Jittering
    x_combined = aug.permutation(x_combined)
    x_combined = aug.magnitude_warp(x_combined)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.jitter(x_combined)

    augmentation_tags = f"ads_sequential_combined4"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined5(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Window Slicing, Jittering, Window Warping, Scaling
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.scaling(x_combined)

    augmentation_tags = f"ads_sequential_combined5"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined6(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Window Slicing, Rotation, Time Warping, Magnitude Warping
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.magnitude_warp(x_combined)

    augmentation_tags = f"ads_sequential_combined6"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined7(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Window Slicing, Scaling, Window Warping, Jittering
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.scaling(x_combined)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.jitter(x_combined)

    augmentation_tags = f"ads_sequential_combined7"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined8(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Time Warping, Rotation, Window Warping, Scaling
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.scaling(x_combined)

    augmentation_tags = f"ads_sequential_combined8"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined9(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Time Warping, Jittering, Window Warping, Magnitude Warping
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.window_warp(x_combined)
    x_combined = aug.magnitude_warp(x_combined)

    augmentation_tags = f"ads_sequential_combined9"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined10(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Time Warping, Scaling, Window Slicing, Rotation
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.scaling(x_combined)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.rotation(x_combined)

    augmentation_tags = f"ads_sequential_combined10"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined11(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Time Warping, Rotation, Window Slicing, Magnitude Warping
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.rotation(x_combined)
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.magnitude_warp(x_combined)

    augmentation_tags = f"ads_sequential_combined11"
    return x_combined, y_combined, augmentation_tags


def ads_sequential_combined12(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_combined, y_combined = partition.duplicate_and_concatenate(x, y, ratio)

    # Apply: Window Slicing, Jittering, Time Warping, Rotation
    x_combined = aug.window_slice(x_combined)
    x_combined = aug.jitter(x_combined)
    x_combined = aug.time_warp(x_combined)
    x_combined = aug.rotation(x_combined)

    augmentation_tags = f"ads_sequential_combined12"
    return x_combined, y_combined, augmentation_tags


'''Parallel 
Combined 
Methods
'''


# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║             Parallel Combined Augmentation Methods               ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def adp_parallel_combined1(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_permutation = aug.permutation(x_split[0].copy())
    x_rotation = aug.rotation(x_split[1].copy())
    x_time_warp = aug.time_warp(x_split[2].copy())
    x_scaling = aug.scaling(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_permutation, ratio, axis=0),
        np.repeat(x_rotation, ratio, axis=0),
        np.repeat(x_time_warp, ratio, axis=0),
        np.repeat(x_scaling, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined1"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined2(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_permutation = aug.permutation(x_split[0].copy())
    x_jitter = aug.jitter(x_split[1].copy())
    x_time_warp = aug.time_warp(x_split[2].copy())
    x_magnitude_warp = aug.magnitude_warp(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_permutation, ratio, axis=0),
        np.repeat(x_jitter, ratio, axis=0),
        np.repeat(x_time_warp, ratio, axis=0),
        np.repeat(x_magnitude_warp, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined2"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined3(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_permutation = aug.permutation(x_split[0].copy())
    x_scaling = aug.scaling(x_split[1].copy())
    x_window_slice = aug.window_slice(x_split[2].copy())
    x_rotation = aug.rotation(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_permutation, ratio, axis=0),
        np.repeat(x_scaling, ratio, axis=0),
        np.repeat(x_window_slice, ratio, axis=0),
        np.repeat(x_rotation, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined3"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined4(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_permutation = aug.permutation(x_split[0].copy())
    x_magnitude_warp = aug.magnitude_warp(x_split[1].copy())
    x_window_slice = aug.window_slice(x_split[2].copy())
    x_jitter = aug.jitter(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_permutation, ratio, axis=0),
        np.repeat(x_magnitude_warp, ratio, axis=0),
        np.repeat(x_window_slice, ratio, axis=0),
        np.repeat(x_jitter, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined4"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined5(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_window_slice = aug.window_slice(x_split[0].copy())
    x_jitter = aug.jitter(x_split[1].copy())
    x_window_warp = aug.window_warp(x_split[2].copy())
    x_scaling = aug.scaling(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_window_slice, ratio, axis=0),
        np.repeat(x_jitter, ratio, axis=0),
        np.repeat(x_window_warp, ratio, axis=0),
        np.repeat(x_scaling, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined5"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined6(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_window_slice = aug.window_slice(x_split[0].copy())
    x_rotation = aug.rotation(x_split[1].copy())
    x_time_warp = aug.time_warp(x_split[2].copy())
    x_magnitude_warp = aug.magnitude_warp(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_window_slice, ratio, axis=0),
        np.repeat(x_rotation, ratio, axis=0),
        np.repeat(x_time_warp, ratio, axis=0),
        np.repeat(x_magnitude_warp, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined6"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined7(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_window_slice = aug.window_slice(x_split[0].copy())
    x_scaling = aug.scaling(x_split[1].copy())
    x_window_warp = aug.window_warp(x_split[2].copy())
    x_jitter = aug.jitter(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_window_slice, ratio, axis=0),
        np.repeat(x_scaling, ratio, axis=0),
        np.repeat(x_window_warp, ratio, axis=0),
        np.repeat(x_jitter, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined7"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined8(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_time_warp = aug.time_warp(x_split[0].copy())
    x_rotation = aug.rotation(x_split[1].copy())
    x_window_warp = aug.window_warp(x_split[2].copy())
    x_scaling = aug.scaling(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_time_warp, ratio, axis=0),
        np.repeat(x_rotation, ratio, axis=0),
        np.repeat(x_window_warp, ratio, axis=0),
        np.repeat(x_scaling, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined8"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined9(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_time_warp = aug.time_warp(x_split[0].copy())
    x_jitter = aug.jitter(x_split[1].copy())
    x_window_warp = aug.window_warp(x_split[2].copy())
    x_magnitude_warp = aug.magnitude_warp(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_time_warp, ratio, axis=0),
        np.repeat(x_jitter, ratio, axis=0),
        np.repeat(x_window_warp, ratio, axis=0),
        np.repeat(x_magnitude_warp, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined9"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined10(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_time_warp = aug.time_warp(x_split[0].copy())
    x_scaling = aug.scaling(x_split[1].copy())
    x_window_slice = aug.window_slice(x_split[2].copy())
    x_rotation = aug.rotation(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_time_warp, ratio, axis=0),
        np.repeat(x_scaling, ratio, axis=0),
        np.repeat(x_window_slice, ratio, axis=0),
        np.repeat(x_rotation, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined10"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined11(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    x_time_warp = aug.time_warp(x_split[0].copy())
    x_rotation = aug.rotation(x_split[1].copy())
    x_window_slice = aug.window_slice(x_split[2].copy())
    x_magnitude_warp = aug.magnitude_warp(x_split[3].copy())

    x_combined = np.concatenate([
        np.repeat(x_time_warp, ratio, axis=0),
        np.repeat(x_rotation, ratio, axis=0),
        np.repeat(x_window_slice, ratio, axis=0),
        np.repeat(x_magnitude_warp, ratio, axis=0)
    ])
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined11"
    return x_combined, y_combined, augmentation_tags


def adp_parallel_combined12(x, y, ratio=1):
    import utils.augmentation as aug
    import utils.data_partitioning as partition

    x_split, y_split = partition.divide_dataset(x, y, 4)

    # Apply: Window Slicing, Jittering, Time Warping, Rotation
    x_window_slice = aug.window_slice(x_split[0].copy())
    x_jitter = aug.jitter(x_split[1].copy())
    x_time_warp = aug.time_warp(x_split[2].copy())
    x_rotation = aug.rotation(x_split[3].copy())

    # Explicitly repeat each array along the samples axis
    x_combined = np.concatenate([
        np.repeat(x_window_slice, ratio, axis=0),
        np.repeat(x_jitter, ratio, axis=0),
        np.repeat(x_time_warp, ratio, axis=0),
        np.repeat(x_rotation, ratio, axis=0)
    ])

    # Similarly, repeat the corresponding labels
    y_combined = np.concatenate([
        np.repeat(y_split[i], ratio, axis=0) for i in range(4)
    ])

    augmentation_tags = "adp_parallel_combined12"
    return x_combined, y_combined, augmentation_tags

