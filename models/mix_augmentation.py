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

import numpy as np
import re

def run_augmentation(x, y, args):
    print("Augmenting %s" % args.dataset)
    np.random.seed(args.seed)
    x_aug = x
    y_aug = y
    augmentation_tags = args.extra_tag

    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            if 'sequential' in args.augmentation_method:
                # For sequential methods, use the updated data from the previous round
                x_temp, y_temp, temp_tags = augment_sequential(x_aug, y_aug, args)
            else:
                # For simple and parallel methods, use the original data
                if args.augmentation_method == 'simple':
                    x_temp, temp_tags = augment_data_simple(x, args)
                elif 'parallel_magnitude' in args.augmentation_method:
                    method_num = re.search(r'\d+', args.augmentation_method).group()
                    x_temp, y_temp, temp_tags = globals()[f'augment_data_parallel_magnitude{method_num}'](x, y)
                elif 'parallel_time' in args.augmentation_method:
                    method_num = re.search(r'\d+', args.augmentation_method).group()
                    x_temp, y_temp, temp_tags = globals()[f'augment_data_parallel_time{method_num}'](x, y)
                elif 'parallel_combined' in args.augmentation_method:
                    method_num = re.search(r'\d+', args.augmentation_method).group()
                    x_temp, y_temp, temp_tags = globals()[f'augment_data_parallel_combined{method_num}'](x, y)
                else:
                    raise ValueError("Unknown augmentation method")

            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y_temp, axis=0)
            print("Round %d: %s done" % (n, augmentation_tags + temp_tags))

        augmentation_tags += temp_tags
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    return x_aug, y_aug, augmentation_tags

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


def augment_sequential(x, y, args):
    if 'sequential_magnitude' in args.augmentation_method:
        method_num = args.augmentation_method[-1]
        return globals()[f'augment_data_sequential_magnitude{method_num}'](x, y)
    elif 'sequential_time' in args.augmentation_method:
        method_num = args.augmentation_method[-1]
        return globals()[f'augment_data_sequential_time{method_num}'](x, y)
    elif 'sequential_combined' in args.augmentation_method:
        method_num = args.augmentation_method[-1]
        return globals()[f'augment_data_sequential_combined{method_num}'](x, y)
    else:
        raise ValueError("Unknown sequential augmentation method")


def augment_parallel(x, y, args):
    if 'parallel_magnitude' in args.augmentation_method:
        method_num = args.augmentation_method[-1]
        return globals()[f'augment_data_parallel_magnitude{method_num}'](x, y)
    elif 'parallel_time' in args.augmentation_method:
        method_num = args.augmentation_method[-1]
        return globals()[f'augment_data_parallel_time{method_num}'](x, y)
    elif 'parallel_combined' in args.augmentation_method:
        method_num = args.augmentation_method[-1]
        return globals()[f'augment_data_parallel_combined{method_num}'](x, y)
    else:
        raise ValueError("Unknown parallel augmentation method")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║             Sequential Augmentation Methods                      ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def augment_data_sequential_magnitude1(x, y):
    import utils.augmentation as aug
    x = aug.magnitude_warp(x)
    x = aug.magnitude_warp(x)
    x = aug.magnitude_warp(x)
    x = aug.magnitude_warp(x)
    augmentation_tags = "_augment_data_sequential_magnitude1"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude2(x, y):
    import utils.augmentation as aug
    x = aug.scaling(x)
    x = aug.scaling(x)
    x = aug.scaling(x)
    x = aug.scaling(x)
    augmentation_tags = "_augment_data_sequential_magnitude2"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude3(x, y):
    import utils.augmentation as aug
    x = aug.jitter(x)
    x = aug.jitter(x)
    x = aug.jitter(x)
    x = aug.jitter(x)
    augmentation_tags = "_augment_data_sequential_magnitude3"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude4(x, y):
    import utils.augmentation as aug
    x = aug.rotation(x)
    x = aug.rotation(x)
    x = aug.rotation(x)
    x = aug.rotation(x)
    augmentation_tags = "_augment_data_sequential_magnitude4"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude5(x, y):
    import utils.augmentation as aug
    x = aug.jitter(x)
    x = aug.scaling(x)
    x = aug.rotation(x)
    x = aug.magnitude_warp(x)
    augmentation_tags = "_augment_data_sequential_magnitude5"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude6(x, y):
    import utils.augmentation as aug
    x = aug.scaling(x, sigma=0.05)
    x = aug.rotation(x)
    x = aug.magnitude_warp(x, sigma=0.2, knot=4)
    x = aug.jitter(x, sigma=0.03)
    augmentation_tags = "_augment_data_sequential_magnitude6"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude7(x, y):
    import utils.augmentation as aug
    x = aug.rotation(x)
    x = aug.magnitude_warp(x)
    x = aug.scaling(x)
    x = aug.jitter(x)
    augmentation_tags = "_augment_data_sequential_magnitude7"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude8(x, y):
    import utils.augmentation as aug
    x = aug.magnitude_warp(x)
    x = aug.scaling(x)
    x = aug.jitter(x)
    x = aug.rotation(x)
    augmentation_tags = "_augment_data_sequential_magnitude8"
    return x, y, augmentation_tags


def augment_data_sequential_time1(x, y):
    import utils.augmentation as aug
    x = aug.window_slice(x)
    x = aug.window_slice(x)
    x = aug.window_slice(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_time1"
    return x, y, augmentation_tags


def augment_data_sequential_time2(x, y):
    import utils.augmentation as aug
    x = aug.permutation(x)
    x = aug.permutation(x)
    x = aug.permutation(x)
    x = aug.permutation(x)
    augmentation_tags = "_augment_data_sequential_time2"
    return x, y, augmentation_tags


def augment_data_sequential_time3(x, y):
    import utils.augmentation as aug
    x = aug.time_warp(x)
    x = aug.time_warp(x)
    x = aug.time_warp(x)
    x = aug.time_warp(x)
    augmentation_tags = "_augment_data_sequential_time3"
    return x, y, augmentation_tags


def augment_data_sequential_time4(x, y):
    import utils.augmentation as aug
    x = aug.window_warp(x)
    x = aug.window_warp(x)
    x = aug.window_warp(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_time4"
    return x, y, augmentation_tags


def augment_data_sequential_time5(x, y):
    import utils.augmentation as aug
    x = aug.permutation(x)
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_time5"
    return x, y, augmentation_tags


def augment_data_sequential_time6(x, y):
    import utils.augmentation as aug
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    x = aug.window_slice(x)
    x = aug.permutation(x)
    augmentation_tags = "_augment_data_sequential_time6"
    return x, y, augmentation_tags


def augment_data_sequential_time7(x, y):
    import utils.augmentation as aug
    x = aug.window_warp(x)
    x = aug.window_slice(x)
    x = aug.permutation(x)
    x = aug.time_warp(x)
    augmentation_tags = "_augment_data_sequential_time7"
    return x, y, augmentation_tags


def augment_data_sequential_time8(x, y):
    import utils.augmentation as aug
    x = aug.window_slice(x)
    x = aug.permutation(x)
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_time8"
    return x, y, augmentation_tags



# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║             Parallel Augmentation Methods                        ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def augment_data_parallel_magnitude1(x, y):
    import utils.augmentation as aug
    x_combined = []
    y_combined = []

    for _ in range(4):
        x_aug = aug.jitter(x.copy())
        x_combined.append(x_aug)
        y_combined.append(y)

    x_combined = np.concatenate(x_combined)
    y_combined = np.concatenate(y_combined)
    augmentation_tags = "_augment_data_parallel_magnitude1"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude2(x, y):
    import utils.augmentation as aug
    x_combined = []
    y_combined = []

    for _ in range(4):
        x_aug = aug.rotation(x.copy())
        x_combined.append(x_aug)
        y_combined.append(y)

    x_combined = np.concatenate(x_combined)
    y_combined = np.concatenate(y_combined)
    augmentation_tags = "_augment_data_parallel_magnitude2"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude3(x, y):
    import utils.augmentation as aug
    x_combined = []
    y_combined = []

    for _ in range(4):
        x_aug = aug.scaling(x.copy())
        x_combined.append(x_aug)
        y_combined.append(y)

    x_combined = np.concatenate(x_combined)
    y_combined = np.concatenate(y_combined)
    augmentation_tags = "_augment_data_parallel_magnitude3"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude4(x, y):
    import utils.augmentation as aug
    x_combined = []
    y_combined = []

    for _ in range(4):
        x_aug = aug.magnitude_warp(x.copy())
        x_combined.append(x_aug)
        y_combined.append(y)

    x_combined = np.concatenate(x_combined)
    y_combined = np.concatenate(y_combined)
    augmentation_tags = "_augment_data_parallel_magnitude4"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude5(x, y):
    import utils.augmentation as aug
    x_jitter = aug.jitter(x.copy())
    x_scaling = aug.scaling(x.copy())
    x_rotation = aug.rotation(x.copy())
    x_magnitude_warp = aug.magnitude_warp(x.copy())

    x_combined = np.concatenate((x_jitter, x_scaling, x_rotation, x_magnitude_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_magnitude5"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude6(x, y):
    import utils.augmentation as aug
    x_scaling = aug.scaling(x.copy())
    x_rotation = aug.rotation(x.copy())
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_jitter = aug.jitter(x.copy())

    x_combined = np.concatenate((x_scaling, x_rotation, x_magnitude_warp, x_jitter))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_magnitude6"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude7(x, y):
    import utils.augmentation as aug
    x_rotation = aug.rotation(x.copy())
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_scaling = aug.scaling(x.copy())
    x_jitter = aug.jitter(x.copy())

    x_combined = np.concatenate((x_rotation, x_magnitude_warp, x_scaling, x_jitter))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_magnitude7"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude8(x, y):
    import utils.augmentation as aug
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_scaling = aug.scaling(x.copy())
    x_jitter = aug.jitter(x.copy())
    x_rotation = aug.rotation(x.copy())

    x_combined = np.concatenate((x_magnitude_warp, x_scaling, x_jitter, x_rotation))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_magnitude8"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude1(x, y):
    import utils.augmentation as aug
    x_combined = []
    y_combined = []

    for _ in range(4):
        x_aug = aug.permutation(x.copy())
        x_combined.append(x_aug)
        y_combined.append(y)

    x_combined = np.concatenate(x_combined)
    y_combined = np.concatenate(y_combined)
    augmentation_tags = "_augment_data_parallel_magnitude1"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude2(x, y):
    import utils.augmentation as aug
    x_combined = []
    y_combined = []

    for _ in range(4):
        x_aug = aug.time_warp(x.copy())
        x_combined.append(x_aug)
        y_combined.append(y)

    x_combined = np.concatenate(x_combined)
    y_combined = np.concatenate(y_combined)
    augmentation_tags = "_augment_data_parallel_magnitude2"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude3(x, y):
    import utils.augmentation as aug
    x_combined = []
    y_combined = []

    for _ in range(4):
        x_aug = aug.window_warp(x.copy())
        x_combined.append(x_aug)
        y_combined.append(y)

    x_combined = np.concatenate(x_combined)
    y_combined = np.concatenate(y_combined)
    augmentation_tags = "_augment_data_parallel_magnitude3"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude4(x, y):
    import utils.augmentation as aug
    x_combined = []
    y_combined = []

    for _ in range(4):
        x_aug = aug.window_slice(x.copy())
        x_combined.append(x_aug)
        y_combined.append(y)

    x_combined = np.concatenate(x_combined)
    y_combined = np.concatenate(y_combined)
    augmentation_tags = "_augment_data_parallel_magnitude4"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_time5(x, y):
    import utils.augmentation as aug
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())

    x_combined = np.concatenate((x_permutation, x_time_warp, x_window_warp, x_window_slice))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_time5"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_time6(x, y):
    import utils.augmentation as aug
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())

    x_combined = np.concatenate((x_time_warp, x_window_warp, x_window_slice, x_permutation))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_time6"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_time7(x, y):
    import utils.augmentation as aug
    x_window_warp = aug.window_warp(x.copy(), window_ratio=0.05, scales=[0.1, 2.])
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())

    x_combined = np.concatenate((x_window_warp, x_window_slice, x_permutation, x_time_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_time7"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_time8(x, y):
    import utils.augmentation as aug
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())

    x_combined = np.concatenate((x_window_slice, x_permutation, x_time_warp, x_window_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_time8"
    return x_combined, y_combined, augmentation_tags



# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║             Sequential Combined Augmentation Methods             ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def augment_data_sequential_combined1(x, y):
    import utils.augmentation as aug
    x = aug.jitter(x)
    x = aug.permutation(x)
    x = aug.scaling(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_combined1"
    return x, y, augmentation_tags


def augment_data_sequential_combined2(x, y):
    import utils.augmentation as aug
    x = aug.scaling(x)
    x = aug.time_warp(x)
    x = aug.rotation(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_combined2"
    return x, y, augmentation_tags


def augment_data_sequential_combined3(x, y):
    import utils.augmentation as aug
    x = aug.rotation(x)
    x = aug.window_warp(x)
    x = aug.jitter(x)
    x = aug.permutation(x)
    augmentation_tags = "_augment_data_sequential_combined3"
    return x, y, augmentation_tags


def augment_data_sequential_combined4(x, y):
    import utils.augmentation as aug
    x = aug.magnitude_warp(x)
    x = aug.time_warp(x)
    x = aug.scaling(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_combined4"
    return x, y, augmentation_tags


def augment_data_sequential_combined5(x, y):
    import utils.augmentation as aug
    x = aug.scaling(x)
    x = aug.jitter(x)
    x = aug.permutation(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_combined5"
    return x, y, augmentation_tags


def augment_data_sequential_combined6(x, y):
    import utils.augmentation as aug
    x = aug.jitter(x)
    x = aug.time_warp(x)
    x = aug.magnitude_warp(x)
    x = aug.rotation(x)
    augmentation_tags = "_augment_data_sequential_combined6"
    return x, y, augmentation_tags


def augment_data_sequential_combined7(x, y):
    import utils.augmentation as aug
    x = aug.magnitude_warp(x)
    x = aug.scaling(x)
    x = aug.permutation(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_combined7"
    return x, y, augmentation_tags


def augment_data_sequential_combined8(x, y):
    import utils.augmentation as aug
    x = aug.rotation(x)
    x = aug.jitter(x)
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_combined8"
    return x, y, augmentation_tags


def augment_data_sequential_combined9(x, y):
    import utils.augmentation as aug
    x = aug.permutation(x)
    x = aug.time_warp(x)
    x = aug.magnitude_warp(x)
    x = aug.scaling(x)
    augmentation_tags = "_augment_data_sequential_combined9"
    return x, y, augmentation_tags


def augment_data_sequential_combined10(x, y):
    import utils.augmentation as aug
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    x = aug.rotation(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_combined10"
    return x, y, augmentation_tags


def augment_data_sequential_combined11(x, y):
    import utils.augmentation as aug
    x = aug.window_warp(x)
    x = aug.window_slice(x)
    x = aug.permutation(x)
    x = aug.time_warp(x)
    augmentation_tags = "_augment_data_sequential_combined11"
    return x, y, augmentation_tags


def augment_data_sequential_combined12(x, y):
    import utils.augmentation as aug
    x = aug.window_slice(x)
    x = aug.permutation(x)
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_combined12"
    return x, y, augmentation_tags


def augment_data_sequential_combined13(x, y):
    import utils.augmentation as aug
    x = aug.permutation(x)
    x = aug.window_slice(x)
    x = aug.window_warp(x)
    x = aug.time_warp(x)
    augmentation_tags = "_augment_data_sequential_combined13"
    return x, y, augmentation_tags


def augment_data_sequential_combined14(x, y):
    import utils.augmentation as aug
    x = aug.time_warp(x)
    x = aug.permutation(x)
    x = aug.window_slice(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_combined14"
    return x, y, augmentation_tags


def augment_data_sequential_combined15(x, y):
    import utils.augmentation as aug
    x = aug.window_slice(x)
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    x = aug.permutation(x)
    augmentation_tags = "_augment_data_sequential_combined15"
    return x, y, augmentation_tags


def augment_data_sequential_combined16(x, y):
    import utils.augmentation as aug
    x = aug.window_warp(x)
    x = aug.time_warp(x)
    x = aug.permutation(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_combined16"
    return x, y, augmentation_tags


def augment_data_sequential_combined17(x, y):
    import utils.augmentation as aug
    x = aug.permutation(x)
    x = aug.window_warp(x)
    x = aug.time_warp(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_combined17"
    return x, y, augmentation_tags


def augment_data_sequential_combined18(x, y):
    import utils.augmentation as aug
    x = aug.time_warp(x)
    x = aug.permutation(x)
    x = aug.window_warp(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_combined18"
    return x, y, augmentation_tags


def augment_data_sequential_combined19(x, y):
    import utils.augmentation as aug
    x = aug.window_slice(x)
    x = aug.window_warp(x)
    x = aug.time_warp(x)
    x = aug.permutation(x)
    augmentation_tags = "_augment_data_sequential_combined19"
    return x, y, augmentation_tags


def augment_data_sequential_combined20(x, y):
    import utils.augmentation as aug
    x = aug.window_warp(x)
    x = aug.permutation(x)
    x = aug.window_slice(x)
    x = aug.time_warp(x)
    augmentation_tags = "_augment_data_sequential_combined20"
    return x, y, augmentation_tags


'''Parallel 
Combined 
Methods
'''

# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║             Parallel Combined Augmentation Methods               ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def augment_data_parallel_combined1(x, y):
    import utils.augmentation as aug
    x_jitter = aug.jitter(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_scaling = aug.scaling(x.copy())
    x_window_warp = aug.window_warp(x.copy())

    x_combined = np.concatenate((x_jitter, x_permutation, x_scaling, x_window_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined1"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined2(x, y):
    import utils.augmentation as aug
    x_scaling = aug.scaling(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_rotation = aug.rotation(x.copy())
    x_window_slice = aug.window_slice(x.copy())

    x_combined = np.concatenate((x_scaling, x_time_warp, x_rotation, x_window_slice))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined2"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined3(x, y):
    import utils.augmentation as aug
    x_rotation = aug.rotation(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_jitter = aug.jitter(x.copy())
    x_permutation = aug.permutation(x.copy())

    x_combined = np.concatenate((x_rotation, x_window_warp, x_jitter, x_permutation))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined3"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined4(x, y):
    import utils.augmentation as aug
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_scaling = aug.scaling(x.copy())
    x_window_slice = aug.window_slice(x.copy())

    x_combined = np.concatenate((x_magnitude_warp, x_time_warp, x_scaling, x_window_slice))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined4"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined5(x, y):
    import utils.augmentation as aug
    x_scaling = aug.scaling(x.copy())
    x_jitter = aug.jitter(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_window_warp = aug.window_warp(x.copy())

    x_combined = np.concatenate((x_scaling, x_jitter, x_permutation, x_window_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined5"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined6(x, y):
    import utils.augmentation as aug
    x_jitter = aug.jitter(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_rotation = aug.rotation(x.copy())

    x_combined = np.concatenate((x_jitter, x_time_warp, x_magnitude_warp, x_rotation))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined6"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined7(x, y):
    import utils.augmentation as aug
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_scaling = aug.scaling(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_window_slice = aug.window_slice(x.copy())

    x_combined = np.concatenate((x_magnitude_warp, x_scaling, x_permutation, x_window_slice))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined7"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined8(x, y):
    import utils.augmentation as aug
    x_rotation = aug.rotation(x.copy())
    x_jitter = aug.jitter(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())

    x_combined = np.concatenate((x_rotation, x_jitter, x_time_warp, x_window_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined8"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined9(x, y):
    import utils.augmentation as aug
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_scaling = aug.scaling(x.copy())

    x_combined = np.concatenate((x_permutation, x_time_warp, x_magnitude_warp, x_scaling))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined9"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined10(x, y):
    import utils.augmentation as aug
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_rotation = aug.rotation(x.copy())
    x_window_slice = aug.window_slice(x.copy())

    x_combined = np.concatenate((x_time_warp, x_window_warp, x_rotation, x_window_slice))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined10"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined11(x, y):
    import utils.augmentation as aug
    x_window_warp = aug.window_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())

    x_combined = np.concatenate((x_window_warp, x_window_slice, x_permutation, x_time_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined11"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined12(x, y):
    import utils.augmentation as aug
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())

    x_combined = np.concatenate((x_window_slice, x_permutation, x_time_warp, x_window_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined12"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined13(x, y):
    import utils.augmentation as aug
    x_permutation = aug.permutation(x.copy())
    x_window_slice = aug.window_slice(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_time_warp = aug.time_warp(x.copy())

    x_combined = np.concatenate((x_permutation, x_window_slice, x_window_warp, x_time_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined13"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined14(x, y):
    import utils.augmentation as aug
    x_time_warp = aug.time_warp(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_window_slice = aug.window_slice(x.copy())
    x_window_warp = aug.window_warp(x.copy())

    x_combined = np.concatenate((x_time_warp, x_permutation, x_window_slice, x_window_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined14"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined15(x, y):
    import utils.augmentation as aug
    x_window_warp = aug.window_warp(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())

    x_combined = np.concatenate((x_window_warp, x_time_warp, x_window_slice, x_permutation))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined15"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined16(x, y):
    import utils.augmentation as aug
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())

    x_combined = np.concatenate((x_window_slice, x_permutation, x_time_warp, x_window_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined16"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined17(x, y):
    import utils.augmentation as aug
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())

    x_combined = np.concatenate((x_permutation, x_time_warp, x_window_warp, x_window_slice))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined17"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined18(x, y):
    import utils.augmentation as aug
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())

    x_combined = np.concatenate((x_time_warp, x_window_warp, x_window_slice, x_permutation))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined18"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined19(x, y):
    import utils.augmentation as aug
    x_window_slice = aug.window_slice(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_permutation = aug.permutation(x.copy())

    x_combined = np.concatenate((x_window_slice, x_window_warp, x_time_warp, x_permutation))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined19"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined20(x, y):
    import utils.augmentation as aug
    x_window_warp = aug.window_warp(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_window_slice = aug.window_slice(x.copy())
    x_time_warp = aug.time_warp(x.copy())

    x_combined = np.concatenate((x_window_warp, x_permutation, x_window_slice, x_time_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_combined20"
    return x_combined, y_combined, augmentation_tags
