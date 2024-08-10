#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/10/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""
import numpy as np


def divide_dataset(x, y, n_parts):
    """
    Divides the dataset into n equal parts.
    :param x: Input data
    :param y: Labels
    :param n_parts: Number of parts to divide the dataset into
    :return: List of divided datasets and labels
    """
    x_split = np.array_split(x, n_parts)
    y_split = np.array_split(y, n_parts)
    return x_split, y_split



def duplicate_and_concatenate(x, y, ratio):
    """
    Duplicates the dataset based on the ratio and concatenates it to form a larger dataset.
    :param x: Input data
    :param y: Labels
    :param ratio: Number of times the dataset should be duplicated
    :return: Duplicated and concatenated dataset and labels
    """
    x_augmented = np.tile(x, (ratio, 1))
    y_augmented = np.tile(y, ratio)
    return x_augmented, y_augmented

