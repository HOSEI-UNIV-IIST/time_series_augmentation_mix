#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/09/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

ucr_data = [
    "CBF",
    "ECG200",
    "ECG5000",
    "FordB",
    "GunPointAgeSpan",
    #"ScreenType",
    "Strawberry",
    "Yoga",
]

techniques_groups = {
    'Sequential': [
        "ads_sequential_combined1_1x", "ads_sequential_combined2_1x",
        "ads_sequential_combined3_1x", "ads_sequential_combined4_1x",
        "ads_sequential_combined5_1x", "ads_sequential_combined6_1x",
        "ads_sequential_combined7_1x", "ads_sequential_combined8_1x",
        "ads_sequential_combined9_1x", "ads_sequential_combined10_1x",
        "ads_sequential_combined11_1x", "ads_sequential_combined12_1x"
    ],
    'Parallel': [
        "adp_parallel_combined1_1x", "adp_parallel_combined2_1x",
        "adp_parallel_combined3_1x", "adp_parallel_combined4_1x",
        "adp_parallel_combined5_1x", "adp_parallel_combined6_1x",
        "adp_parallel_combined7_1x", "adp_parallel_combined8_1x",
        "adp_parallel_combined9_1x", "adp_parallel_combined10_1x",
        "adp_parallel_combined11_1x", "adp_parallel_combined12_1x"
    ],
    'Time-based': [
        "ads_time_uniq1_1x", "ads_time_uniq2_1x",
        "ads_time_uniq3_1x", "ads_time_uniq4_1x",
        "ads_time_multi1_1x", "ads_time_multi2_1x",
        "ads_time_multi3_1x", "ads_time_multi4_1x",
        "adp_time_uniq1_1x", "adp_time_uniq2_1x",
        "adp_time_uniq3_1x", "adp_time_uniq4_1x",
        "adp_time_multi1_1x", "adp_time_multi2_1x",
        "adp_time_multi3_1x", "adp_time_multi4_1x"
    ],
    'Magnitude-based': [
        "ads_magnitude_uniq1_1x", "ads_magnitude_uniq2_1x",
        "ads_magnitude_uniq3_1x", "ads_magnitude_uniq4_1x",
        "ads_magnitude_multi1_1x", "ads_magnitude_multi2_1x",
        "ads_magnitude_multi3_1x", "ads_magnitude_multi4_1x",
        "adp_magnitude_uniq1_1x", "adp_magnitude_uniq2_1x",
        "adp_magnitude_uniq3_1x", "adp_magnitude_uniq4_1x",
        "adp_magnitude_multi1_1x", "adp_magnitude_multi2_1x",
        "adp_magnitude_multi3_1x", "adp_magnitude_multi4_1x"
    ]
}

technique_abbreviations = {
    "ads_sequential_combined1_1x": "ads_c1",
    "ads_sequential_combined2_1x": "ads_c2",
    "ads_sequential_combined3_1x": "ads_c3",
    "ads_sequential_combined4_1x": "ads_c4",
    "ads_sequential_combined5_1x": "ads_c5",
    "ads_sequential_combined6_1x": "ads_c6",
    "ads_sequential_combined7_1x": "ads_c7",
    "ads_sequential_combined8_1x": "ads_c8",
    "ads_sequential_combined9_1x": "ads_c9",
    "ads_sequential_combined10_1x": "ads_c10",
    "ads_sequential_combined11_1x": "ads_c11",
    "ads_sequential_combined12_1x": "ads_c12",
    "adp_parallel_combined1_1x": "adp_c1",
    "adp_parallel_combined2_1x": "adp_c2",
    "adp_parallel_combined3_1x": "adp_c3",
    "adp_parallel_combined4_1x": "adp_c4",
    "adp_parallel_combined5_1x": "adp_c5",
    "adp_parallel_combined6_1x": "adp_c6",
    "adp_parallel_combined7_1x": "adp_c7",
    "adp_parallel_combined8_1x": "adp_c8",
    "adp_parallel_combined9_1x": "adp_c9",
    "adp_parallel_combined10_1x": "adp_c10",
    "adp_parallel_combined11_1x": "adp_c11",
    "adp_parallel_combined12_1x": "adp_c12",
    "ads_time_uniq1_1x": "ads_tim_u1",
    "ads_time_uniq2_1x": "ads_tim_u2",
    "ads_time_uniq3_1x": "ads_tim_u3",
    "ads_time_uniq4_1x": "ads_tim_u4",
    "ads_time_multi1_1x": "ads_tim_m1",
    "ads_time_multi2_1x": "ads_tim_m2",
    "ads_time_multi3_1x": "ads_tim_m3",
    "ads_time_multi4_1x": "ads_tim_m4",
    "adp_time_uniq1_1x": "adp_tim_u1",
    "adp_time_uniq2_1x": "adp_tim_u2",
    "adp_time_uniq3_1x": "adp_tim_u3",
    "adp_time_uniq4_1x": "adp_tim_u4",
    "adp_time_multi1_1x": "adp_tim_m1",
    "adp_time_multi2_1x": "adp_tim_m2",
    "adp_time_multi3_1x": "adp_tim_m3",
    "adp_time_multi4_1x": "adp_tim_m4",
    "ads_magnitude_uniq1_1x": "ads_mag_u1",
    "ads_magnitude_uniq2_1x": "ads_mag_u2",
    "ads_magnitude_uniq3_1x": "ads_mag_u3",
    "ads_magnitude_uniq4_1x": "ads_mag_u4",
    "ads_magnitude_multi1_1x": "ads_mag_m1",
    "ads_magnitude_multi2_1x": "ads_mag_m2",
    "ads_magnitude_multi3_1x": "ads_mag_m3",
    "ads_magnitude_multi4_1x": "ads_mag_m4",
    "adp_magnitude_uniq1_1x": "adp_mag_u1",
    "adp_magnitude_uniq2_1x": "adp_mag_u2",
    "adp_magnitude_uniq3_1x": "adp_mag_u3",
    "adp_magnitude_uniq4_1x": "adp_mag_u4",
    "adp_magnitude_multi1_1x": "adp_mag_m1",
    "adp_magnitude_multi2_1x": "adp_mag_m2",
    "adp_magnitude_multi3_1x": "adp_mag_m3",
    "adp_magnitude_multi4_1x": "adp_mag_m4"
}
