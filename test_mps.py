#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/03/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (mps_device)
    print (x)
else:
    print ("MPS device not found.")

