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

import torch
import torch.nn as nn


def get_model(model_name, input_shape, nb_class):
    if model_name == "lstm1":
        model = SimpleLSTM(input_shape, nb_class)
    else:
        raise ValueError(f"Model {model_name} not found")
    return model


class SimpleLSTM(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(SimpleLSTM, self).__init__()
        self.nb_dims = input_shape[1]
        self.lstm = nn.LSTM(self.nb_dims, 100, batch_first=True)
        self.fc = nn.Linear(100, nb_class)

    def forward(self, x):
        # x shape: (batch, nb_timesteps, nb_dims)
        out, (hn, cn) = self.lstm(x)  # LSTM output
        out = out[:, -1, :]  # Take the output of the last timestep
        out = self.fc(out)  # Pass through a linear layer
        return torch.softmax(out, dim=1)  # Softmax to get probabilities
