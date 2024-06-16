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

from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(model_name, input_shape, nb_class):
    if model_name == "vgg":
        model = VGG(input_shape, nb_class)
    elif model_name == "fcnn":
        model = FCNN(input_shape, nb_class)
    elif model_name == "resnet":
        model = ResNet(input_shape, nb_class)
    else:
        raise ValueError(f"Model {model_name} not found")
    return model


class VGG(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(VGG, self).__init__()
        nb_cnn = int(round(log(input_shape[0], 2)) - 3)
        print("Pooling layers:", nb_cnn)

        layers = []
        in_channels = input_shape[1]  # input_shape mus be is (timesteps, features)

        for i in range(nb_cnn):
            num_filters = min(64 * 2 ** i, 512)
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if i > 1:
                layers.append(nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = num_filters

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        conv_output_size = num_filters * (input_shape[0] // (2 ** nb_cnn))
        self.fc1 = nn.Linear(conv_output_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, nb_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from (batch, timesteps, features) to (batch, features, timesteps)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class FCNN(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(FCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1], 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, nb_class)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout4(x)
        x = self.fc4(x)
        return torch.softmax(x, dim=1)


class ResNet(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(ResNet, self).__init__()
        self.in_channels = input_shape[1]  # input_shape is (timesteps, features)

        # Residual Block 1
        self.block1_conv1 = nn.Conv1d(self.in_channels, 64, kernel_size=8, padding=4)
        self.block1_bn1 = nn.BatchNorm1d(64)
        self.block1_conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.block1_bn2 = nn.BatchNorm1d(64)
        self.block1_conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.block1_bn3 = nn.BatchNorm1d(64)
        self.block1_residual = nn.Conv1d(self.in_channels, 64, kernel_size=1)

        # Residual Block 2
        self.block2_conv1 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.block2_bn1 = nn.BatchNorm1d(128)
        self.block2_conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.block2_bn2 = nn.BatchNorm1d(128)
        self.block2_conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.block2_bn3 = nn.BatchNorm1d(128)
        self.block2_residual = nn.Conv1d(64, 128, kernel_size=1)

        # Residual Block 3
        self.block3_conv1 = nn.Conv1d(128, 128, kernel_size=8, padding=4)
        self.block3_bn1 = nn.BatchNorm1d(128)
        self.block3_conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.block3_bn2 = nn.BatchNorm1d(128)
        self.block3_conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.block3_bn3 = nn.BatchNorm1d(128)
        self.block3_residual = nn.Conv1d(128, 128, kernel_size=1)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(128, nb_class)

    def forward(self, x):
        # x shape: (batch_size, timesteps, features)
        x = x.permute(0, 2, 1)  # Change to (batch_size, features, timesteps)

        # Block 1
        residual = self.block1_residual(x)
        x = F.relu(self.block1_bn1(self.block1_conv1(x)))
        x = F.relu(self.block1_bn2(self.block1_conv2(x)))
        x = F.relu(self.block1_bn3(self.block1_conv3(x)))
        if x.shape[-1] != residual.shape[-1]:
            residual = F.pad(residual, (0, x.shape[-1] - residual.shape[-1]))
        x = x + residual  # Add residual connection

        # Block 2
        residual = self.block2_residual(x)
        x = F.relu(self.block2_bn1(self.block2_conv1(x)))
        x = F.relu(self.block2_bn2(self.block2_conv2(x)))
        x = F.relu(self.block2_bn3(self.block2_conv3(x)))
        if x.shape[-1] != residual.shape[-1]:
            residual = F.pad(residual, (0, x.shape[-1] - residual.shape[-1]))
        x = x + residual  # Add residual connection

        # Block 3
        residual = self.block3_residual(x)
        x = F.relu(self.block3_bn1(self.block3_conv1(x)))
        x = F.relu(self.block3_bn2(self.block3_conv2(x)))
        x = F.relu(self.block3_bn3(self.block3_conv3(x)))
        if x.shape[-1] != residual.shape[-1]:
            residual = F.pad(residual, (0, x.shape[-1] - residual.shape[-1]))
        x = x + residual  # Add residual connection

        # Global Average Pooling
        x = self.gap(x)
        x = x.squeeze(-1)  # Remove last dimension

        # Fully Connected Layer
        x = self.fc(x)
        return F.softmax(x, dim=1)