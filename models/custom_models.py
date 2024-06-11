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
    print(type(model_name), type("mlp4"))
    if model_name == "lstm1":
        model = SimpleLSTM1(input_shape, nb_class)
    if model_name == "gru1":
        model = SimpleGRU1(input_shape, nb_class)
    elif model_name == "lstm2":
        model = SimpleLSTM2(input_shape, nb_class)
    elif model_name == "gru2":
        model = SimpleGRU2(input_shape, nb_class)
    elif model_name == "mpl4":
        model = MLP4(input_shape, nb_class)
    elif model_name == "lenet":
        model = CNNLeNet(input_shape, nb_class)
    elif model_name == "vgg":
        model = CNNVGG(input_shape, nb_class)
    elif model_name == "resnet":
        model = CNNResNet(input_shape, nb_class)
    elif model_name == "lfcn":
        model = LSTMFCN(input_shape, nb_class)
    elif model_name == "gfcn":
        model = GRUFCN(input_shape, nb_class)
    else:
        raise ValueError(f"Model {model_name} not found")
    return model


class SimpleLSTM1(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(SimpleLSTM1, self).__init__()
        self.nb_dims = input_shape[1]
        self.lstm = nn.LSTM(self.nb_dims, 100, batch_first=True)
        self.fc = nn.Linear(100, nb_class)

    def forward(self, x):
        # x shape: (batch, nb_timesteps, nb_dims)
        out, (hn, cn) = self.lstm(x)  # LSTM output
        out = out[:, -1, :]  # Take the output of the last timestep
        out = self.fc(out)  # Pass through a linear layer
        return torch.softmax(out, dim=1)  # Softmax to get probabilities


class SimpleGRU1(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(SimpleGRU1, self).__init__()
        self.nb_dims = input_shape[1]
        self.gru = nn.GRU(self.nb_dims, 100, batch_first=True)
        self.fc = nn.Linear(100, nb_class)

    def forward(self, x):
        # x shape: (batch, nb_timesteps, nb_dims)
        out, hn = self.gru(x)  # GRU output
        out = out[:, -1, :]  # Take the output of the last timestep
        out = self.fc(out)  # Pass through a linear layer
        return torch.softmax(out, dim=1)  # Softmax to get probabilities


class SimpleLSTM2(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(SimpleLSTM2, self).__init__()
        self.nb_dims = input_shape[1]
        self.lstm1 = nn.LSTM(self.nb_dims, 100, batch_first=True)
        self.lstm2 = nn.LSTM(100, 100, batch_first=True)
        self.fc = nn.Linear(100, nb_class)

    def forward(self, x):
        # Xshape: (batch, nb_timesteps, nb_dims)
        out, (hn, cn) = self.lstm1(x)  # First LSTM layer
        out, (hn, cn) = self.lstm2(out)  # Second LSTM layer
        out = out[:, -1, :]  # take the output of the last timestep
        out = self.fc(out)  # Pass through a linear layer
        return torch.softmax(out, dim=1)  # Softmax to get probabilities


class SimpleGRU2(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(SimpleGRU2, self).__init__()
        self.nb_dims = input_shape[1]
        self.gru1 = nn.GRU(self.nb_dims, 100, batch_first=True)
        self.gru2 = nn.GRU(100, 100, batch_first=True)
        self.fc = nn.Linear(100, nb_class)

    def forward(self, x):
        # x shape: (batch, nb_timesteps, nb_dims)
        out, _ = self.gru1(x)  # First GRU layer
        out, _ = self.gru2(out)  # Second GRU layer
        out = out[:, -1, :]  # Take the output of the last timestep
        out = self.fc(out)  # Pass through a linear layer
        return torch.softmax(out, dim=1)  # Softmax to get probabilities


class MLP4(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(MLP4, self).__init__()
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


class CNNLeNet(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(CNNLeNet, self).__init__()
        nb_cnn = int(round(log(input_shape[0], 2)) - 3)
        print("Pooling layers: %d" % nb_cnn)

        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[1]  # Assuming input_shape is (timesteps, features)
        out_channels = 6

        for i in range(nb_cnn):
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels
            out_channels += 10

        self.flatten = nn.Flatten()
        conv_output_size = in_channels * (input_shape[0] // (2 ** nb_cnn))
        self.fc1 = nn.Linear(conv_output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nb_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class CNNVGG(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(CNNVGG, self).__init__()
        nb_cnn = int(round(log(input_shape[0], 2)) - 3)
        print("Pooling layers: %d" % nb_cnn)

        layers = []
        in_channels = input_shape[1]

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
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if out.size() != residual.size():
            diff = residual.size(2) - out.size(2)
            out = F.pad(out, (0, diff))
        out += residual
        out = self.activation(out)
        return out


class CNNResNet(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(CNNResNet, self).__init__()
        self.input_shape = input_shape
        self.nb_class = nb_class

        self.block1 = ResidualBlock(input_shape[1], 64)
        self.block2 = ResidualBlock(64, 128)
        self.block3 = ResidualBlock(128, 128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, nb_class)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from (batch, timesteps, features) to (batch, features, timesteps)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # Remove last dimension
        x = self.fc(x)
        return F.softmax(x, dim=1)


class LSTMFCN(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(LSTMFCN, self).__init__()
        self.nb_timesteps = input_shape[0]
        self.nb_dims = input_shape[1]

        self.lstm = nn.LSTM(self.nb_dims, 128, batch_first=True)
        self.dropout = nn.Dropout(0.8)

        self.conv1 = nn.Conv1d(self.nb_dims, 128, kernel_size=8, padding=4)  # padding=4 for 'same' padding
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)  # padding=2 for 'same' padding
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)  # padding=1 for 'same' padding
        self.bn3 = nn.BatchNorm1d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128 + 128, nb_class)

    def forward(self, x):
        # LSTM part
        lstm_out, _ = self.lstm(x)  # x shape: (batch, timesteps, features)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last timestep
        lstm_out = self.dropout(lstm_out)

        # Convolutional part
        x_permuted = x.permute(0, 2, 1)  # Permute to shape: (batch, features, timesteps)
        conv_out = F.relu(self.bn1(self.conv1(x_permuted)))
        conv_out = F.relu(self.bn2(self.conv2(conv_out)))
        conv_out = F.relu(self.bn3(self.conv3(conv_out)))
        conv_out = self.global_avg_pool(conv_out)
        conv_out = conv_out.squeeze(-1)  # Remove last dimension

        # Concatenate LSTM and convolutional outputs
        combined_out = torch.cat((lstm_out, conv_out), dim=1)

        # Fully connected layer
        out = self.fc(combined_out)
        return torch.softmax(out, dim=1)


class GRUFCN(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(GRUFCN, self).__init__()
        self.nb_timesteps = input_shape[0]
        self.nb_dims = input_shape[1]

        self.gru = nn.GRU(self.nb_dims, 128, batch_first=True)
        self.dropout = nn.Dropout(0.8)

        self.conv1 = nn.Conv1d(self.nb_dims, 128, kernel_size=8, padding=4)  # padding=4 for 'same' padding
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)  # padding=2 for 'same' padding
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)  # padding=1 for 'same' padding
        self.bn3 = nn.BatchNorm1d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128 + 128, nb_class)

    def forward(self, x):
        # GRU part
        gru_out, _ = self.gru(x)  # x shape: (batch, timesteps, features)
        gru_out = gru_out[:, -1, :]  # Take the output of the last timestep
        gru_out = self.dropout(gru_out)

        # Convolutional part
        x_permuted = x.permute(0, 2, 1)  # Permute to shape: (batch, features, timesteps)
        conv_out = F.relu(self.bn1(self.conv1(x_permuted)))
        conv_out = F.relu(self.bn2(self.conv2(conv_out)))
        conv_out = F.relu(self.bn3(self.conv3(conv_out)))
        conv_out = self.global_avg_pool(conv_out)
        conv_out = conv_out.squeeze(-1)  # Remove last dimension

        # Concatenate GRU and convolutional outputs
        combined_out = torch.cat((gru_out, conv_out), dim=1)

        # Fully connected layer
        out = self.fc(combined_out)
        return torch.softmax(out, dim=1)
