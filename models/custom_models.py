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


"""
custom_models.py

This module provides a suite of flexible neural network models for time series and sequence modeling tasks. Each model
class allows for customizable configurations, including the number of layers, hidden dimensions, and filter sizes.
The models integrate convolutional layers, LSTM, GRU, and bidirectional variants to support various architectures.

Functions:
    get_model(model_name, input_shape, n_steps, hidden_size, n_layers, num_filters, kernel_size):
        Retrieves a model instance based on the specified name and hyperparameters.

Classes:
    FlexibleCNN:           A multi-layer CNN model with adjustable filters and kernel size.
    FlexibleLSTM:          A standard LSTM model with configurable hidden size and layers.
    FlexibleGRU:           A standard GRU model with configurable hidden size and layers.
    FlexibleCNN_LSTM:      CNN layers followed by an LSTM.
    FlexibleCNN_GRU:       CNN layers followed by a GRU.
    FlexibleCNN_BiLSTM:    CNN layers followed by a bidirectional LSTM.
    FlexibleCNN_BiGRU:     CNN layers followed by a bidirectional GRU.
    FlexibleGRU_CNN_GRU:   GRU layers, followed by CNN layers, ending with a GRU.
    FlexibleLSTM_CNN_LSTM: LSTM layers, followed by CNN layers, ending with an LSTM.
    FlexibleBiGRU_CNN_BiGRU: Bidirectional GRU, followed by CNN layers, ending with another bidirectional GRU.
    FlexibleBiLSTM_CNN_BiLSTM: Bidirectional LSTM, followed by CNN layers, ending with another bidirectional LSTM.

Each class supports dynamic configurations, allowing models to be tailored to specific time series forecasting
or sequence modeling tasks. Models are designed for integration with PyTorch-based training pipelines.
"""


import torch
import torch.nn as nn


def get_model(model_name, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
    """Retrieves the specified model by name with the provided hyperparameters."""
    # Dictionary to map model names to model classes
    models = {
        "lstm": FlexibleLSTM,
        "gru": FlexibleGRU,
        "cnn": FlexibleCNN,
        "cnn_lstm": FlexibleCNN_LSTM,
        "cnn_gru": FlexibleCNN_GRU,
        "cnn_bilstm": FlexibleCNN_BiLSTM,
        "cnn_bigru": FlexibleCNN_BiGRU,
        "gru_cnn_gru": FlexibleGRU_CNN_GRU,
        "lstm_cnn_lstm": FlexibleLSTM_CNN_LSTM,
        "bigru_cnn_bigru": FlexibleBiGRU_CNN_BiGRU,
        "bilstm_cnn_bilstm": FlexibleBiLSTM_CNN_BiLSTM,
    }

    # Separate models into groups based on their parameter requirements
    cnn_only_models = ["cnn"]
    rnn_only_models = ["lstm", "gru"]
    cnn_rnn_hybrid_models = [
        "cnn_lstm", "cnn_gru", "cnn_bilstm", "cnn_bigru", "gru_cnn_gru",
        "lstm_cnn_lstm", "bigru_cnn_bigru", "bilstm_cnn_bilstm"
    ]

    # Initialize model based on group
    if model_name in cnn_only_models:
        # CNN-only models only need num_filters, kernel_size, and n_layers
        return models[model_name](input_shape, n_steps=n_steps, num_filters=num_filters, kernel_size=kernel_size,
                                  n_layers=n_layers)

    elif model_name in rnn_only_models:
        # RNN-only models only need hidden_size and n_layers
        return models[model_name](input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers)

    elif model_name in cnn_rnn_hybrid_models:
        # Hybrid models require all parameters
        return models[model_name](input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                  num_filters=num_filters, kernel_size=kernel_size)

    # Raise an error if the model name is not found
    raise ValueError(f"Model {model_name} not found")

class FlexibleCNN(nn.Module):
    """Simple CNN with flexible number of layers and filters."""
    def __init__(self, input_shape, n_steps=10, num_filters=64, kernel_size=3, n_layers=2):
        super().__init__()
        self.num_features = input_shape[1]
        layers = [nn.Conv1d(self.num_features, num_filters, kernel_size=kernel_size, padding=1), nn.ReLU()]
        in_channels = num_filters
        for _ in range(1, n_layers):
            layers += [nn.Conv1d(in_channels, in_channels * 2, kernel_size=kernel_size, padding=1), nn.ReLU()]
            in_channels *= 2
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, n_steps)

    def forward(self, x):
        x = self.conv_layers(x.transpose(1, 2)).mean(dim=2)  # (batch, num_features, seq_len) -> pooled
        return self.fc(x)

class FlexibleLSTM(nn.Module):
    """Standard LSTM model."""
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_shape[1], hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Last time step output

class FlexibleGRU(nn.Module):
    """Standard GRU model."""
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_shape[1], hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class FlexibleCNN_LSTM(nn.Module):
    """CNN followed by LSTM."""
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super().__init__()
        self.cnn_layers = self._build_cnn_layers(input_shape[1], num_filters, kernel_size, n_layers)
        self.lstm = nn.LSTM(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1), nn.ReLU()]
            in_channels = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class FlexibleCNN_BiLSTM(nn.Module):
    """CNN followed by Bidirectional LSTM."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super().__init__()
        self.cnn_layers = self._build_cnn_layers(input_shape[1], num_filters, kernel_size, n_layers)
        self.lstm = nn.LSTM(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1), nn.ReLU()]
            in_channels = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(torch.cat((hn[-2], hn[-1]), dim=1))  # Concatenate hidden states from both directions

class FlexibleCNN_GRU(nn.Module):
    """CNN followed by GRU."""
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super().__init__()
        self.cnn_layers = self._build_cnn_layers(input_shape[1], num_filters, kernel_size, n_layers)
        self.gru = nn.GRU(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1), nn.ReLU()]
            in_channels = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, hn = self.gru(x)
        return self.fc(hn[-1])


class FlexibleGRU_CNN_GRU(nn.Module):
    """GRU, followed by CNN, and another GRU."""
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super().__init__()
        self.initial_gru = nn.GRU(input_shape[1], hidden_size, n_layers, batch_first=True)
        self.cnn_layers = self._build_cnn_layers(hidden_size, num_filters, kernel_size, n_layers)
        self.final_gru = nn.GRU(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1), nn.ReLU()]
            in_channels = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out, _ = self.initial_gru(x)
        x = self.cnn_layers(out.transpose(1, 2)).transpose(1, 2)
        _, hn = self.final_gru(x)
        return self.fc(hn[-1])


class FlexibleLSTM_CNN_LSTM(nn.Module):
    """LSTM, followed by CNN, and another LSTM."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super().__init__()
        self.initial_lstm = nn.LSTM(input_shape[1], hidden_size, n_layers, batch_first=True)
        self.cnn_layers = self._build_cnn_layers(hidden_size, num_filters, kernel_size, n_layers)
        self.final_lstm = nn.LSTM(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1), nn.ReLU()]
            in_channels = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x, _ = self.initial_lstm(x)
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, (hn, _) = self.final_lstm(x)
        return self.fc(hn[-1])

class FlexibleCNN_BiGRU(nn.Module):
    """CNN followed by Bidirectional GRU."""
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super().__init__()
        self.cnn_layers = self._build_cnn_layers(input_shape[1], num_filters, kernel_size, n_layers)
        self.gru = nn.GRU(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1), nn.ReLU()]
            in_channels = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, hn = self.gru(x)
        return self.fc(torch.cat((hn[-2], hn[-1]), dim=1))


class FlexibleBiGRU_CNN_BiGRU(nn.Module):
    """Bidirectional GRU, followed by CNN and another Bidirectional GRU."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super().__init__()
        self.initial_gru = nn.GRU(input_shape[1], hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.cnn_layers = self._build_cnn_layers(hidden_size * 2, num_filters, kernel_size, n_layers)
        self.final_gru = nn.GRU(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True,
                                bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1), nn.ReLU()]
            in_channels = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x, _ = self.initial_gru(x)
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, hn = self.final_gru(x)
        return self.fc(torch.cat((hn[-2], hn[-1]), dim=1))


class FlexibleBiLSTM_CNN_BiLSTM(nn.Module):
    """Bidirectional LSTM, followed by CNN and another bidirectional LSTM."""
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super().__init__()
        self.initial_lstm = nn.LSTM(input_shape[1], hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.cnn_layers = self._build_cnn_layers(hidden_size * 2, num_filters, kernel_size, n_layers)
        self.final_lstm = nn.LSTM(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True,
                                  bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1), nn.ReLU()]
            in_channels = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x, _ = self.initial_lstm(x)
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, (hn, _) = self.final_lstm(x)
        return self.fc(torch.cat((hn[-2], hn[-1]), dim=1))
