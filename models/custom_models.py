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

# custom_models.py

import torch
import torch.nn as nn


def get_model(model_name, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3,
              pool_size=2, dropout=0.3):
    """Retrieves the specified model by name with the provided hyperparameters."""
    models = {
        "lstm": FlexibleLSTM,
        "gru": FlexibleGRU,
        "cnn": FlexibleCNN,
        "cnn_lstm": FlexibleCNN_LSTM,
        "cnn_gru": FlexibleCNN_GRU,
        "bigru_cnn_bigru": FlexibleBiGRU_CNN_BiGRU,
        "bilstm_cnn_bilstm": FlexibleBiLSTM_CNN_BiLSTM,
        "cnn_attention_bigru": FlexibleCNN_Attention_BiGRU,
        "cnn_attention_bilstm": FlexibleCNN_Attention_BiLSTM
    }

    cnn_only_models = ["cnn"]
    rnn_only_models = ["lstm", "gru"]
    cnn_rnn_hybrid_models = [
        "cnn_lstm", "cnn_gru", "bigru_cnn_bigru", "bilstm_cnn_bilstm",
        "cnn_attention_bigru", "cnn_attention_bilstm"
    ]

    if model_name in cnn_only_models:
        return models[model_name](input_shape, n_steps=n_steps, num_filters=num_filters, kernel_size=kernel_size,
                                  n_layers=n_layers, pool_size=pool_size, dropout=dropout)
    elif model_name in rnn_only_models:
        return models[model_name](input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                  dropout=dropout)
    elif model_name in cnn_rnn_hybrid_models:
        return models[model_name](input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                  num_filters=num_filters, kernel_size=kernel_size, pool_size=pool_size,
                                  dropout=dropout)
    raise ValueError(f"Model {model_name} not found")


class FlexibleCNN(nn.Module):
    """Simple CNN with BatchNorm, ReLU, MaxPooling, and Dropout."""

    def __init__(self, input_shape, n_steps=10, num_filters=64, kernel_size=3, n_layers=2, pool_size=2, dropout=0.3):
        super().__init__()
        self.input_shape = input_shape  # Store input shape for use in helper functions
        self.num_features = input_shape[1]
        layers = []
        in_channels = self.num_features
        output_dim = self.input_shape[0]  # look_back dimension

        for _ in range(n_layers):
            kernel_size = min(kernel_size, output_dim)
            effective_pool_size = min(pool_size, output_dim)
            layers += [
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(effective_pool_size),
                nn.Dropout(dropout)
            ]
            in_channels = num_filters
            num_filters *= 2
            output_dim = (output_dim + 1) // effective_pool_size  # Update output dimension for next layer

        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, n_steps)

    def forward(self, x):
        x = self.conv_layers(x.transpose(1, 2)).mean(dim=2)
        return self.fc(x)


class FlexibleLSTM(nn.Module):
    """Standard LSTM model with Dropout."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, dropout=0.3):
        super().__init__()
        self.input_shape = input_shape
        self.lstm = nn.LSTM(self.input_shape[1], hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class FlexibleGRU(nn.Module):
    """Standard GRU model with Dropout."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, dropout=0.3):
        super().__init__()
        self.input_shape = input_shape
        self.gru = nn.GRU(self.input_shape[1], hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class FlexibleCNN_LSTM(nn.Module):
    """CNN followed by LSTM with BatchNorm, ReLU, MaxPooling, and Dropout."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3, pool_size=2,
                 dropout=0.3):
        super().__init__()
        self.input_shape = input_shape
        self.cnn_layers = self._build_cnn_layers(self.input_shape[1], num_filters, kernel_size, n_layers, pool_size,
                                                 dropout)
        self.lstm = nn.LSTM(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers, pool_size, dropout):
        layers = []
        output_dim = self.input_shape[0]  # look_back dimension

        for _ in range(n_layers):
            kernel_size = min(kernel_size, output_dim)
            effective_pool_size = min(pool_size, output_dim)
            layers += [
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(effective_pool_size),
                nn.Dropout(dropout)
            ]
            in_channels = num_filters
            num_filters *= 2
            output_dim = (output_dim + 1) // effective_pool_size

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class FlexibleCNN_GRU(nn.Module):
    """CNN followed by GRU with BatchNorm, ReLU, MaxPooling, and Dropout."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3, pool_size=2,
                 dropout=0.3):
        super().__init__()
        self.input_shape = input_shape
        self.cnn_layers = self._build_cnn_layers(self.input_shape[1], num_filters, kernel_size, n_layers, pool_size,
                                                 dropout)
        self.gru = nn.GRU(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers, pool_size, dropout):
        layers = []
        output_dim = self.input_shape[0]

        for _ in range(n_layers):
            kernel_size = min(kernel_size, output_dim)
            effective_pool_size = min(pool_size, output_dim)
            layers += [
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(effective_pool_size),
                nn.Dropout(dropout)
            ]
            in_channels = num_filters
            num_filters *= 2
            output_dim = (output_dim + 1) // effective_pool_size

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, hn = self.gru(x)
        return self.fc(hn[-1])


class FlexibleBiGRU_CNN_BiGRU(nn.Module):
    """Bidirectional GRU, followed by CNN, and another Bidirectional GRU."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3, pool_size=2,
                 dropout=0.3):
        super().__init__()
        self.input_shape = input_shape
        self.initial_gru = nn.GRU(self.input_shape[1], hidden_size, n_layers, batch_first=True, bidirectional=True,
                                  dropout=dropout)
        self.cnn_layers = self._build_cnn_layers(hidden_size * 2, num_filters, kernel_size, n_layers, pool_size,
                                                 dropout)
        self.final_gru = nn.GRU(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True,
                                bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers, pool_size, dropout):
        layers = []
        output_dim = self.input_shape[0]

        for _ in range(n_layers):
            kernel_size = min(kernel_size, output_dim)
            effective_pool_size = min(pool_size, output_dim)
            layers += [
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(effective_pool_size),
                nn.Dropout(dropout)
            ]
            in_channels = num_filters
            num_filters *= 2
            output_dim = (output_dim + 1) // effective_pool_size

        return nn.Sequential(*layers)

    def forward(self, x):
        x, _ = self.initial_gru(x)
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, hn = self.final_gru(x)
        return self.fc(torch.cat((hn[-2], hn[-1]), dim=1))


class FlexibleBiLSTM_CNN_BiLSTM(nn.Module):
    """Bidirectional LSTM, followed by CNN and another Bidirectional LSTM."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3, pool_size=2,
                 dropout=0.3):
        super().__init__()
        self.input_shape = input_shape
        self.initial_lstm = nn.LSTM(self.input_shape[1], hidden_size, n_layers, batch_first=True, bidirectional=True,
                                    dropout=dropout)
        self.cnn_layers = self._build_cnn_layers(hidden_size * 2, num_filters, kernel_size, n_layers, pool_size,
                                                 dropout)
        self.final_lstm = nn.LSTM(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True,
                                  bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers, pool_size, dropout):
        layers = []
        output_dim = self.input_shape[0]

        for _ in range(n_layers):
            kernel_size = min(kernel_size, output_dim)
            effective_pool_size = min(pool_size, output_dim)
            layers += [
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(effective_pool_size),
                nn.Dropout(dropout)
            ]
            in_channels = num_filters
            num_filters *= 2
            output_dim = (output_dim + 1) // effective_pool_size

        return nn.Sequential(*layers)

    def forward(self, x):
        x, _ = self.initial_lstm(x)
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        _, (hn, _) = self.final_lstm(x)
        return self.fc(torch.cat((hn[-2], hn[-1]), dim=1))


class FlexibleCNN_Attention_BiGRU(nn.Module):
    """CNN followed by an Attention Mechanism and Bidirectional GRU."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3, pool_size=2,
                 dropout=0.3):
        super().__init__()
        self.input_shape = input_shape
        self.cnn_layers = self._build_cnn_layers(self.input_shape[1], num_filters, kernel_size, n_layers, pool_size,
                                                 dropout)
        self.attention = nn.MultiheadAttention(embed_dim=num_filters * (2 ** (n_layers - 1)), num_heads=1,
                                               batch_first=True)
        self.bigru = nn.GRU(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers, pool_size, dropout):
        layers = []
        output_dim = self.input_shape[0]

        for _ in range(n_layers):
            kernel_size = min(kernel_size, output_dim)
            effective_pool_size = min(pool_size, output_dim)
            layers += [
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(effective_pool_size),
                nn.Dropout(dropout)
            ]
            in_channels = num_filters
            num_filters *= 2
            output_dim = (output_dim + 1) // effective_pool_size

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        attn_output, _ = self.attention(x, x, x)
        _, hn = self.bigru(attn_output)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(out)


class FlexibleCNN_Attention_BiLSTM(nn.Module):
    """CNN followed by an Attention Mechanism and Bidirectional LSTM."""

    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3, pool_size=2,
                 dropout=0.3):
        super().__init__()
        self.input_shape = input_shape
        self.cnn_layers = self._build_cnn_layers(self.input_shape[1], num_filters, kernel_size, n_layers, pool_size,
                                                 dropout)
        self.attention = nn.MultiheadAttention(embed_dim=num_filters * (2 ** (n_layers - 1)), num_heads=1,
                                               batch_first=True)
        self.bilstm = nn.LSTM(num_filters * (2 ** (n_layers - 1)), hidden_size, n_layers, batch_first=True,
                              bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, n_steps)

    def _build_cnn_layers(self, in_channels, num_filters, kernel_size, n_layers, pool_size, dropout):
        layers = []
        output_dim = self.input_shape[0]

        for _ in range(n_layers):
            kernel_size = min(kernel_size, output_dim)
            effective_pool_size = min(pool_size, output_dim)
            layers += [
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(effective_pool_size),
                nn.Dropout(dropout)
            ]
            in_channels = num_filters
            num_filters *= 2
            output_dim = (output_dim + 1) // effective_pool_size

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_layers(x.transpose(1, 2)).transpose(1, 2)
        attn_output, _ = self.attention(x, x, x)
        _, (hn, _) = self.bilstm(attn_output)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(out)
