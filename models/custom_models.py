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


def get_model(model_name, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
    if model_name == "lstm":
        model = FlexibleLSTM(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers)
    elif model_name == "gru":
        model = FlexibleGRU(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers)
    elif model_name == "cnn":
        model = FlexibleCNN(input_shape, n_steps=n_steps, num_filters=num_filters, kernel_size=kernel_size,
                            n_layers=n_layers)
    elif model_name == "cnn_lstm":
        model = FlexibleCNN_LSTM(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                 num_filters=num_filters, kernel_size=kernel_size)
    elif model_name == "cnn_gru":
        model = FlexibleCNN_GRU(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                num_filters=num_filters, kernel_size=kernel_size)
    elif model_name == "cnn_bilstm":
        model = FlexibleCNN_BiLSTM(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                   num_filters=num_filters, kernel_size=kernel_size)
    elif model_name == "cnn_bigru":
        model = FlexibleCNN_BiGRU(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                  num_filters=num_filters, kernel_size=kernel_size)
    elif model_name == "gru_cnn_gru":
        model = FlexibleGRU_CNN_GRU(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                    num_filters=num_filters, kernel_size=kernel_size)
    elif model_name == "lstm_cnn_lstm":
        model = FlexibleLSTM_CNN_LSTM(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                      num_filters=num_filters, kernel_size=kernel_size)
    elif model_name == "bigru_cnn_bigru":
        model = FlexibleBiGRU_CNN_BiGRU(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                        num_filters=num_filters, kernel_size=kernel_size)
    elif model_name == "bilstm_cnn_bilstm":
        model = FlexibleBiLSTM_CNN_BiLSTM(input_shape, n_steps=n_steps, hidden_size=hidden_size, n_layers=n_layers,
                                          num_filters=num_filters, kernel_size=kernel_size)
    else:
        raise ValueError(f"Model {model_name} not found")
    return model


class FlexibleCNN(nn.Module):
    def __init__(self, input_shape, n_steps=10, num_filters=64, kernel_size=3, n_layers=2):
        super(FlexibleCNN, self).__init__()
        self.num_features = input_shape[1]
        layers = []

        in_channels = self.num_features
        for _ in range(n_layers):
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            in_channels = num_filters
            num_filters *= 2  # Increase filters with each layer, can be customized

        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, n_steps)  # Final FC layer to output n_steps

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, num_features, sequence_length)
        x = self.conv_layers(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        return x


class FlexibleLSTM(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1):
        super(FlexibleLSTM, self).__init__()
        self.num_features = input_shape[1]
        self.lstm = nn.LSTM(input_size=self.num_features, hidden_size=hidden_size, num_layers=n_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        return out


class FlexibleGRU(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1):
        super(FlexibleGRU, self).__init__()
        self.num_features = input_shape[1]

        # Define a flexible GRU with multiple layers
        self.gru = nn.GRU(input_size=self.num_features, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)

        # Fully connected layer to map GRU output to n_steps prediction
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        # Pass the input through GRU layers
        out, hn = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class FlexibleCNN_LSTM(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super(FlexibleCNN_LSTM, self).__init__()
        self.num_features = input_shape[1]

        # CNN layer stack
        cnn_layers = []
        in_channels = self.num_features
        for _ in range(n_layers):
            cnn_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1))
            cnn_layers.append(nn.ReLU())
            in_channels = num_filters
            num_filters *= 2  # Increase filters per layer
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        x = x.transpose(1, 2)  # to (batch_size, num_features, sequence_length)
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, channels) for LSTM
        _, (hn, _) = self.lstm(x)
        x = hn[-1]  # Last hidden state
        x = self.fc(x)
        return x


class FlexibleCNN_GRU(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super(FlexibleCNN_GRU, self).__init__()
        self.num_features = input_shape[1]

        # CNN layer stack
        cnn_layers = []
        in_channels = self.num_features
        for _ in range(n_layers):
            cnn_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1))
            cnn_layers.append(nn.ReLU())
            in_channels = num_filters
            num_filters *= 2  # Increase filters per layer
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # GRU layer
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, num_features, sequence_length)
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, channels) for GRU
        _, hn = self.gru(x)
        x = hn[-1]  # Last hidden state
        x = self.fc(x)
        return x


class FlexibleCNN_BiLSTM(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super(FlexibleCNN_BiLSTM, self).__init__()
        self.num_features = input_shape[1]

        # CNN layers
        cnn_layers = []
        in_channels = self.num_features
        for _ in range(n_layers):
            cnn_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1))
            cnn_layers.append(nn.ReLU())
            in_channels = num_filters
            num_filters *= 2
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                            bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, n_steps)  # BiLSTM doubles hidden_size

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        x = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate last hidden states from both directions
        x = self.fc(x)
        return x


class FlexibleCNN_BiGRU(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super(FlexibleCNN_BiGRU, self).__init__()
        self.num_features = input_shape[1]

        # CNN layers
        cnn_layers = []
        in_channels = self.num_features
        for _ in range(n_layers):
            cnn_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1))
            cnn_layers.append(nn.ReLU())
            in_channels = num_filters
            num_filters *= 2
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # BiGRU layer
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                          bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, n_steps)  # BiGRU doubles hidden_size

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)
        _, hn = self.gru(x)
        x = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate last hidden states from both directions
        x = self.fc(x)
        return x


class FlexibleGRU_CNN_GRU(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super(FlexibleGRU_CNN_GRU, self).__init__()
        self.num_features = input_shape[1]

        # Initial GRU layers
        self.initial_gru = nn.GRU(input_size=self.num_features, hidden_size=hidden_size, num_layers=n_layers,
                                  batch_first=True)

        # CNN layers
        cnn_layers = []
        in_channels = hidden_size
        for _ in range(n_layers):
            cnn_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=1))
            cnn_layers.append(nn.ReLU())
            in_channels = num_filters
            num_filters *= 2
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # Final GRU layer
        self.final_gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        # Initial GRU
        out, _ = self.initial_gru(x)

        # CNN layers
        x = out.transpose(1, 2)  # Prepare for Conv1d
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)  # Prepare for final GRU

        # Final GRU
        _, hn = self.final_gru(x)
        x = hn[-1]
        x = self.fc(x)
        return x


class FlexibleLSTM_CNN_LSTM(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super(FlexibleLSTM_CNN_LSTM, self).__init__()
        self.num_features = input_shape[1]

        # Initial LSTM layers
        self.initial_lstm = nn.LSTM(input_size=self.num_features, hidden_size=hidden_size,
                                    num_layers=n_layers, batch_first=True)

        # CNN layers
        cnn_layers = []
        in_channels = hidden_size
        current_num_filters = num_filters
        for _ in range(n_layers):
            cnn_layers.append(nn.Conv1d(in_channels, current_num_filters, kernel_size=kernel_size, padding=1))
            cnn_layers.append(nn.ReLU())
            in_channels = current_num_filters
            current_num_filters *= 2
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # Final LSTM layer
        self.final_lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers,
                                  batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        # Initial LSTM
        out, _ = self.initial_lstm(x)

        # CNN layers
        x = out.transpose(1, 2)  # Prepare for Conv1d
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)  # Prepare for final LSTM

        # Final LSTM
        _, (hn, _) = self.final_lstm(x)
        x = hn[-1]
        x = self.fc(x)
        return x


class FlexibleBiGRU_CNN_BiGRU(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super(FlexibleBiGRU_CNN_BiGRU, self).__init__()
        self.num_features = input_shape[1]

        # Initial bidirectional GRU layers
        self.initial_gru = nn.GRU(input_size=self.num_features, hidden_size=hidden_size, num_layers=n_layers,
                                  batch_first=True, bidirectional=True)

        # CNN layers
        cnn_layers = []
        in_channels = hidden_size * 2  # Bidirectional doubles the hidden size
        current_num_filters = num_filters
        for _ in range(n_layers):
            cnn_layers.append(nn.Conv1d(in_channels, current_num_filters, kernel_size=kernel_size, padding=1))
            cnn_layers.append(nn.ReLU())
            in_channels = current_num_filters
            current_num_filters *= 2
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # Final bidirectional GRU layer
        self.final_gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers,
                                batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, n_steps)  # Bidirectional doubles hidden size

    def forward(self, x):
        # Initial BiGRU
        out, _ = self.initial_gru(x)

        # CNN layers
        x = out.transpose(1, 2)  # Prepare for Conv1d
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)  # Prepare for final BiGRU

        # Final BiGRU
        _, hn = self.final_gru(x)
        x = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate last hidden states from both directions
        x = self.fc(x)
        return x


class FlexibleBiLSTM_CNN_BiLSTM(nn.Module):
    def __init__(self, input_shape, n_steps=10, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        super(FlexibleBiLSTM_CNN_BiLSTM, self).__init__()
        self.num_features = input_shape[1]

        # Initial bidirectional LSTM layers
        self.initial_lstm = nn.LSTM(input_size=self.num_features, hidden_size=hidden_size, num_layers=n_layers,
                                    batch_first=True, bidirectional=True)

        # CNN layers
        cnn_layers = []
        in_channels = hidden_size * 2  # Bidirectional doubles the hidden size
        current_num_filters = num_filters
        for _ in range(n_layers):
            cnn_layers.append(nn.Conv1d(in_channels, current_num_filters, kernel_size=kernel_size, padding=1))
            cnn_layers.append(nn.ReLU())
            in_channels = current_num_filters
            current_num_filters *= 2
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # Final bidirectional LSTM layer
        self.final_lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers,
                                  batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, n_steps)  # Bidirectional doubles hidden size

    def forward(self, x):
        # Initial BiLSTM
        out, _ = self.initial_lstm(x)

        # CNN layers
        x = out.transpose(1, 2)  # Prepare for Conv1d
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)  # Prepare for final BiLSTM

        # Final BiLSTM
        _, (hn, _) = self.final_lstm(x)
        x = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate last hidden states from both directions
        x = self.fc(x)
        return x
