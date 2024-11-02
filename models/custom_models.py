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
import torch.nn.functional as F

def get_model(model_name, input_shape, nb_class=3, n_steps=10):
    if model_name == "lstm1":
        model = SimpleLSTM1(input_shape, n_steps=n_steps)
    elif model_name == "gru1":
        model = SimpleGRU1(input_shape, nb_class)
    elif model_name == "lstm2":
        model = SimpleLSTM2(input_shape, nb_class)
    elif model_name == "gru2":
        model = SimpleGRU2(input_shape, nb_class)
    elif model_name == "fcnn":
        model = FCNN(input_shape, nb_class)
    elif model_name == "lfcn":
        model = LSTMFCN(input_shape, nb_class)
    elif model_name == "esat":
        model = ESAT(input_shape, nb_class)
    else:
        raise ValueError(f"Model {model_name} not found")
    return model


class SimpleLSTM1(nn.Module):
    def __init__(self, input_shape, n_steps=10):
        """
        Initializes an LSTM model for multi-step forecasting.

        Parameters:
        - input_shape (tuple): Shape of the input data (look_back, num_features).
        - n_steps (int): Number of steps to predict ahead (forecast horizon).
        """
        super(SimpleLSTM1, self).__init__()

        self.num_features = input_shape[1]  # Number of features in input data
        self.n_steps = n_steps

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=self.num_features, hidden_size=100, batch_first=True)

        # Fully connected layer to map to desired output shape (n_steps)
        self.fc = nn.Linear(100, n_steps)

    def forward(self, x):
        # LSTM layer
        out, (hn, cn) = self.lstm(x)

        # Use the output of the last timestep
        out = out[:, -1, :]

        # Fully connected layer for forecasting multiple steps ahead
        out = self.fc(out)

        return out

# class SimpleLSTM1(nn.Module):
#     def __init__(self, input_shape, nb_class):
#         super(SimpleLSTM1, self).__init__()
#         self.nb_dims = input_shape[1]
#         self.lstm = nn.LSTM(self.nb_dims, 100, batch_first=True)
#         self.fc = nn.Linear(100, nb_class)
#
#     def forward(self, x):
#         # x shape: (batch, nb_timesteps, nb_dims)
#         out, (hn, cn) = self.lstm(x)  # LSTM output
#         out = out[:, -1, :]  # Take the output of the last timestep
#         out = self.fc(out)  # Pass through a linear layer
#         return torch.softmax(out, dim=1)  # Softmax to get probabilities


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
        return out


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


class LSTMFCN(nn.Module):
    def __init__(self, input_shape, nb_class):
        super(LSTMFCN, self).__init__()
        self.in_channels = input_shape[1]  # input_shape is (timesteps, features)

        # LSTM part
        self.lstm = nn.LSTM(input_shape[1], 128, batch_first=True)
        self.dropout = nn.Dropout(0.8)

        # Fully Convolutional Network part
        self.conv1 = nn.Conv1d(input_shape[1], 128, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(256, nb_class)  # 128 from LSTM + 128 from GAP

    def forward(self, x):
        # x shape: (batch_size, timesteps, features)

        # LSTM part
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, timesteps, 128)
        lstm_out = lstm_out[:, -1, :]  # Taking the output of the last time step
        lstm_out = self.dropout(lstm_out)

        # FCN part
        x = x.permute(0, 2, 1)  # Change to (batch_size, features, timesteps)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.squeeze(-1)  # Remove last dimension

        # Concatenate LSTM output and FCN output
        x = torch.cat((lstm_out, x), dim=1)

        # Fully Connected Layer
        x = self.fc(x)
        return F.softmax(x, dim=1)


class ESAT(nn.Module):
    def __init__(self, input_shape, nb_class, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super(ESAT, self).__init__()
        self.input_shape = input_shape
        self.nb_class = nb_class

        # Learnable Positional Encoding
        self.pos_encoder = nn.Embedding(input_shape[0], d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                        dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                        dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, nb_class)

    def forward(self, x):
        # x shape: (batch_size, timesteps, features)
        positions = torch.arange(0, x.size(1)).unsqueeze(0).expand(x.size(0), -1).to(x.device)
        x = x + self.pos_encoder(positions)  # Apply learned positional encoding

        memory = self.transformer_encoder(x)
        memory = self.layer_norm(memory)
        out = self.transformer_decoder(memory, memory)

        out = out.mean(dim=1)  # Global average pooling
        out = self.fc(out)
        return F.softmax(out, dim=1)
