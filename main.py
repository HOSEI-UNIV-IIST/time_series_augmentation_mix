#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/08/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import utils.datasets as ds
from models import custom_models as mod, mix_augmentation_refined as aug
from utils.argument_parser import argument_parser
from utils.cache_loss_accuracy import CacheLossAccuracy
from utils.input_data import get_datasets
from utils.save_result import save_accuracy


def prepare_multi_step_data(x, y, look_back, n_steps):
    x_seq, y_seq = [], []
    for i in range(len(x) - look_back - n_steps + 1):
        # Use `look_back` timesteps for input sequences
        x_seq.append(x[i:i + look_back])  # Shape: (look_back, num_features)
        # Use the next `n_steps` timesteps as the target
        y_seq.append(y[i + look_back:i + look_back + n_steps])  # Shape: (n_steps, num_features)

    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)

    # Ensure x_seq has 3 dimensions: (batch_size, look_back, num_features)
    if x_seq.ndim == 2:
        x_seq = np.expand_dims(x_seq, axis=-1)

    return x_seq, y_seq


class Trainer:
    def __init__(self, args, look_back=5, n_steps=10):
        self.args = args
        self.look_back = look_back  # Number of past timesteps to use as input
        self.n_steps = n_steps  # Number of steps to forecast
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load and process data
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_and_prepare_data()

        # Automatically infer input shape from training data
        self.input_shape = (self.x_train.shape[1], self.x_train.shape[2])  # (sequence_length, num_features)

        # Initialize model
        self.model = self.initialize_model()
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler()

        # Set loss function
        self.criterion = nn.MSELoss()

        # Create DataLoaders
        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(self.x_test, self.y_test), batch_size=args.batch_size, shuffle=False)

    def load_and_prepare_data(self):
        x_train_, y_train_, x_test_, y_test_ = get_datasets(self.args)

        # Prepare data for multi-step forecasting
        x_train, y_train = prepare_multi_step_data(x_train_, y_train_, self.look_back, self.n_steps)
        x_test, y_test = prepare_multi_step_data(x_test_, y_test_, self.look_back, self.n_steps)

        # Convert to tensors and check shapes
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        print(f"x_train shape after preparation: {x_train.shape}")  # Expected: (batch_size, look_back, num_features)
        print(f"y_train shape after preparation: {y_train.shape}")  # Expected: (batch_size, n_steps)

        return x_train, y_train, x_test, y_test

    def initialize_model(self):
        # Initialize the model using automatically inferred input_shape and n_steps
        model = mod.get_model(self.args.model, self.input_shape, n_steps=self.n_steps).to(self.device)
        return self.wrap_model_with_dataparallel(model)

    def wrap_model_with_dataparallel(self, model):
        if torch.cuda.is_available() and self.args.gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.args.gpus)))
        return model.to(self.device)

    def setup_optimizer_and_scheduler(self):
        if self.args.optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == "nadam":
            optimizer = optim.NAdam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == "adadelta":
            optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr, rho=0.95, eps=1e-8)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                      patience=int(np.ceil(self.args.iterations / (30 * self.args.batch_size))),
                                      min_lr=1e-5,
                                      cooldown=int(np.ceil(self.args.iterations / (40 * self.args.batch_size))))
        return optimizer, scheduler

    def train_and_validate(self, nb_epochs):
        for epoch in range(nb_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()

            print(f'Epoch {epoch + 1}/{nb_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            self.scheduler.step(val_loss)

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        for data, labels in self.train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def validate_one_epoch(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        return val_loss / len(self.test_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                total_loss += self.criterion(outputs, labels).item()
        return total_loss / len(self.test_loader)

    def plot_validation_predictions(self, num_samples=100):
        self.model.eval()
        true_labels = []
        predicted_labels = []

        # Collect predictions and true labels for the test set
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                true_labels.extend(labels.cpu().numpy().flatten())
                predicted_labels.extend(outputs.cpu().numpy().flatten())

        plt.figure(figsize=(14, 8))
        plt.plot(true_labels[-num_samples:], color="blue", label="True Labels", linewidth=1)
        plt.plot(predicted_labels[-num_samples:], color="red", label="Predicted Labels", linestyle="--", linewidth=1)
        plt.xlabel("Sample Index")
        plt.ylabel("Predicted Value")
        plt.title(f"True vs Predicted Values for a Subset of Test Set (Last {num_samples} Samples)")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    args = argument_parser()
    look_back = 7  # Number of past timesteps to consider as input
    n_steps = 3  # Number of steps ahead to predict
    trainer = Trainer(args, look_back=look_back, n_steps=n_steps)

    nb_iterations = args.iterations
    nb_epochs = int(np.ceil(nb_iterations * (args.batch_size / trainer.x_train.shape[0])))

    if args.train:
        trainer.train_and_validate(nb_epochs)

    print("Evaluation")
    loss = trainer.evaluate()
    print(f'Best Validation Loss: {loss:.4f}')

    trainer.plot_validation_predictions(num_samples=100)