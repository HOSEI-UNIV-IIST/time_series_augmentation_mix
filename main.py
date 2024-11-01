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
from utils.input_data import get_datasets, prepare_multi_step_data
from utils.save_result import save_accuracy


class Trainer:
    def __init__(self, args, n_steps=10):
        self.args = args
        self.n_steps = n_steps  # Number of steps for multi-step forecasting
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nb_class = ds.nb_classes(args.dataset)
        self.nb_dims = ds.nb_dims(args.dataset)

        # Load and process data
        self.x_train, self.y_train, self.x_test, self.y_test, self.input_shape = self.load_and_prepare_data()
        self.model = self.initialize_model()
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler()
        self.criterion = nn.CrossEntropyLoss()

        # Setup directories for saving models and logs
        if args.save:
            self.log_dir, self.weight_dir, self.output_dir, self.model_prefix = self.setup_directories()
            self.csv_logger = SummaryWriter(log_dir=os.path.join(self.log_dir, self.model_prefix))

        # Create DataLoaders
        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=args.batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(TensorDataset(self.x_test, self.y_test), batch_size=args.batch_size,
                                      shuffle=False)

    def load_and_prepare_data(self):
        x_train_, y_train_, x_test_, y_test_ = get_datasets(self.args)
        print(
            f"x_train_ shape before augmentation: {x_train_.shape}, y_train_ shape before augmentation: {y_train_.shape}")
        nb_timesteps = int(x_train_.shape[1] / self.nb_dims)
        input_shape = (nb_timesteps, self.nb_dims)

        # Reshape and convert data to tensors
        x_train = torch.tensor(x_train_.reshape((-1, *input_shape)), dtype=torch.float32)
        x_test = torch.tensor(x_test_.reshape((-1, *input_shape)), dtype=torch.float32)
        y_train = torch.tensor(ds.class_offset(y_train_, self.args.dataset), dtype=torch.long)
        y_test = torch.tensor(ds.class_offset(y_test_, self.args.dataset), dtype=torch.long)

        # Data augmentation
        if not self.args.original:
            print(f"Augmentation method: {self.args.augmentation_method}")
            start_time = time.time() * 1000
            x_train, y_train, augmentation_tags = aug.run_augmentation_refined(x_train.numpy(), y_train.numpy(),
                                                                               self.args)
            duration = time.time() * 1000 - start_time
            print(f"Augmentation process took {duration:.2f} ms")
            x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        else:
            augmentation_tags = '_original'
            duration = 0

        # Prepare data for multi-step forecasting
        x_train, y_train = prepare_multi_step_data(x_train, y_train, self.n_steps)
        x_test, y_test = prepare_multi_step_data(x_test, y_test, self.n_steps)

        # Convert x_train and x_test back to tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)

        # Convert y_train and y_test to one-hot encoding and then to tensors
        y_train = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes=self.nb_class).float()
        y_test = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes=self.nb_class).float()

        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]

        return x_train, y_train, x_test, y_test, input_shape

    def initialize_model(self):
        model = mod.get_model(self.args.model, self.input_shape, self.nb_class, n_steps=self.n_steps).to(self.device)
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

    def setup_directories(self):
        augmentation_tags = '_original' if self.args.original else f'_{self.args.augmentation_method}'
        model_prefix = f"{self.args.dataset}{augmentation_tags}"
        log_dir = os.path.join(self.args.log_dir, str(self.device), str(self.args.model), str(self.args.dataset),
                               str(self.args.augmentation_ratio))
        weight_dir = os.path.join(self.args.weight_dir, str(self.device), str(self.args.model), str(self.args.dataset),
                                  str(self.args.augmentation_ratio))
        output_dir = os.path.join(self.args.output_dir, str(self.device), str(self.args.model), str(self.args.dataset),
                                  str(self.args.augmentation_ratio))

        for dir_path in [log_dir, weight_dir, output_dir]:
            os.makedirs(dir_path, exist_ok=True)

        return log_dir, weight_dir, output_dir, model_prefix

    def train_and_validate(self, nb_epochs):
        val_losses, val_accuracies = [], []
        best_val_loss, epochs_no_improve = float('inf'), 0
        early_stopping_patience = 200 if self.args.model in ["fcnn", "lstm1", "lstm2", "gru1", "gru2"] else 200

        for epoch in range(nb_epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate_one_epoch()
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(
                f'Epoch {epoch + 1}/{nb_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss, epochs_no_improve = val_loss, 0
                torch.save(self.model.state_dict(),
                           os.path.join(self.weight_dir, f"{self.model_prefix}_best_weights.pth"))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        if self.args.save:
            torch.save(self.model.state_dict(), os.path.join(self.weight_dir, f"{self.model_prefix}_final_weights.pth"))
            cache = CacheLossAccuracy(val_losses, val_accuracies, self.output_dir, self.model_prefix)
            cache.save_training_data()
            cache.plot_training_data()

    def train_one_epoch(self):
        self.model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for data, labels in self.train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        return epoch_loss / len(self.train_loader), epoch_correct / epoch_total

    def validate_one_epoch(self):
        self.model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        return val_loss / len(self.test_loader), val_correct / val_total

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                labels = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = (100 * correct) / total
        return accuracy

    def plot_predictions(self, num_samples=1):
        """
        Plots the historical data, look-back window, and multi-step forecast for a subset of test samples.

        Parameters:
        num_samples (int): Number of samples to display in the plot.
        """
        self.model.eval()

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # Get model output for the data
                outputs = self.model(data)  # Shape: (batch_size, n_steps, nb_class)
                predictions = torch.argmax(outputs, dim=2).cpu().numpy()  # Convert predictions to class indices
                labels = torch.argmax(labels, dim=2).cpu().numpy()  # Convert true labels to class indices

                # Plot only the specified number of samples
                for i in range(num_samples):
                    # Get the historical data (input sequence) and forecasted data
                    historical_data = data[i].cpu().numpy()
                    true_future = labels[i]
                    predicted_future = predictions[i]

                    # Prepare indices for plotting
                    look_back_length = historical_data.shape[0]
                    forecast_length = self.n_steps
                    total_length = look_back_length + forecast_length

                    # Plot historical data
                    plt.figure(figsize=(12, 6))
                    plt.plot(range(look_back_length), historical_data[:, 0], color="blue",
                             label="Historical Data (Target)")

                    # Plot look-back window for prediction
                    plt.plot(range(look_back_length - forecast_length, look_back_length),
                             historical_data[-forecast_length:, 0], color="green",
                             label=f"{forecast_length}-Day Look-Back Window")

                    # Plot the multi-step forecast starting from the end of historical data
                    forecast_indices = range(look_back_length, total_length)
                    plt.plot(forecast_indices, predicted_future, linestyle="--", color="orange",
                             label=f"{forecast_length}-Day Forecast")

                    plt.xlabel("Day")
                    plt.ylabel("Target Value")
                    plt.title(
                        f"{forecast_length}-Day Forecast for Target Variable with {forecast_length}-Day Look-Back Window")
                    plt.legend()
                    plt.show()

                break  # Only plot for the first batch to avoid excessive plotting


if __name__ == '__main__':
    args = argument_parser()
    n_steps = 10  # Number of steps ahead to predict
    trainer = Trainer(args, n_steps=n_steps)

    # Calculate iterations and epochs
    nb_iterations = args.iterations
    nb_epochs = int(np.ceil(nb_iterations * (args.batch_size / trainer.x_train.shape[0])))

    # Training and Validation
    if args.train:
        print("¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨Training")
        trainer.train_and_validate(nb_epochs)

    # Evaluation
    print("¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨Evaluation")
    accuracy = trainer.evaluate()
    print(f'Best Train Accuracy: {accuracy:.2f}%')

    # Save accuracy results
    save_accuracy(accuracy, f'{args.dataset}_accuracy', trainer.output_dir,
                  f'{args.augmentation_ratio}_{args.dataset}_accuracies.json', 0)

    # Plot Predictions
    trainer.plot_predictions(num_samples=2)
