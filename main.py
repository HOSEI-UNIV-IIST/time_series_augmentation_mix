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
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from models import custom_models as mod, mix_augmentation_refined as aug
from models.tuning import HyperparametersTuner
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
        self.x_train, self.y_train, self.x_test, self.y_test, self.augmentation_tags, self.duration = self.load_and_prepare_data()

        # Automatically infer input shape from training data
        self.input_shape = (self.x_train.shape[1], self.x_train.shape[2])  # (sequence_length, num_features)

        # Initialize model
        self.model = self.initialize_model()
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler()

        # Set loss function
        self.criterion = nn.MSELoss()

        # Create DataLoaders
        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=args.batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(TensorDataset(self.x_test, self.y_test), batch_size=args.batch_size,
                                      shuffle=False)

        self.model_prefix = f"{args.dataset}_{self.augmentation_tags}"
        # Create directories if they don't exist
        self.log_dir = os.path.join(args.log_dir, str(self.device), str(args.model), str(args.dataset),
                                    str(args.augmentation_ratio))
        self.weight_dir = os.path.join(args.weight_dir, str(self.device), str(args.model), str(args.dataset),
                                       str(args.augmentation_ratio))
        self.output_dir = os.path.join(args.output_dir, str(self.device), str(args.model), str(args.dataset),
                                       str(args.augmentation_ratio))
        # Directory and filename setup for best parameters
        self.best_params_dir = os.path.join(args.output_dir, str(self.device), str(args.model), str(args.dataset),
                                            str(args.augmentation_ratio))
        self.best_params_file_name = f"{args.dataset}_back{self.look_back}_step{self.n_steps}"

        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

            # Initialize the CSV logger
        self.csv_logger = SummaryWriter(log_dir=os.path.join(self.log_dir, self.model_prefix))

    def load_and_prepare_data(self):
        x_train_, y_train_, x_test_, y_test_ = get_datasets(self.args)

        # Augment data before slicing windows
        if args.original:
            augmentation_tags = '_original'
            duration = 0
        else:
            print(f"Augmentation method: {args.augmentation_method}")
            started_at = time.time() * 1000  # Convert to milliseconds
            x_train, y_train, augmentation_tags = aug.run_augmentation_refined(x_train_, y_train_, args)
            ended_at = time.time() * 1000  # Convert to milliseconds
            duration = ended_at - started_at
            print(f"Augmentation process took {duration:.2f} ms")
            print(
                f"x_train shape after augmentation: {x_train.shape}, y_train shape after augmentation: {y_train.shape}")

        # Prepare data for multi-step forecasting after augmentation
        x_train, y_train = prepare_multi_step_data(x_train_, y_train_, self.look_back, self.n_steps)
        x_test, y_test = prepare_multi_step_data(x_test_, y_test_, self.look_back, self.n_steps)

        # Convert to tensors and check shapes
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        print(f"x_train shape after preparation: {x_train.shape}")  # Expected: (batch_size, look_back, num_features)
        print(f"y_train shape after preparation: {y_train.shape}")  # Expected: (batch_size, n_steps)

        return x_train, y_train, x_test, y_test, augmentation_tags, duration

    def initialize_model(self, hidden_size=100, n_layers=1, num_filters=64, kernel_size=3):
        # Initialize the model using automatically inferred input_shape and n_steps
        model = mod.get_model(self.args.model, self.input_shape, n_steps=self.n_steps, hidden_size=hidden_size,
                              n_layers=n_layers, num_filters=num_filters, kernel_size=kernel_size).to(self.device)
        return self.wrap_model_with_dataparallel(model)

    def wrap_model_with_dataparallel(self, model):
        if torch.cuda.is_available() and self.args.gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.args.gpus)))
        return model.to(self.device)

    def setup_optimizer_and_scheduler(self, learning_rate=1e-3, optimizer_type='adam', factor=0.1, patience=10):
        if optimizer_type == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == "nadam":
            optimizer = optim.NAdam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == "adadelta":
            optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate, rho=0.95, eps=1e-8)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
        n_epochs = int(np.ceil(self.args.iterations * (self.args.batch_size / self.x_train.shape[0])))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                                      min_lr=1e-7, cooldown=int(np.ceil(n_epochs / 40.)))

        return optimizer, scheduler

    def train_and_validate(self, nb_epochs):
        # Lists to store loss and accuracy values
        val_losses = []
        val_accuracies = []

        # Early stopping parameters
        best_val_loss = float('inf')
        if args.model == "fcnn":
            early_stopping_patience = 100
        elif args.model in ["lstm", "gru"]:
            early_stopping_patience = 100
        else:
            early_stopping_patience = 20

        epochs_no_improve = 0

        for epoch in range(nb_epochs):
            train_loss, train_accu = self.train_one_epoch()
            val_loss, val_accuracy = self.validate_one_epoch()

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(
                f'Epoch {epoch + 1}/{nb_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accu:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the best model
                torch.save(self.model.state_dict(),
                           os.path.join(self.weight_dir, f"{self.model_prefix}_best_weights.pth"))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        return val_losses, val_accuracies

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for data, labels in self.train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Calculate loss and accuracy
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        epoch_loss /= len(self.train_loader)
        epoch_accuracy = epoch_correct / epoch_total

        return epoch_loss, epoch_accuracy

    def validate_one_epoch(self):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        val_loss /= len(self.test_loader)
        val_accuracy = val_correct / val_total

        return val_loss, val_accuracy

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        mae_sum = 0
        mse_sum = 0
        rmse_sum = 0
        mape_sum = 0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)

                # Convert outputs and labels to CPU for metric calculations
                outputs_np = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                all_labels.extend(labels_np.flatten())
                all_outputs.extend(outputs_np.flatten())

                # Compute MSE loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate MAE, MSE, RMSE, MAPE for this batch
                mae_sum += np.abs(outputs_np - labels_np).mean()
                mse_sum += ((outputs_np - labels_np) ** 2).mean()
                rmse_sum += np.sqrt(((outputs_np - labels_np) ** 2).mean())
                mape_sum += (np.abs(
                    (labels_np - outputs_np) / np.maximum(labels_np, 1e-10))).mean() * 100  # Avoid zero division

                total_samples += 1

        # Average metrics across all batches
        avg_loss = total_loss / total_samples
        avg_mae = mae_sum / total_samples
        avg_mse = mse_sum / total_samples
        avg_rmse = rmse_sum / total_samples
        avg_mape = mape_sum / total_samples

        # Calculate R2 score for all predictions vs true values
        r2 = r2_score(all_labels, all_outputs)

        # Define accuracy as an inverse function of the MAE
        accuracy = 1 / (1 + avg_mae)

        # Print or return all metrics
        return {
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "r2": r2,
            "mae": avg_mae,
            "mse": avg_mse,
            "rmse": avg_rmse,
            "mape": avg_mape
        }

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

    tuner = HyperparametersTuner(trainer)

    if args.tune:
        # Update Trainer with best hyperparameters
        print("Starting hyperparameter tuning...")
        tuner.tune_hyperparameters(n_trials=5)
        best_params = tuner.load_best_params()
        trainer.model = trainer.initialize_model(
            hidden_size=best_params.get('hidden_size', 100),
            n_layers=best_params.get('n_layers', 1),
            num_filters=best_params.get('num_filters', 64),
            kernel_size=best_params.get('kernel_size', 3)
        )

        trainer.optimizer, trainer.scheduler = trainer.setup_optimizer_and_scheduler(
            learning_rate=best_params['learning_rate'],
            optimizer_type=best_params['optimizer'],
            factor=best_params['factor'],
            patience=best_params['patience']
        )

    nb_iterations = args.iterations
    nb_epochs = int(np.ceil(nb_iterations * (args.batch_size / trainer.x_train.shape[0])))

    if args.train:
        val_losses, val_accuracies = trainer.train_and_validate(nb_epochs)
        if args.save:
            torch.save(trainer.model.state_dict(),
                       os.path.join(trainer.weight_dir, f"{trainer.model_prefix}_final_weights.pth"))
            cache = CacheLossAccuracy(val_losses, val_accuracies, trainer.output_dir, trainer.model_prefix)
            cache.save_training_data()
            cache.plot_training_data()

    print("Evaluation")
    # Run evaluation and retrieve metrics
    accuracies = trainer.evaluate()

    # Print the evaluation metrics
    print(f"Best Evaluation - Loss: {accuracies['avg_loss']:.4f}, "
          f"Accuracy: {accuracies['accuracy']:.4f}, "
          f"R2: {accuracies['r2']:.4f}, "
          f"MAE: {accuracies['mae']:.4f}, "
          f"MSE: {accuracies['mse']:.4f}, "
          f"RMSE: {accuracies['rmse']:.4f}, "
          f"MAPE: {accuracies['mape']:.4f}")

    # Save the accuracy metrics
    file_name = f"{args.augmentation_ratio}_{args.dataset}_accuracies.json"
    save_accuracy(accuracies, f"{args.dataset}_{trainer.augmentation_tags}", trainer.output_dir, file_name,
                  trainer.duration)

    trainer.plot_validation_predictions(num_samples=100)
