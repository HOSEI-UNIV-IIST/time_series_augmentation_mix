#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/04/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

# tuning.py
import json
import os

import optuna
from optuna import Trial


class HyperparametersTuner:
    def __init__(self, trainer):
        self.trainer = trainer

    def objective(self, trial: Trial):
        """
        Objective function for Optuna hyperparameter tuning.

        Parameters:
        - trial (optuna.Trial): An Optuna trial to suggest hyperparameters.

        Returns:
        - float: Validation loss of the model.
        """
        # Sample hyperparameters for LSTM and optimizer/scheduler
        params = self.sample_hyperparameters(trial)

        # Initialize model and optimizer with suggested parameters
        self.trainer.model = self.trainer.initialize_model(hidden_size=params['hidden_size'],
                                                           n_layers=params['n_layers'],
                                                           num_filters=params['num_filters'],
                                                           kernel_size=params['kernel_size'])
        self.trainer.optimizer, self.trainer.scheduler = self.trainer.setup_optimizer_and_scheduler(
            learning_rate=params['learning_rate'],
            optimizer_type=params['optimizer'],
            factor=params['factor'],
            patience=params['patience']
        )

        # Train and validate with fewer epochs for tuning
        val_losses, _ = self.trainer.train_and_validate(nb_epochs=10)  # Use a lower epoch count for tuning speed
        return val_losses[-1]

    def sample_hyperparameters(self, trial):
        # Define model-specific and optimizer parameters
        model_name = self.trainer.args.model

        # Set hidden_size and n_layers for models with RNN or GRU/LSTM layers
        if model_name != "cnn":  # Only "cnn" model is an exception, other models have RNN/GRU/LSTM layers
            hidden_size = trial.suggest_categorical('hidden_size', [50, 100, 150, 200])
            n_layers = trial.suggest_categorical('n_layers', [1, 2, 3])
        else:
            hidden_size, n_layers = None, None

        # Set num_filters and kernel_size only for models that include CNN layers
        if model_name in ["cnn", "cnn_lstm", "cnn_gru", "cnn_bilstm", "cnn_bigru", "gru_cnn_gru","bilstm_cnn_bilstm", "bigru_cnn_bigru", "lstm_cnn_lstm"]:
            num_filters = trial.suggest_categorical('num_filters', [32, 64, 96, 128])
            kernel_size = trial.suggest_categorical('kernel_size', [2, 3])
        else:
            num_filters, kernel_size = None, None

        # Common hyperparameters
        learning_rate = trial.suggest_categorical('learning_rate', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'nadam'])
        factor = trial.suggest_uniform('factor', 0.1, 0.5)
        patience = trial.suggest_int('patience', 40, 55)

        return {
            'hidden_size': hidden_size,
            'n_layers': n_layers,
            'num_filters': num_filters,
            'kernel_size': kernel_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'factor': round(factor, 2),
            'patience': patience
        }

    def tune_hyperparameters(self, n_trials=50):
        """
        Runs the hyperparameter optimization process using Optuna.

        Parameters:
        - n_trials (int): Number of trials for Optuna study.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        best_params = study.best_params
        self.save_best_params(best_params)

    def save_best_params(self, best_params):
        """
        Saves the best hyperparameters to a JSON file.

        Parameters:
        - best_params (dict): Best parameters from Optuna study.
        """
        os.makedirs(self.trainer.best_params_dir, exist_ok=True)
        file_path = os.path.join(self.trainer.best_params_dir, f"{self.trainer.best_params_file_name}.json")
        with open(file_path, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Best parameters saved to {file_path}")

    def load_best_params(self):
        """
        Loads the best hyperparameters from a JSON file.

        Returns:
        - dict: Dictionary of best hyperparameters.
        """
        file_path = os.path.join(self.trainer.best_params_dir, f"{self.trainer.best_params_file_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                best_params = json.load(f)
            print(f"Best parameters loaded from {file_path}")
            return best_params
        else:
            raise FileNotFoundError(f"Best parameters file not found at {file_path}")
