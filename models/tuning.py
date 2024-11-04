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
import time
import tracemalloc
import optuna
from optuna import Trial


class HyperparametersTuner:
    def __init__(self, trainer, accuracy_weight=0.5):
        """
        Initializes the hyperparameter tuner.

        Parameters:
        - trainer: The training module responsible for training and validation.
        - accuracy_weight: Weight for accuracy in the composite score. Higher values give more importance to accuracy.
        """
        self.trainer = trainer
        self.accuracy_weight = accuracy_weight  # Weight for accuracy in composite score

    def objective(self, trial: Trial):
        """
        Objective function for Optuna hyperparameter tuning. Minimizes a composite score based on validation loss
        and accuracy.

        Parameters:
        - trial (optuna.Trial): An Optuna trial to suggest hyperparameters.

        Returns:
        - float: Composite score based on validation loss and accuracy.
        """
        # Sample hyperparameters for model and optimizer
        params = self.sample_hyperparameters(trial)

        # Initialize model and optimizer with suggested parameters
        self.trainer.model = self.trainer.initialize_model(
            hidden_size=params['hidden_size'],
            n_layers=params['n_layers'],
            num_filters=params['num_filters'],
            kernel_size=params['kernel_size']
        )
        self.trainer.optimizer, self.trainer.scheduler = self.trainer.setup_optimizer_and_scheduler(
            learning_rate=params['learning_rate'],
            optimizer_type=params['optimizer'],
            factor=params['factor'],
            patience=params['patience']
        )

        # Track time and memory usage during tuning
        tracemalloc.start()
        start_time = time.time()

        # Use existing train_and_validate with nb_epochs from main.py
        nb_epochs = self.trainer.nb_epochs
        val_losses, val_accuracies = self.trainer.train_and_validate(nb_epochs=nb_epochs)

        # Measure elapsed time and memory
        duration_seconds = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Convert duration to hours:minutes:seconds format and memory to MB
        duration_hms = time.strftime('%H:%M:%S', time.gmtime(duration_seconds))
        memory_mb = peak / (1024 * 1024)  # Convert bytes to MB

        # Update params with additional information
        params.update({
            'duration': duration_hms,
            'memory_used_mb': round(memory_mb, 2),
            'epochs_run': len(val_losses)
        })

        # Calculate composite score for Optuna, using the formula:
        # score = val_loss - accuracy_weight * val_accuracy
        final_val_loss = val_losses[-1]
        final_val_accuracy = val_accuracies[-1]
        combined_score = final_val_loss - self.accuracy_weight * final_val_accuracy

        # Save additional metrics for analysis
        trial.set_user_attr("final_val_loss", final_val_loss)
        trial.set_user_attr("final_val_accuracy", final_val_accuracy)
        trial.set_user_attr("duration_hms", duration_hms)
        trial.set_user_attr("memory_used_mb", round(memory_mb, 2))
        trial.set_user_attr("epochs_run", len(val_losses))

        # Store params in trial for Optuna's record-keeping
        trial.set_user_attr("params", params)

        # Return combined score for Optuna to minimize
        return combined_score

    def sample_hyperparameters(self, trial):
        """Sample model-specific hyperparameters for optimization."""
        model_name = self.trainer.args.model

        # Set hidden_size and n_layers for models with RNN/GRU/LSTM layers
        if model_name != "cnn":  # Only "cnn" model is an exception
            hidden_size = trial.suggest_categorical('hidden_size', [50, 100, 150, 200])
        else:
            hidden_size, n_layers = None, None

        # Set num_filters and kernel_size for models with CNN layers
        if model_name in [
            "cnn", "lstm", "gru", "cnn_lstm", "cnn_gru", "cnn_bigru", "cnn_bilstm",
            "gru_cnn_gru", "lstm_cnn_lstm", "bigru_cnn_bigru", "bilstm_cnn_bilstm"
        ]:
            n_layers = trial.suggest_categorical('n_layers', [1, 2, 3])
            num_filters = trial.suggest_categorical('num_filters', [64, 96, 128])
            kernel_size = trial.suggest_categorical('kernel_size', [2, 3])
        else:
            num_filters, kernel_size = None, None

        # Common hyperparameters
        learning_rate = trial.suggest_categorical('learning_rate', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'nadam'])
        factor = trial.suggest_float('factor', 0.1, 0.5)
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

        # Retrieve and save best parameters with additional metrics
        best_params = study.best_trial.user_attrs["params"]  # Use full params dictionary
        self.save_best_params(best_params)

    def save_best_params(self, best_params):
        """
        Saves the best hyperparameters and additional info to a JSON file.

        Parameters:
        - best_params (dict): Best parameters from Optuna study, with added metrics.
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
