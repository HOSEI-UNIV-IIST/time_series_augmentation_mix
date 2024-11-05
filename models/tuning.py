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
import yaml
from optuna import Trial, pruners

class HyperparametersTuner:
    def __init__(self, trainer, accuracy_weight=0.5, loss_weight=0.5, config_path="config/hyperparameters.yml"):
        """
        Initializes the hyperparameter tuner.

        Parameters:
        - trainer: The training module responsible for training and validation.
        - accuracy_weight: Weight for accuracy in the composite score.
        - loss_weight: Weight for loss in the composite score.
        - config_path: Path to YAML/JSON configuration file for hyperparameter ranges.
        """
        self.trainer = trainer
        self.accuracy_weight = accuracy_weight
        self.loss_weight = loss_weight
        self.config = self.load_config(config_path)

    def load_config(self, path):
        """Loads configuration from a YAML or JSON file."""
        with open(path, "r") as file:
            if path.endswith('.yml') or path.endswith('.yaml'):
                return yaml.safe_load(file)
            elif path.endswith('.json'):
                return json.load(file)
            else:
                raise ValueError("Unsupported config file format. Use .yml, .yaml, or .json.")

    def objective(self, trial: Trial):
        """
        Objective function for Optuna hyperparameter tuning. Minimizes a composite score based on validation loss
        and accuracy.

        Parameters:
        - trial (optuna.Trial): An Optuna trial to suggest hyperparameters.

        Returns:
        - float: Composite score based on validation loss and accuracy.
        """
        # Sample hyperparameters based on the model type
        params = self.sample_hyperparameters(trial)

        # Initialize model and optimizer with suggested parameters
        model_name = self.trainer.args.model
        if model_name in ["cnn"]:
            self.trainer.model = self.trainer.initialize_model(
                num_filters=params['num_filters'],
                kernel_size=params['kernel_size'],
                pool_size=params['pool_size'],
                dropout=params['dropout']
            )
        elif model_name in ["lstm", "gru"]:
            self.trainer.model = self.trainer.initialize_model(
                hidden_size=params['hidden_size'],
                n_layers=params['n_layers'],
                dropout=params['dropout']
            )
        else:  # Hybrid CNN-RNN models
            self.trainer.model = self.trainer.initialize_model(
                hidden_size=params['hidden_size'],
                n_layers=params['n_layers'],
                num_filters=params['num_filters'],
                kernel_size=params['kernel_size'],
                pool_size=params['pool_size'],
                dropout=params['dropout']
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

        # Use existing train_and_validate with early stopping
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

        # Calculate composite score for Optuna, using a weighted formula:
        # score = loss_weight * val_loss - accuracy_weight * val_accuracy
        final_val_loss = val_losses[-1]
        final_val_accuracy = val_accuracies[-1]
        combined_score = self.loss_weight * final_val_loss - self.accuracy_weight * final_val_accuracy

        # Save additional metrics for analysis
        trial.set_user_attr("final_val_loss", final_val_loss)
        trial.set_user_attr("final_val_accuracy", final_val_accuracy)
        trial.set_user_attr("duration_hms", duration_hms)
        trial.set_user_attr("memory_used_mb", round(memory_mb, 2))
        trial.set_user_attr("epochs_run", len(val_losses))

        # Logging progress
        print(f"Trial {trial.number}: Loss = {final_val_loss}, Accuracy = {final_val_accuracy}, "
              f"Combined Score = {combined_score}, Duration = {duration_hms}, Memory = {memory_mb:.2f} MB")

        # Store params in trial for Optuna's record-keeping
        trial.set_user_attr("params", params)

        # Return combined score for Optuna to minimize
        return combined_score

    def sample_hyperparameters(self, trial):
        """Sample model-specific hyperparameters for optimization using configuration file values."""
        model_name = self.trainer.args.model
        params_config = self.config.get(model_name, {})

        # Sampling based on model type
        if model_name == "cnn":
            # Only sample CNN-specific parameters
            num_filters = trial.suggest_categorical('num_filters', params_config.get('num_filters', [64, 128, 256]))
            kernel_size = trial.suggest_categorical('kernel_size', params_config.get('kernel_size', [3, 5]))
            pool_size = trial.suggest_categorical('pool_size', params_config.get('pool_size', [2, 3]))
            dropout = trial.suggest_categorical('dropout', params_config.get('dropout', [0.2, 0.3, 0.5]))

            return {
                'num_filters': num_filters,
                'kernel_size': kernel_size,
                'pool_size': pool_size,
                'dropout': dropout,
                **self.sample_common_hyperparameters(trial, params_config)
            }

        elif model_name in ["lstm", "gru"]:
            # Only sample RNN-specific parameters
            hidden_size = trial.suggest_categorical('hidden_size', params_config.get('hidden_size', [50, 100, 200]))
            n_layers = trial.suggest_categorical('n_layers', params_config.get('n_layers', [1, 2, 3]))
            dropout = trial.suggest_categorical('dropout', params_config.get('dropout', [0.2, 0.3, 0.5]))

            return {
                'hidden_size': hidden_size,
                'n_layers': n_layers,
                'dropout': dropout,
                **self.sample_common_hyperparameters(trial, params_config)
            }

        else:
            # Hybrid CNN-RNN models: sample all relevant parameters
            hidden_size = trial.suggest_categorical('hidden_size', params_config.get('hidden_size', [50, 100, 150]))
            n_layers = trial.suggest_categorical('n_layers', params_config.get('n_layers', [1, 2]))
            num_filters = trial.suggest_categorical('num_filters', params_config.get('num_filters', [32, 64, 128]))
            kernel_size = trial.suggest_categorical('kernel_size', params_config.get('kernel_size', [3, 5]))
            pool_size = trial.suggest_categorical('pool_size', params_config.get('pool_size', [2, 3]))
            dropout = trial.suggest_categorical('dropout', params_config.get('dropout', [0.2, 0.3, 0.5]))

            return {
                'hidden_size': hidden_size,
                'n_layers': n_layers,
                'num_filters': num_filters,
                'kernel_size': kernel_size,
                'pool_size': pool_size,
                'dropout': dropout,
                **self.sample_common_hyperparameters(trial, params_config)
            }

    def sample_common_hyperparameters(self, trial, params_config):
        """Sample common hyperparameters for all models."""
        learning_rate = trial.suggest_categorical('learning_rate',
                                                  params_config.get('learning_rate', [1e-6, 1e-5, 1e-4, 1e-3]))
        optimizer = trial.suggest_categorical('optimizer', params_config.get('optimizer', ['adam', 'sgd', 'rmsprop']))
        factor = trial.suggest_float('factor', *params_config.get('factor', [0.1, 0.5]))
        patience = trial.suggest_int('patience', *params_config.get('patience', [40, 50]))

        return {
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'factor': round(factor, 2),
            'patience': patience
        }

    def tune_hyperparameters(self, n_trials=50):
        """Runs the hyperparameter optimization process using Optuna."""
        study = optuna.create_study(direction="minimize", pruner=pruners.MedianPruner())
        study.optimize(self.objective, n_trials=n_trials)

        # Retrieve and save best parameters with additional metrics
        best_params = study.best_trial.user_attrs["params"]
        self.save_best_params(best_params)

    def save_best_params(self, best_params):
        """Saves the best hyperparameters and additional info to a JSON file."""
        os.makedirs(self.trainer.best_params_dir, exist_ok=True)
        file_path = os.path.join(self.trainer.best_params_dir, f"{self.trainer.best_params_file_name}.json")
        with open(file_path, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Best parameters saved to {file_path}")

    def load_best_params(self):
        """Loads the best hyperparameters from a JSON file."""
        file_path = os.path.join(self.trainer.best_params_dir, f"{self.trainer.best_params_file_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                best_params = json.load(f)
            print(f"Best parameters loaded from {file_path}")
            return best_params
        else:
            raise FileNotFoundError(f"Best parameters file not found at {file_path}")
