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
# main.py

import os

import numpy as np
import torch

from models.training import Trainer
from models.tuning import HyperparametersTuner
from utils.argument_parser import argument_parser
from utils.cache_loss_accuracy import CacheLossAccuracy
from utils.save_result import save_accuracy

if __name__ == '__main__':
    args = argument_parser()
    np.random.seed(args.seed)
    look_back = 7
    n_steps = 6
    trainer = Trainer(args, look_back=look_back, n_steps=n_steps)

    tuner = HyperparametersTuner(trainer, accuracy_weight=0.5, loss_weight=0.5,
                                 config_path="config/hyperparameters.yml")

    if args.tune:
        print("Starting hyperparameter tuning...")
        tuner.tune_hyperparameters(n_trials=1)  # 1 for test
        best_params = tuner.load_best_params()
        trainer.model = trainer.initialize_model(
            hidden_size=best_params.get('hidden_size', 100),
            cnn_layers=best_params.get('cnn_layers', 2),
            am_layers=best_params.get('am_layers', 2),
            gru_layers=best_params.get('gru_layers', 2),
            lstm_layers=best_params.get('lstm_layers', 2),
            bigru_layers=best_params.get('bigru_layers', 2),
            bilstm_layers=best_params.get('bilstm_layers', 2),
            bigru1_layers=best_params.get('bigru1_layers', 2),
            bigru2_layers=best_params.get('bigru2_layers', 2),
            bilstm1_layers=best_params.get('bilstm1_layers', 2),
            bilstm2_layers=best_params.get('bilstm2_layers', 2),
            num_filters=best_params.get('num_filters', 64),
            kernel_size=best_params.get('kernel_size', 3),
            pool_size=best_params.get('pool_size', 2),
            dropout=best_params.get('dropout', 0.2)
        )

        trainer.optimizer, trainer.scheduler = trainer.setup_optimizer_and_scheduler(
            learning_rate=best_params['learning_rate'],
            optimizer_type=best_params['optimizer'],
            factor=best_params['factor'],
            patience=best_params['patience']
        )

    if args.train:
        val_losses, val_accuracies = trainer.train_and_validate()
        if args.save:
            torch.save(trainer.model.state_dict(),
                       os.path.join(trainer.weight_dir, f"{trainer.model_prefix}_final_weights.pth"))
            cache = CacheLossAccuracy(val_losses, val_accuracies, trainer.output_dir, trainer.model_prefix)
            cache.save_training_data()
            cache.plot_training_data()

    print("Evaluation")
    accuracies = trainer.evaluate()

    print(f"Best Evaluation - Loss: {accuracies['avg_loss']:.4f}, "
          f"Accuracy: {accuracies['accuracy']:.4f}, "
          f"R2: {accuracies['r2']:.4f}, "
          f"MAE: {accuracies['mae']:.4f}, "
          f"MSE: {accuracies['mse']:.4f}, "
          f"RMSE: {accuracies['rmse']:.4f}, "
          f"MAPE: {accuracies['mape']:.4f}")

    file_name = f"{trainer.augmentation_tags}_accuracies.json"
    save_accuracy(accuracies, f"{args.dataset}_{trainer.augmentation_tags}", trainer.output_dir, file_name,
                  trainer.duration)

    trainer.plot_validation_predictions(num_samples=200)
    #trainer.plot_realtime_predictions(num_samples=100)

    # SHAPE OR LIME for  INTERPRETATION
    if args.interpret:
        print("Interpreting predictions with SHAP or LIME...")
        trainer.interpret_predictions_bar(samples=10, method=args.interpret_method)
        trainer.interpret_predictions_line(samples=50, method=args.interpret_method)

