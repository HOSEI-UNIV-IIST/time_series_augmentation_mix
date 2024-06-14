#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/14/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import os
import numpy as np
import matplotlib.pyplot as plt


class CacheLossAccuracy:
    def __init__(self, losses, accuracies, output_dir, model_prefix):
        self.losses = losses
        self.accuracies = accuracies
        self.output_dir = output_dir
        self.model_prefix = model_prefix

    def save_training_data(self):
        # Save loss data
        loss_file = os.path.join(self.output_dir, f"{self.model_prefix}_losses.npy")
        np.save(loss_file, np.array(self.losses))

        # Save accuracy data
        accuracy_file = os.path.join(self.output_dir, f"{self.model_prefix}_accuracies.npy")
        np.save(accuracy_file, np.array(self.accuracies))

    def plot_training_data(self):
        # Load loss and accuracy data
        losses = np.load(os.path.join(self.output_dir, f"{self.model_prefix}_losses.npy"))
        accuracies = np.load(os.path.join(self.output_dir, f"{self.model_prefix}_accuracies.npy"))

        # Plot loss data
        plt.figure()
        plt.plot(losses, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{self.model_prefix}_loss_plot.png"))

        # Plot accuracy data
        plt.figure()
        plt.plot(accuracies, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{self.model_prefix}_accuracy_plot.png"))
