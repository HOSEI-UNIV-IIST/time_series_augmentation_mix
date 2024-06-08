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

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

def get_device():
    """Determine if a GPU is available and return the appropriate device context."""
    if tf.config.list_physical_devices('GPU'):
        return '/GPU:0'
    else:
        return '/CPU:0'

def build_model():
    """Build a simple neural network model."""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

    # Build model
    model = build_model()

    # Print device being used
    device = get_device()
    print(f"Using device: {device}")

    # Train the model
    with tf.device(device):
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Evaluate the model
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {acc:.2f}")

if __name__ == '__main__':
    main()