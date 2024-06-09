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

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import models.custom_models as mod
import utils.datasets as ds
from utils.input_data import get_datasets, run_augmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs augmentation model.')
    # General settings
    parser.add_argument('--gpus', type=str, default="0", help="Sets CUDA_VISIBLE_DEVICES")
    parser.add_argument('--dataset', type=str, default='CBF', help='Name of dataset to test (required, ex: unipen1a)')
    parser.add_argument('--model', type=str, default="lstm1", help="Set model name")
    parser.add_argument('--train', default=True, action="store_true", help="Train?")
    parser.add_argument('--save', default=True, action="store_true", help="save to disk?")

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=1, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=20240609, help="Randomization seed")
    parser.add_argument('--original', type=bool, default=False, help="Original dataset without augmentation")
    parser.add_argument('--jitter', type=bool, default=False, help="Jitter preset augmentation")
    parser.add_argument('--scaling', type=bool, default=False, help="Scaling preset augmentation")
    parser.add_argument('--permutation', type=bool, default=False,
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', type=bool, default=False,
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', type=bool, default=False, help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', type=bool, default=False, help="Time warp preset augmentation")
    parser.add_argument('--windowslice', type=bool, default=False, help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', type=bool, default=False, help="Window warp preset augmentation")
    parser.add_argument('--rotation', type=bool, default=False, help="Rotation preset augmentation")
    parser.add_argument('--spawner', type=bool, default=False, help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', type=bool, default=False, help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', type=bool, default=False, help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', type=bool, default=False, help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', type=bool, default=False,
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', type=bool, default=False,
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="default", help="Anything extra")

    # File settings
    parser.add_argument('--preset_files', default=False, action="store_true", help="Use preset files")
    parser.add_argument('--ucr', default=False, action="store_true", help="Use UCR 2015")
    parser.add_argument('--ucr2018', default=False, action="store_true", help="Use UCR 2018")
    parser.add_argument('--data_dir', type=str, default="data", help="Data dir")
    parser.add_argument('--train_data_file', type=str, default="", help="Train data file")
    parser.add_argument('--train_labels_file', type=str, default="", help="Train label file")
    parser.add_argument('--test_data_file', type=str, default="", help="Test data file")
    parser.add_argument('--test_labels_file', type=str, default="", help="Test label file")
    parser.add_argument('--test_split', type=int, default=0, help="test split")
    parser.add_argument('--weight_dir', type=str, default="weights", help="Weight path")
    parser.add_argument('--log_dir', type=str, default="logs", help="Log path")
    parser.add_argument('--output_dir', type=str, default="output", help="Output path")
    parser.add_argument('--normalize_input', default=False, action="store_true", help="Normalize between [-1,1]")
    parser.add_argument('--delimiter', type=str, default=" ", help="Delimiter")

    # Network settings
    parser.add_argument('--optimizer', type=str, default="sgd", help="Which optimizer")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
    parser.add_argument('--validation_split', type=int, default=0, help="size of validation set")
    parser.add_argument('--iterations', type=int, default=10000, help="Number of iterations")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('--verbose', type=int, default=1, help="verbose")

    args = parser.parse_args()
    # print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Number of classes and dimensions
    nb_class = ds.nb_classes(args.dataset)
    nb_dims = ds.nb_dims(args.dataset)

    # Load data
    x_train_, y_train_, x_test_, y_test_ = get_datasets(args)
    print(f"x_train_ shape before augmentation: {x_train_.shape}, y_train_ shape before augmentation: {y_train_.shape}")
    # Calculate number of timesteps
    nb_timesteps = int(x_train_.shape[1] / nb_dims)
    input_shape = (nb_timesteps, nb_dims)

    # Process data
    x_test = x_test_.reshape((-1, input_shape[0], input_shape[1]))
    x_train = x_train_.reshape((-1, input_shape[0], input_shape[1]))

    # Convert to torch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    # Offset class labels and convert to one-hot encoding
    y_train = torch.tensor(ds.class_offset(y_train_, args.dataset), dtype=torch.long)
    y_test = torch.tensor(ds.class_offset(y_test_, args.dataset), dtype=torch.long)

    # One-hot encode the labels
    y_train = F.one_hot(y_train, nb_class).float()
    y_test = F.one_hot(y_test, nb_class).float()

    # Augment data
    if args.original:
        augmentation_tags = '_original'
    else:
        x_train, y_train, augmentation_tags = run_augmentation(x_train.numpy(), y_train.numpy(), args)
        print(f"x_train shape after augmentation: {x_train.shape}, y_train shape after augmentation: {y_train.shape}")

    # Convert augmented data to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Ensure compatibility before creating DataLoaders
    assert x_train.size(0) == y_train.size(0), "x_train and y_train must have the same number of samples"
    assert x_test.size(0) == y_test.size(0), "x_test and y_test must have the same number of samples"

    # Calculate iterations and epochs
    nb_iterations = args.iterations
    batch_size = args.batch_size
    nb_epochs = int(np.ceil(nb_iterations * (batch_size / x_train.shape[0])))

    # Assume `mod.get_model` returns a PyTorch model
    model = mod.get_model(args.model, input_shape, nb_class).to(device)

    # Define optimizer and learning rate scheduler
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "nadam":
        optimizer = optim.NAdam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, eps=1e-8)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    reduce_lr = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=int(np.ceil(nb_epochs / 20.)),
                                  min_lr=1e-5, cooldown=int(np.ceil(nb_epochs / 40.)))

    # Model saving and logging setup
    if args.save:
        model_prefix = f"{args.model}/{args.dataset}/{args.dataset}{augmentation_tags}"
        # Create directories if they don't exist
        weight_dir = os.path.join(args.weight_dir, str(device), str(model_prefix))
        log_dir = os.path.join(args.log_dir, str(device), str(model_prefix))

        if not os.path.exists(args.weight_dir):
            os.mkdir(args.weight_dir)
        if not os.path.exists(os.path.join(args.weight_dir, str(device))):
            os.mkdir(os.path.join(args.weight_dir, str(device)))
        if not os.path.exists(os.path.join(args.weight_dir, str(device), str(args.model))):
            os.mkdir(os.path.join(args.weight_dir, str(device), str(args.model)))
        if not os.path.exists(os.path.join(args.weight_dir, str(device), str(args.model), str(args.dataset))):
            os.mkdir(os.path.join(args.weight_dir, str(device), str(args.model), str(args.dataset)))
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)

        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        if not os.path.exists(os.path.join(args.log_dir, str(device))):
            os.mkdir(os.path.join(args.log_dir, str(device)))
        if not os.path.exists(os.path.join(args.log_dir, str(device), str(args.model))):
            os.mkdir(os.path.join(args.log_dir, str(device), str(args.model)))
        if not os.path.exists(os.path.join(args.log_dir, str(device), str(args.model), str(args.dataset))):
            os.mkdir(os.path.join(args.log_dir, str(device), str(args.model), str(args.dataset)))
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # Initialize the CSV logger
        csv_logger = SummaryWriter(log_dir=log_dir)

    # Ensure compatibility before creating DataLoaders
    if not args.original:
        assert x_train.size(0) == y_train.size(0), "x_train and y_train must have the same number of samples"
        assert x_test.size(0) == y_test.size(0), "x_test and y_test must have the same number of samples"

    # Create DataLoaders
    print(f"x_train shape before DataLoader: {x_train.shape}, y_train shape before DataLoader: {y_train.shape}")
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    if args.train:
        model.train()
        for epoch in range(nb_epochs):
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{nb_epochs}, Loss: {loss.item()}')

            # Log the loss and adjust learning rate if needed
            if args.save:
                csv_logger.add_scalar('Loss/train', loss.item(), epoch)
            reduce_lr.step(loss.item())

        if args.save:
            torch.save(model.state_dict(),
                       os.path.join(args.weight_dir, model_prefix, f"{model_prefix}_final_weights.pth"))

    # Evaluation
    else:
        model.load_state_dict(
            torch.load(os.path.join(args.weight_dir, model_prefix, f"{model_prefix}_final_weights.pth")))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            labels = torch.argmax(labels, dim=1)  # Convert one-hot encoded labels to class indices
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Save predictions
    if args.save:
        y_preds = []
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                y_preds.extend(predicted.cpu().numpy())
        y_preds = np.array(y_preds)
        np.savetxt(os.path.join(args.output_dir, f"{model_prefix}_{100 * correct / total:.15f}.txt"), y_preds, fmt="%d")

    # Load best model and evaluate
    if os.path.exists(os.path.join(args.weight_dir, model_prefix, f"{model_prefix}_best_train_acc_weights.pth")):
        model.load_state_dict(
            torch.load(os.path.join(args.weight_dir, model_prefix, f"{model_prefix}_best_train_acc_weights.pth")))
        model.eval()
        correct = 0
        total = 0
        with torch.no.grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Best Train Acc, Test Accuracy: {100 * correct / total:.2f}%')
