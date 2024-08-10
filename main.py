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


def wrap_model_with_dataparallel(model, device, gpus):
    """
    Wrap the model using DataParallel to use multiple GPUs if available.

    Parameters:
    model (torch.nn.Module): The model to wrap.
    device (torch.device): The device to move the model to.
    gpus (int): The number of GPUs to use.

    Returns:
    torch.nn.Module: The wrapped model.
    """
    if torch.cuda.is_available():
        assert torch.cuda.device_count() >= gpus, f"Not enough GPUs available. Required: {gpus}, Available: {torch.cuda.device_count()}"
        gpu_ids = [i for i in range(gpus)]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        model = nn.DataParallel(model, device_ids=gpu_ids)
    return model.to(device)


if __name__ == '__main__':
    args = argument_parser()

    # Check if CUDA is available and set the device
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
        print(f"Augmentation method: {args.augmentation_method}")
        started_at = time.time() * 1000  # Convert to milliseconds
        x_train, y_train, augmentation_tags = aug.run_augmentation_refined(x_train.numpy(), y_train.numpy(), args)
        ended_at = time.time() * 1000  # Convert to milliseconds
        duration = ended_at - started_at
        print(f"Augmentation process took {duration:.2f} ms")
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
    nb_epochs = 1#int(np.ceil(nb_iterations * (batch_size / x_train.shape[0])))
    print(f'epoch: {nb_epochs}')

    # `mod.get_model` returns a PyTorch model
    model = mod.get_model(args.model, input_shape, nb_class).to(device)

    # Wrap the model using DataParallel to use multiple GPUs if available
    model = wrap_model_with_dataparallel(model, device, args.gpus)

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
    reduce_lr = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(np.ceil(nb_epochs / 30.)),
                                  min_lr=1e-5, cooldown=int(np.ceil(nb_epochs / 40.)))

    # Model saving and logging setup
    if args.save:
        model_prefix = f"{args.dataset}{augmentation_tags}"
        # Create directories if they don't exist
        log_dir = os.path.join(args.log_dir, str(device), str(args.model), str(args.dataset),
                               str(args.augmentation_ratio))
        weight_dir = os.path.join(args.weight_dir, str(device), str(args.model), str(args.dataset),
                                  str(args.augmentation_ratio))
        output_dir = os.path.join(args.output_dir, str(device), str(args.model), str(args.dataset),
                                  str(args.augmentation_ratio))

        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize the CSV logger
        csv_logger = SummaryWriter(log_dir=os.path.join(log_dir, model_prefix))

    # Ensure compatibility before creating DataLoaders
    if not args.original:
        assert x_train.size(0) == y_train.size(0), "x_train and y_train must have the same number of samples"
        assert x_test.size(0) == y_test.size(0), "x_test and y_test must have the same number of samples"

    # Create DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Lists to store loss and accuracy values
    val_losses = []
    val_accuracies = []

    # Early stopping parameters
    best_val_loss = float('inf')
    early_stopping_patience = 2
    epochs_no_improve = 0

    # Training loop
    if args.train:
        for epoch in range(nb_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate loss and accuracy
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

            epoch_loss /= len(train_loader)
            epoch_accuracy = epoch_correct / epoch_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

            val_loss /= len(test_loader)
            val_accuracy = val_correct / val_total

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(
                f'Epoch {epoch + 1}/{nb_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

            # Log the loss and adjust learning rate if needed
            reduce_lr.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the best model
                torch.save(model.state_dict(), os.path.join(weight_dir, f"{model_prefix}_best_weights.pth"))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        if args.save:
            torch.save(model.state_dict(), os.path.join(weight_dir, f"{model_prefix}_final_weights.pth"))
            cache = CacheLossAccuracy(val_losses, val_accuracies, output_dir, model_prefix)
            cache.save_training_data()
            cache.plot_training_data()

    # Evaluation
    else:
        model.load_state_dict(torch.load(os.path.join(weight_dir, f"{model_prefix}_final_weights.pth")))
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
    accur = f'{(100 * correct) / total:.2f}%'
    file_name = f'{args.augmentation_ratio}_{args.dataset}_accuracies.json'
    print(f'Best Train Accuracy: {accur}')
    save_accuracy(accur, f'{args.dataset}{augmentation_tags}', output_dir, file_name, duration)


    # Save predictions
    # if args.save:
    #     y_preds = []
    #     with torch.no_grad():
    #         for data, labels in test_loader:
    #             data = data.to(device)
    #             outputs = model(data)
    #             _, predicted = torch.max(outputs.data, 1)
    #             y_preds.extend(predicted.cpu().numpy())
    #     y_preds = np.array(y_preds)
    #     out = f'{(100 * correct) / total:.2f}%'
    #     np.savetxt(os.path.join(output_dir, f"{out}.txt"), y_preds, fmt="%d")
