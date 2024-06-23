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

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, nb_class):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, nb_class)

    def forward(self, x):
        # Forward propagate the LSTM
        out, _ = self.lstm(x)
        # Select the output of the last time step
        out = out[:, -1, :]
        return self.fc(out)

def train(model, device, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Reshape data for sequence processing
            data = data.view(-1, 28, 28).to(device)  # batch_size, sequence_length, input_size
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu_old')
    print(f'Using device: {device}')

    # Hyperparameters
    input_size = 28  # Each row of the image is treated as a sequence step
    hidden_size = 100  # Number of features in the hidden state of the LSTM
    nb_class = 10  # Number of classes in MNIST
    learning_rate = 0.001
    batch_size = 64
    epochs = 5

    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # Model
    model = SimpleLSTM(input_size, hidden_size, nb_class).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, device, train_loader, optimizer, criterion, epochs)

if __name__ == "__main__":
    main()