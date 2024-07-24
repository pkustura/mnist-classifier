import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import mnist_loader

class MNISTNet(nn.Module):
    def __init__(self, layer_sizes):
        super(MNISTNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.sigmoid(layer(x))
            if i == len(self.layers) - 2:  # If it's the second to last layer
                break  # Don't apply sigmoid to the last layer
        x = self.layers[-1](x)  # Last layer without sigmoid
        return x

def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # Load data
    (train_data, train_labels), (test_data, test_labels) = mnist_loader.structured_load_torch()

    # Convert to PyTorch tensors
    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels)
    test_data = torch.from_numpy(test_data)
    test_labels = torch.from_numpy(test_labels)

    # Create datasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = MNISTNet([784, 16, 16, 10])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, criterion, optimizer, epochs=30)

    # Evaluate the model
    accuracy = evaluate(model, test_loader)
    print(f'Test Accuracy: {accuracy}%')

if __name__ == "__main__":
    main()
