# classifier.py


import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader 


# Constants and Hyperparameters
DATA_DIR = "."          # Directory where the dataset is stored
MODEL_PATH = "fashion_mnist_ann.pth"   
BATCH_SIZE = 64                         
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(
    root = DATA_DIR,
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.FashionMNIST(
    root = DATA_DIR,
    train = False,
    download = False,
    transform = transform
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False
)


# The neural network model

class fashionANN(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10):

        # Network layers definition
        # The model is a simple feedforward neural network with 3 layers
        # Input layer: 28x28 pixels (flattened to 784)
        # Hidden layer 1: 128 neurons
        # Hidden layer 2: 64 neurons
        # Output layer: 10 neurons (one for each class)
        # The model uses ReLU activation function for the hidden layers
        # and no activation function for the output layer
      
        super(fashionANN, self).__init__()
        self.flatten = nn.Flatten()       # Layer to flatten the 28x28 image
        self.fc1 = nn.Linear(input_size, hidden_size) # First fully connected layer
        self.relu = nn.ReLU()             # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, num_classes) # Output layer

    """ 
    Forward pass
    The forward method defines how the input data flows through the network
    It takes the input tensor x, flattens it, applies the first fully connected layer,
    applies the ReLU activation function, and then applies the second fully connected layer
    The output of the forward method is the raw scores (logits) for each class
    The forward method is called when we pass data through the model
    The input tensor x is expected to have the shape [batch_size, 1, 28, 28]
    The output tensor will have the shape [batch_size, num_classes]
    """
    def forward(self, x):
        x = self.flatten(x)    # Shape becomes [batch_size, 784]
        x = self.fc1(x)         # Shape becomes [batch_size, hidden_size]
        x = self.relu(x)        # Apply activation, shape remains [batch_size, hidden_size]
        x = self.fc2(x)         # Shape becomes [batch_size, num_classes] - these are raw scores (logits
        return x

# Initialize the model
input_size = 28 * 28 # Flattened size of the input image
hidden_size = 128   # Number of neurons in the hidden layer
num_classes = 10    # Number of classes (10 for Fashion MNIST)

model = fashionANN(input_size = input_size , hidden_size = hidden_size, output_size = num_classes).to(DEVICE)

#print("Model initialized.")
#print(model)    # Print the model architecture

# Loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss()       # Loss function for multi-class classification

# Training loop
for epoch in range(EPOCHS):
    model.train()    # Set the model to training mode
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)   # Move images to the device (GPU or CPU)
        labels = labels.to(DEVICE)   # Move labels to the device

        optimizer.zero_grad()        # Zero the gradients
        outputs = model(images)      # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()             # Backward pass
        optimizer.step()            # Update the weights
        running_loss += loss.item()  # Accumulate the loss
    
    epoch_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}], Average Training Loss: {epoch_loss:.4f}") # Print the average loss for the epoch

# evaluate the model
print("Evaluating the model...")

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation for evaluation
    for images, labels in test_loader:
        images = images.to(DEVICE)  # Move images to the device
        labels = labels.to(DEVICE)  # Move labels to the device

        outputs = model(images)      # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
        total += labels.size(0)     # Total number of samples
        correct += (predicted == labels).sum().item()  # Count correct predictions

accuracy = 100 * correct / total  # Calculate accuracy
print(f"Accuracy of the model on the test set: {accuracy:.2f}%")  # Print accuracy

# saving the model
torch.save(model.state_dict(), MODEL_PATH)  # Save the model state dictionary
print(f"Model saved to {MODEL_PATH}")  # Print confirmation of model saving




print("End.")