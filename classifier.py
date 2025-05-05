# classifier.py


import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from PIL import Image
from torchvision import io
import torch.nn.functional as F


# Constants and Hyperparameters
DATA_DIR = "."          # Directory where the dataset is stored
MODEL_PATH = "fashion_mnist_ann.pth"   
BATCH_SIZE = 64                         
EPOCHS = 15
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


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):

    print("Training the model...")
    log_file = open("log.txt", "w")  # Open a log file to save training details
    # log_file.write("Epoch, Average Training Loss\n")  # Write header to the log file

    model.train()  # Set the model to training mode
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
        
        epoch_loss = running_loss / len(train_loader)  # Average loss for the epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Training Loss: {epoch_loss:.4f}") # Print the average loss for the epoch
        log_file.write(f"Epoch {epoch+1},  Average Training Loss: {epoch_loss:.4f}\n")  # Write epoch and loss to the log file

    print("Training completed.")  # Print confirmation of training completion
    log_file.close()  # Close the log file
    print("Log file saved as log.txt")  # Print confirmation of log file saving

# evaluate the model
def evaluate_model(model, test_loader, device):
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

    # append accuracy to the log file
    with open("log.txt", "a") as log_file:
        log_file.write(f"Test Accuracy: {accuracy:.2f}%\n")

    return accuracy  # Return accuracy

"""
# saving the model
torch.save(model.state_dict(), MODEL_PATH)  # Save the model state dictionary
print(f"Model saved to {MODEL_PATH}")  # Print confirmation of model saving

# loading the model
loaded_model = fashionANN(input_size=input_size, hidden_size=hidden_size, output_size=num_classes)  # Initialize a new model instance
# Then load the saved weights and biases (the state dictionary)
loaded_model.load_state_dict(torch.load(MODEL_PATH))
loaded_model.to(DEVICE) # Move the loaded model to the correct device
loaded_model.eval() # Set it to evaluation mode immediately after loading
print(f"Model loaded from {MODEL_PATH}")
"""

def predict_image(image_path, model, transform, device, class_names):

    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    img_tensor = transform(img) # Apply the same transformations as during training
    img_batch = img_tensor.unsqueeze(0).to(device)

    model.eval()    # Set the model to evaluation mode
    # Perform inference
    with torch.no_grad():
        output = model(img_batch)   # Forward pass
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_index = torch.argmax(probabilities, dim=1).item() # Get the index of the predicted class

    return class_names[predicted_index] # Return the predicted class name

if __name__ == "__main__":
    model = fashionANN().to(DEVICE)  # Initialize the model

    if os.path.exists(MODEL_PATH): # Check if the model file exists
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Load the model state dictionary
    else:
        print("No pre-trained model found. Training a new model...")
        # Train the model if no pre-trained model is found
        train_dataset = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        #train
        train_model(model, train_loader, criterion, optimizer, EPOCHS, DEVICE)

        # evaluate
        evaluate_model(model, test_loader, DEVICE)

        # Save the trained model
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    print("\n--- Interactive Classifier ---")
    while True:
        image_path = input("Enter the path to the image (or 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break
        predicted_class = predict_image(image_path, model, transform, DEVICE, CLASS_NAMES)
        if predicted_class is not None:
            print(f"Predicted class: {predicted_class}")




print("End.")