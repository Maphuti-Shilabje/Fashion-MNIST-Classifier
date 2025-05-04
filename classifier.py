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



print("End.")