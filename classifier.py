# classifier.py


import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np

DATA_DIR = "."
MODEL_PATH = "fashion_mnist_ann.pth"
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']