"""
    Summary: 
"""
# Importing necessary libraries
import pandas as pd                 # For handling data in tabular format
import numpy as np                  # For numerical operations
import matplotlib.pyplot as plt     # For data visualization

# Importing scikit-learn modules
from sklearn.model_selection import train_test_split   # For splitting data into training and testing sets
from sklearn.datasets import load_iris                  # For loading the Iris dataset

# Importing PyTorch modules
import torch                       # PyTorch library
import torch.nn as nn              # Neural network module
import torch.optim as optim        # Optimization algorithms
from torch.utils.data import DataLoader, TensorDataset  # For creating custom datasets and data loaders