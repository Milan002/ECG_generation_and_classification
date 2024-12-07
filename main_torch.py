import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
from sklearn.metrics import accuracy_score
from tqdm import tqdm




class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=21, stride=1, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=23, stride=1, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding='same')
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=27, stride=1, padding='same')
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=1, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(50, 128)  # Adjust the input features to match LSTM output
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 5)  # Number of classes = 5

    def forward(self, x):
        x = x.reshape(-1, 1, 187)  # Assuming input length is 300
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        
        # Reshape for LSTM Layer: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # Rearrange batch to be (batch_size, seq_len, features)
        
        # LSTM layers
        x, (h_n, c_n) = self.lstm(x)
        x = x[:, -1, :]  # Taking only the last output of LSTM
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

