# Numpy and matplotlib
import numpy as np 
import matplotlib.pyplot as plt 

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

# PySINDy
import pysindy as ps


import torch.nn.init as init

from .weight_initialization import * 

class WaveStructureClassifier(nn.Module):
    ''' 
    This net decides, given left and right states of a SRHD problem, what the Riemann fan looks like:
    It will classify the initial state into one of three categories:
        - 2 shocks -> Label 0
        - 2 rarefactions -> Label 1 
        - L rarefaction and R shock -> Label 2
    '''
    def __init__(self, input_max, input_min, d_ff=32, d_conv=16):
        super(WaveStructureClassifier, self).__init__()

        self.activation = nn.ReLU()
        input_dim = 3  # Three primitive variables: log(rho), log(P), v
        output_dim = 3  # Possible wave labels
        self.input_max = input_max 
        self.input_min = input_min 

        # Convolutional layer to extract L/R differences
        self.conv = nn.Conv2d(in_channels=1, out_channels=d_conv, kernel_size=(3,2), stride=1, padding=0)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(6 + d_conv, 2 * d_ff)
        self.fc2 = nn.Linear(2*d_ff, d_ff)
        self.fc3 = nn.Linear(d_ff, output_dim)

        self.dropout = nn.Dropout(0.2)
        self.batchnorm1 = nn.BatchNorm1d(d_ff)
        self.batchnorm2 = nn.BatchNorm1d(2*d_ff)
        
        # Initialize weights
        self.apply(self.initialize_weights)
        
    def initialize_weights(self, m):
        """ Xavier initialization for fully connected layers """
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        Input shape: (batch_size, num_primitives, 2) -> (batch_size, 3, 2)
        """
        
        # Min-max normalization centered around zero: Scale to [-1,1]
        x = 2 * (x - self.input_min[None, :, None]) / (self.input_max[None, :, None] - self.input_min[None, :, None]) - 1.
        
        x_res = x.view(x.size(0), -1)
        
        # Apply convolution across the L/R axis
        x = F.relu(self.conv(x.unsqueeze(1)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat((x,x_res), dim=1) 
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        return self.fc3(x)  # Shape: (batch_size, 3) -> Unnormalized scores for classification
