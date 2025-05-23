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


import torch
import torch.nn as nn
import torch.nn.functional as F

class Compactification:
    """
    Handles mapping between physical pressure and normalized [0,1] coordinate,
    depending on wave pattern.
    """
    def __init__(self, method='affine', a=None, b=None, L=1.0):
        self.method = method
        self.a = a  # lower bound
        self.b = b  # upper bound (if known)
        self.L = L  # scale for compactification

    def x_to_xi(self, x):
        if self.method == 'affine':
            return (x - self.a) / (self.b - self.a)
        elif self.method == 'compact':
            return (x - self.a) / (x - self.a + self.L)
        else:
            raise ValueError("Unknown method")

    def xi_to_x(self, xi):
        if self.method == 'affine':
            return self.a + (self.b - self.a) * xi
        elif self.method == 'compact':
            return self.a + self.L *  xi / (1.0-xi)
        else:
            raise ValueError("Unknown method")

def normalize_theta(theta, min_vals, max_vals):
    return 2 * (theta - min_vals[None, :, None]) / (max_vals[None, :, None]-min_vals[None,:,None]) - 1

class RootfindMLP(nn.Module):
    """
    A PyTorch module for a feedforward neural network with input normalization and customizable output mapping.
    This class implements a multi-layer perceptron (MLP) with configurable depth, width, and activation functions.
    It includes input normalization, weight initialization, and a learnable scaling parameter for the output.
    The activation function is SiLU (sigmoid linear unit).
    Attributes:
        net (nn.Sequential): The sequential container for the MLP layers.
        input_max (torch.Tensor): The maximum values for input normalization.
        input_min (torch.Tensor): The minimum values for input normalization.
        beta (nn.Parameter): A learnable scaling parameter applied to the output.
        output_mapping (callable): A function to map the output of the network.
    Args:
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The dimensionality of the output features.
        output_mapping (callable): A function to map the output of the network.
        input_max (torch.Tensor): The maximum values for input normalization.
        input_min (torch.Tensor): The minimum values for input normalization.
        d_ff (int, optional): The number of units in the hidden layers. Default is 32.
        depth (int, optional): The number of hidden layers. Default is 2.
    Methods:
        forward(x):
            Performs a forward pass through the network. Normalizes the input, processes it through the MLP,
            and applies the output mapping.
            Args:
                x (torch.Tensor): The input tensor. Can be 2D (batch_size x input_dim) or 3D 
                                  (batch_size x input_dim x sequence_length).
            Returns:
                torch.Tensor: The processed output tensor after applying the output mapping.
    """
    
    def __init__(self, input_dim, output_dim, output_mapping, input_max, input_min, d_ff=32, depth=2):
        super().__init__() 
        
        # Create layers
        layers = [nn.Linear(input_dim, d_ff), nn.SiLU()] 
        for _ in range(depth):
            layers.append(nn.Linear(d_ff,d_ff))
            layers.append(nn.SiLU()) 
        layers.append(nn.Linear(d_ff, output_dim))
        
        # Forward net
        self.net = nn.Sequential(*layers)
        
        # Store input normalizations
        self.input_max = input_max 
        self.input_min = input_min
        
        # Initialize weights
        self.apply(initialize_weights)
        
        # Output scaling 
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        self.output_mapping = output_mapping
        
    def forward(self, x):
        if x.dim() == 3:
            x_norm = 2 * (x - self.input_min[None, :, None]) / (self.input_max[None, :, None]-self.input_min[None, :, None]) - 1
            x_norm = x_norm.flatten(start_dim=1)
        else:
            x_norm = 2 * (x - self.input_min[None, :]) / (self.input_max[None, :]-self.input_min[None,:]) - 1
        y = self.net(x_norm) 
        return self.output_mapping(self.beta*y)
