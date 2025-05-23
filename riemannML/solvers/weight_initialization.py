# PyTorch
import torch
import torch.nn as nn
import torch.nn.init as init


def initialize_weights(m):
    """ Apply He Initialization to convolutional and linear layers. """
    if isinstance(m, nn.Conv1d):  # Convolutional layers
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)  # Set bias to zero

    elif isinstance(m, nn.Linear):  # Fully connected layers
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)  # Bias = 0 for stability

def initialize_weights_xavier(m):
    """ Apply Xavier Initialization to linear layers. """

    if isinstance(m, nn.Linear):  # Fully connected layers
        init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        if m.bias is not None:
            init.zeros_(m.bias)  # Bias = 0 for stability
            
