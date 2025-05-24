# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torch.nn.functional as F

import numpy as np

from riemannML.exact.hybrid_eos import hybrid_eos

from .limiters import minmod

class HLLESolver(nn.Module):
    """
    HLLESolver is a PyTorch module that implements the HLLE (Harten-Lax-van Leer-Einfeldt) 
    approximate Riemann solver for hyperbolic conservation laws. It computes the state 
    and fluxes based on the input primitive variables, conserved variables, fluxes, 
    and wave speeds.
    Methods
    -------
    get_state(P, U, F, cmax, cmin):
        Computes the HLLE state based on the input variables.
        Parameters:
        - P (torch.Tensor): Placeholder for primitive variables (not used in the current implementation).
        - U (torch.Tensor): Conserved variables with shape [ncells, nvars, 2].
        - F (torch.Tensor): Fluxes with shape [ncells, nvars, 2].
        - cmax (torch.Tensor): Maximum wave speeds with shape [ncells].
        - cmin (torch.Tensor): Minimum wave speeds with shape [ncells].
        Returns:
        - torch.Tensor: HLLE state with shape [ncells, nvars].
    forward(P, U, F, cmax, cmin):
        Computes the HLLE fluxes based on the input variables.
        Parameters:
        - P (torch.Tensor): Placeholder for primitive variables (not used in the current implementation).
        - U (torch.Tensor): Conserved variables with shape [ncells, nvars, 2].
        - F (torch.Tensor): Fluxes with shape [ncells, nvars, 2].
        - cmax (torch.Tensor): Maximum wave speeds with shape [ncells].
        - cmin (torch.Tensor): Minimum wave speeds with shape [ncells].
        Returns:
        - torch.Tensor: HLLE fluxes with shape [ncells, nvars].
    """
    
    def __init__(self):
        super().__init__()
    
    def get_state(self, P, U, F, cmax, cmin):
        # U: [ncells, nvars, 2]
        # F: [ncells, nvars, 2]
        # cmax: [ncells]
        # cmin: [ncells]
        # 
        # Output: [ncells, nvars]
        
        ncells, nvars, _ = U.shape
        
        u_hlle = torch.zeros((ncells,nvars), device=U.device, dtype=U.dtype)
        
        u_hlle = torch.where( cmin[:,None]>=0 , U[:,:,0], u_hlle)
        u_hlle = torch.where( cmax[:,None]<=0 , U[:,:,1], u_hlle)
        u_hlle = torch.where( (cmin[:,None]<=0) & (cmax[:,None]>=0) , (cmax[:,None] * U[:,:,1] - cmin[:,None] * U[:,:,0] + F[:,:,0] - F[:,:,1]) / (cmax[:,None] - cmin[:,None]), u_hlle)
        
        return u_hlle
    
    def forward(self, P, U, F, cmax, cmin):
        # U: [ncells, nvars, 2]
        # F: [ncells, nvars, 2]
        # cmax: [ncells]
        # cmin: [ncells]
        # 
        # Output: [nvars, ncells]
        
        ncells, nvars, _ = U.shape
        
        f_hlle = torch.zeros((ncells,nvars), device=U.device, dtype=U.dtype)
        
        f_hlle = torch.where( cmin[:,None]>=0 , F[:,:,0], f_hlle)
        f_hlle = torch.where( cmax[:,None]<=0 , F[:,:,1], f_hlle)
        f_hlle = torch.where( (cmin[:,None]<=0) & (cmax[:,None]>=0) , (cmax[:,None] * F[:,:,0] - cmin[:,None] * F[:,:,1] + cmax[:,None] * cmin[:,None] * (U[:,:,1] - U[:,:,0]) ) / (cmax[:,None] - cmin[:,None]), f_hlle)
        
        # Compute HLLE fluxes
        return f_hlle 