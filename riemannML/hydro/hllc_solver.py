# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import numpy as np

from .hlle_solver import HLLESolver

class HLLCSolver(nn.Module):
    """
    HLLCSolver is a PyTorch module that implements the HLLC (Harten-Lax-van Leer-Contact) 
    approximate Riemann solver for solving hyperbolic conservation laws. It computes the 
    fluxes across cell interfaces in a numerical simulation.
    Methods:
    --------
    __init__():
        Initializes the HLLCSolver module.
    _contact_speed(U, F):
        Computes the contact wave speed (lambdaC) based on the input states and fluxes.
        Parameters:
        -----------
        U : torch.Tensor
            State tensor of shape [ncells, nvars, 2].
        F : torch.Tensor
            Flux tensor of shape [ncells, nvars, 2].
        Returns:
        --------
        torch.Tensor
            Contact wave speed tensor of shape [ncells].
    forward(P, U, F, cmax, cmin):
        Computes the HLLC fluxes based on the input primitive variables, states, fluxes, 
        and wave speeds.
        Parameters:
        -----------
        P : torch.Tensor
            Primitive variables tensor of shape [ncells, nvars, 2].
        U : torch.Tensor
            State tensor of shape [ncells, nvars, 2].
        F : torch.Tensor
            Flux tensor of shape [ncells, nvars, 2].
        cmax : torch.Tensor
            Maximum wave speed tensor of shape [ncells].
        cmin : torch.Tensor
            Minimum wave speed tensor of shape [ncells].
        Returns:
        --------
        torch.Tensor
            HLLC flux tensor of shape [ncells, nvars].
    """
    
    def __init__(self):
        super().__init__()
        
    def _contact_speed(self, U, F):
        a = (F[:,0] + F[:,1]).double()
        b = (- (U[:,1] + U[:,0] + F[:,2])).double()
        c = U[:,2].double()
        
        det = torch.sqrt(
            torch.clamp(b**2 - 4*a*c, min=0)
        )
        denom = torch.where(
            torch.abs(a) < 1e-45, 
            1e-45,  # Preserve sign of 'a'
            a
        )
        return (-0.5 * (b + det) / denom ).to(U.dtype)
        
    def forward(self, P, U, F, cmax, cmin):
        # U: [ncells, nvars, 2]
        # F: [ncells, nvars, 2]
        # cmax: [ncells]
        # cmin: [ncells]
        # 
        # Output: [nvars, ncells]
        
        hlle = HLLESolver() 
        
        f_hlle = hlle(P, U, F, cmax, cmin)
        u_hlle = hlle.get_state(P,U,F,cmax,cmin)
        
        lambdaC = self._contact_speed(u_hlle, f_hlle)

        pressC = - lambdaC * (f_hlle[:,1] + f_hlle[:,0]) + f_hlle[:,2]
        
        densCL =  U[:,0,0] * (  cmin - P[:,2,0]) / (  cmin - lambdaC )
        densCR =  U[:,0,1] * (  cmax - P[:,2,1]) / (  cmax - lambdaC )
        
        momCL = ( U[:,2,0] * (  cmin - P[:,2,0]) + (pressC-P[:,1,0]) ) / (  cmin - lambdaC )
        momCR = ( U[:,2,1] * (  cmax - P[:,2,1]) + (pressC-P[:,1,1]) ) / (  cmax - lambdaC )
        
        tauCL = ( (U[:,1,0]+U[:,0,0]) * (  cmin - P[:,2,0]) + pressC * lambdaC - P[:,1,0] * P[:,2,0] ) / (  cmin - lambdaC ) - densCL
        tauCR = ( (U[:,1,1]+U[:,0,1]) * (  cmax - P[:,2,1]) + pressC * lambdaC - P[:,1,1] * P[:,2,1] ) / (  cmax - lambdaC ) - densCR 
        
        uCL = torch.stack([densCL, tauCL, momCL], dim=1)
        uCR = torch.stack([densCR, tauCR, momCR], dim=1)
        
        f_hllc = torch.zeros_like(f_hlle, device=f_hlle.device, dtype=f_hlle.dtype)
        
        # Compute masks just once
        mask_L = ( cmin >= 0).unsqueeze(1)
        mask_CL = (( cmin < 0) & (lambdaC > 0)).unsqueeze(1)
        mask_CR = ((lambdaC <= 0) & (cmax > 0)).unsqueeze(1)
        mask_R = (cmax <= 0).unsqueeze(1)

        # Efficient flux selection using precomputed masks
        f_hllc = torch.where(mask_L, F[:, :, 0], f_hllc)
        f_hllc = torch.where(mask_CL, F[:, :, 0] + cmin[:, None] * (uCL - U[:, :, 0]), f_hllc)
        f_hllc = torch.where(mask_CR, F[:, :, 1] + cmax[:, None] * (uCR - U[:, :, 1]), f_hllc)
        f_hllc = torch.where(mask_R, F[:, :, 1], f_hllc)
        
        # Compute HLLC fluxes
        return f_hllc