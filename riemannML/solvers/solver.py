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

import yaml

import numpy as np
import matplotlib.pyplot as plt

# PySINDy
import pysindy as ps


import torch.nn.init as init

from hydro.hlle_solver import HLLESolver
from .weight_initialization import * 

from .MLP import  Compactification, RootfindMLP
from .ensemble_solvers import * 

from riemannML.exact.riemann_solver import get_vel_shock, get_vel_raref, get_csnd, classify_wave_pattern, get_h
from riemannML.exact.hybrid_eos import hybrid_eos
import os

class RarefSolver(nn.Module):
    """
    RarefSolver is a PyTorch module designed to compute the rarefaction wave solution 
    in a Riemann problem using neural networks for predicting certain physical quantities.
    Attributes:
        net (dict): A dictionary containing the neural networks for the left (+1) 
                    and right (-1) going rarefaction waves.
        gamma (float): The adiabatic index (ratio of specific heats) from the equation of state (EOS).
    Methods:
        __init__(left_net, right_net, eos):
            Initializes the RarefSolver with the provided neural networks and EOS.
        forward(P, sign):
            Computes the density, pressure, specific enthalpy, and velocity at xi=0 inside
            the rarefaction wave given the input primitive variables and direction.
    Args:
        left_net (nn.Module): Neural network for the left state.
        right_net (nn.Module): Neural network for the right state.
        eos (object): An object containing the equation of state (EOS) properties, 
                      specifically `gamma_th` (adiabatic index).
    Forward Args:
        P (torch.Tensor): A tensor of shape (batch_size, 3) containing the primitive variables:
                          - P[:, 0]: Density 
                          - P[:, 1]: Pressure 
                          - P[:, 2]: Velocity 
        sign (int): Direction of the wave, either +1 (left) or -1 (right).
    Forward Returns:
        tuple: A tuple containing:
            - rho (torch.Tensor): Computed density inside the rarefaction wave.
            - p (torch.Tensor): Computed pressure inside the rarefaction wave.
            - h (torch.Tensor): Computed specific enthalpy inside the rarefaction wave.
            - vel (torch.Tensor): Computed velocity inside the rarefaction wave.
    """
    
    def __init__(self, left_net, right_net, eos):
        super().__init__() 
        self.net = {
            +1: left_net, 
            -1: right_net
        }
        self.gamma = eos.gamma_th 
        
    def forward(self,P, sign):
        batch, _ = P.shape 
        device   = P.device 
        dtype    = P.dtype 
        
        rhoa   = P[:,0]
        pa     = P[:,1]
        vela   = P[:,2]
        csa    = get_csnd(rhoa,pa,self.gamma)
                
        mapping = Compactification(method="affine", 
                                   a=torch.zeros(batch, device=device, dtype=dtype),
                                   b=csa)
        # Compute the pressure from the MLP 
        prims = torch.stack(
            (torch.log(P[:,0]),
            torch.log(P[:,1]),
            P[:,2]), dim=1 
        )
        xi_pred = self.net[sign](prims).squeeze(-1)
        cs_pred = mapping.xi_to_x(xi_pred)
        
        vel = sign * cs_pred 
        rho = rhoa * ((cs_pred**2*(self.gamma-1-csa**2))/(csa**2*(self.gamma-1-cs_pred**2)))**(1/(self.gamma-1))
        p   = cs_pred**2 * (self.gamma-1) * rho / (self.gamma - 1 - cs_pred**2) / self.gamma 
        
        h = get_h(rho,p,self.gamma)
        
        return rho, p, h, vel
        

class DShockSolver(nn.Module):
    """
    DShockSolver is a PyTorch module that computes the fluxes for a relativistic 
    hydrodynamics problem using a deep learning-based approach. It combines 
    machine learning predictions with physical equations to solve the Riemann 
    problem for the double shock wave pattern.
    Attributes:
        pressnet (nn.Module): A neural network model used to predict the pressure.
        gamma (float): The adiabatic index (ratio of specific heats) from the equation of state (EOS).
        eos (object): An equation of state (EOS) object that provides thermodynamic properties.
    Methods:
        forward(P, U, F, cmax, cmin):
            Computes the fluxes across cell interfaces based on the input primitive 
            variables, conservative variables, and fluxes.
            Args:
                P (torch.Tensor): A tensor of primitive variables with shape 
                    (ncells, 3, 2), where the last dimension represents left and 
                    right states. The variables are density, pressure, and velocity.
                U (torch.Tensor): A tensor of conservative variables (not used in 
                    the current implementation).
                F (torch.Tensor): A tensor of fluxes with shape (ncells, nvars, 2).
                cmax (torch.Tensor): Maximum wave speed (not used in the current implementation).
                cmin (torch.Tensor): Minimum wave speed (not used in the current implementation).
            Returns:
                torch.Tensor: A tensor of computed fluxes with shape (ncells, nvars).
    """
    
    def __init__(self, pressnet, eos):
        super().__init__()
        self.pressnet = pressnet 
        self.gamma = eos.gamma_th 
        self.eos   = eos 
        
    def forward(self,P,U,F,cmax,cmin):
        # Extract dimensions
        ncells, nvars, _ = F.shape 
        
        # Extract left/right prims
        rhoL = (P[:,0,0])
        rhoR = (P[:,0,1])
        pL   = (P[:,1,0])
        pR   = (P[:,1,1])
        vL   = (P[:,2,0])
        vR   = (P[:,2,1])
        
        # Compute the pressure from the MLP 
        prims = torch.stack(
            (torch.log(P[:,0,:]),
            torch.log(P[:,1,:]),
            P[:,2,:]), dim=1 
        )
        
        mapping = Compactification(method='compact', a=torch.max(pL,pR))
        
        pressC = mapping.xi_to_x(self.pressnet(prims).squeeze(-1))
        
        # Get the rest of the state 
        rhoCL, _, hCL, _, vstarL, vshockL = get_vel_shock(pressC,rhoL, pL, vL, -1, self.gamma)
        rhoCR, _, hCR, _, vstarR, vshockR = get_vel_shock(pressC,rhoR, pR, vR, +1, self.gamma)
        
        # Contact speed 
        lambdaC = 0.5 * (vstarR + vstarL)
        
        # CL and CR states
        WC = 1/torch.sqrt(1-lambdaC**2)
        
        densCL  = WC * rhoCL 
        densCR  = WC * rhoCR 
        
        tauCL   = densCL * ( WC * hCL - 1 ) * lambdaC 
        tauCR   = densCR * ( WC * hCR - 1 ) * lambdaC 
        
        momCL   = WC**2 * rhoCL * hCL * lambdaC 
        momCR   = WC**2 * rhoCR * hCR * lambdaC 
        
        uCL     = torch.stack([densCL, tauCL, momCL], dim=1)
        uCR     = torch.stack([densCR, tauCR, momCR], dim=1)
        
        FCL     = torch.zeros_like(uCL, device=F.device, dtype=F.dtype)
        FCR     = torch.zeros_like(uCR, device=F.device, dtype=F.dtype)
        
        FCL[:,0] = uCL[:,0] * lambdaC 
        FCR[:,0] = uCR[:,0] * lambdaC 
        
        FCL[:,1] = uCL[:,0] * (WC * hCL - 1) * lambdaC 
        FCR[:,1] = uCR[:,0] * (WC * hCR - 1) * lambdaC 
        
        FCL[:,2] = uCL[:,2] * lambdaC + pressC  
        FCR[:,2] = uCR[:,2] * lambdaC + pressC 
        
        # masks 
        mask_L   = (vshockL >= 0).unsqueeze(1)
        mask_CL  = ((vshockL<0)  & (lambdaC>0)).unsqueeze(1)
        mask_CR  = ((lambdaC<=0) & (vshockR>0)).unsqueeze(1)
        mask_R   = ((vshockR<=0)).unsqueeze(1)
        
        # Compute fluxes 
        fluxes  = torch.zeros(ncells,nvars, device=F.device, dtype=F.dtype)
        fluxes = torch.where(mask_L , F[:, :, 0], fluxes)
        fluxes = torch.where(mask_CL, FCL, fluxes)
        fluxes = torch.where(mask_CR, FCR, fluxes)
        fluxes = torch.where(mask_R , F[:, :, 1], fluxes)
        
        # Compute HLLC fluxes
        return fluxes
        
class DRarefSolver(nn.Module):
    """
    DRarefSolver is a PyTorch module that computes the fluxes for a relativistic 
    hydrodynamics problem using a deep learning-based approach. It combines 
    machine learning predictions with physical equations to solve the Riemann 
    problem for the double rarefaction wave pattern.
    Attributes:
        pressnet (nn.Module): A neural network model used to predict the pressure 
            at the contact discontinuity.
        raref_solver (callable): A solver function for computing the rarefaction 
            wave states.
        gamma (float): The adiabatic index (ratio of specific heats) from the 
            equation of state (EOS).
        eos (object): An equation of state object that provides thermodynamic 
            properties.
    Methods:
        forward(P, U, F, cmax, cmin):
            Computes the fluxes for the given input states and fluxes.
            Args:
                P (torch.Tensor): A tensor of primitive variables with shape 
                    (ncells, 3, 2), where the last dimension represents left 
                    and right states.
                U (torch.Tensor): A tensor of conserved variables (not directly 
                    used in this implementation).
                F (torch.Tensor): A tensor of fluxes with shape (ncells, nvars, 2).
                cmax (torch.Tensor): Maximum signal speed (not directly used in 
                    this implementation).
                cmin (torch.Tensor): Minimum signal speed (not directly used in 
                    this implementation).
            Returns:
                torch.Tensor: A tensor of computed fluxes with shape 
                (ncells, nvars).
    """
    
    def __init__(self, pressnet, raref_solver, eos):
        super().__init__()
        self.pressnet = pressnet 
        self.raref_solver = raref_solver
        self.gamma = eos.gamma_th 
        self.eos   = eos 
        
    def forward(self,P,U,F,cmax,cmin):
        # Extract dimensions
        ncells, nvars, _ = F.shape 
        
        # Extract left/right prims
        rhoL = (P[:,0,0])
        rhoR = (P[:,0,1])
        pL   = (P[:,1,0])
        pR   = (P[:,1,1])
        vL   = (P[:,2,0])
        vR   = (P[:,2,1])
        csL  = get_csnd(rhoL,pL,self.gamma)
        csR  = get_csnd(rhoR,pR,self.gamma)
        # Compute the pressure from the MLP 
        prims = torch.stack(
            (torch.log(P[:,0,:]),
            torch.log(P[:,1,:]),
            P[:,2,:]), dim=1 
        )
        mapping = Compactification(method='affine', a=torch.zeros_like(pL,device=P.device,dtype=P.dtype), b=torch.min(pL,pR))
        
        pressC = mapping.xi_to_x(self.pressnet(prims).squeeze(-1))
        
        # Get the rest of the state 
        rhoCL, _, hCL, csCL, vstarL, _ = get_vel_raref(pressC,rhoL, pL, vL, -1, self.gamma)
        rhoCR, _, hCR, csCR, vstarR, _ = get_vel_raref(pressC,rhoR, pR, vR, +1, self.gamma)
        
        # Contact speed 
        lambdaC = 0.5 * (vstarR + vstarL)
        
        # Rarefaction speed 
        lambdaRL  = (lambdaC - csCL)/(1-lambdaC*csCL)
        lambdaL   = (vL - csL) / (1-vL*csL) 
        lambdaRR  = (lambdaC + csCR)/(1+lambdaC*csCR)
        lambdaR  = (vR + csR)/(1+vR*csR)
        
        # CL and CR states
        WC = 1/torch.sqrt(1-lambdaC**2)
        
        densCL  = WC * rhoCL 
        densCR  = WC * rhoCR 
        
        tauCL   = densCL * ( WC * hCL - 1 ) * lambdaC 
        tauCR   = densCR * ( WC * hCR - 1 ) * lambdaC 
        
        momCL   = WC**2 * rhoCL * hCL * lambdaC 
        momCR   = WC**2 * rhoCR * hCR * lambdaC 
        
        uCL     = torch.stack([densCL, tauCL, momCL], dim=1)
        uCR     = torch.stack([densCR, tauCR, momCR], dim=1)
        
        FCL     = torch.zeros_like(uCL, device=F.device, dtype=F.dtype)
        FCR     = torch.zeros_like(uCR, device=F.device, dtype=F.dtype)
        
        FCL[:,0] = uCL[:,0] * lambdaC 
        FCR[:,0] = uCR[:,0] * lambdaC 
        
        FCL[:,1] = uCL[:,0] * (WC * hCL - 1) * lambdaC 
        FCR[:,1] = uCR[:,0] * (WC * hCR - 1) * lambdaC 
        
        FCL[:,2] = uCL[:,2]* lambdaC + pressC  
        FCR[:,2] = uCR[:,2]* lambdaC + pressC 
        
        # RR and RL states 
        rho_RL, p_RL, h_RL, v_RL = self.raref_solver(P[:,:3,0], +1)
        rho_RR, p_RR, h_RR, v_RR = self.raref_solver(P[:,:3,1], -1)
        
        WRL = 1/torch.sqrt(1.-v_RL**2)
        WRR = 1/torch.sqrt(1.-v_RR**2)
        
        densRL = WRL * rho_RL 
        densRR = WRR * rho_RR 
        
        tauRL  = densRL * ( WRL * h_RL - 1 ) * lambdaRL 
        tauRR  = densRR * ( WRR * h_RR - 1 ) * lambdaRR 
        
        momRL  = WRL**2 * rho_RL * h_RL * v_RL 
        momRR  = WRR**2 * rho_RR * h_RR * v_RR 
        
        uRL     = torch.stack([densRL, tauRL, momRL], dim=1)
        uRR     = torch.stack([densRR, tauRR, momRR], dim=1)
        
        FRL     = torch.zeros_like(uCL, device=F.device, dtype=F.dtype)
        FRR     = torch.zeros_like(uCL, device=F.device, dtype=F.dtype)
        
        FRL[:,0] = uRL[:,0] * v_RL 
        FRR[:,0] = uRR[:,0] * v_RR 
        
        FRL[:,1] = uRL[:,0] * (WRL * h_RL - 1) * v_RL 
        FRR[:,1] = uRR[:,0] * (WRR * h_RR - 1) * v_RR 
        
        FRL[:,2] = uRL[:,2]* v_RL + p_RL  
        FRR[:,2] = uRR[:,2]* v_RR + p_RR  
        
        # masks 
        mask_L   = (lambdaL >= 0).unsqueeze(1)
        mask_RL  = ((lambdaL < 0) & (lambdaRL>=0)).unsqueeze(1)
        mask_CL  = ((lambdaRL<0)  & (lambdaC>0)).unsqueeze(1)
        mask_CR  = ((lambdaC<=0)  & (lambdaRR>0)).unsqueeze(1)
        mask_RR  = ((lambdaRR<=0) & (lambdaR>0)).unsqueeze(1)
        mask_R   = ((lambdaR<=0)).unsqueeze(1)
        
        # Compute fluxes 
        fluxes  = torch.zeros(ncells,nvars, device=F.device, dtype=F.dtype)
        fluxes = torch.where(mask_L , F[:, :, 0], fluxes)
        fluxes = torch.where(mask_RL, FRL, fluxes)
        fluxes = torch.where(mask_CL, FCL, fluxes)
        fluxes = torch.where(mask_CR, FCR, fluxes)
        fluxes = torch.where(mask_RR, FRR, fluxes)
        fluxes = torch.where(mask_R , F[:, :, 1], fluxes)
        
        # Compute HLLC fluxes
        return fluxes

class RarefShockSolver(nn.Module):
    """
    RarefShockSolver is a PyTorch module that computes the fluxes for a relativistic 
    hydrodynamics problem using a deep learning-based approach. It combines 
    machine learning predictions with physical equations to solve the Riemann 
    problem for the shock rarefaction wave pattern.
    Attributes:
        pressnet (nn.Module): A neural network model used to predict the contact pressure.
        raref_solver (callable): A solver for rarefaction waves.
        gamma (float): The adiabatic index (ratio of specific heats) from the equation of state (EOS).
        eos (object): An equation of state object that provides thermodynamic properties.
    Methods:
        forward(P, U, F, cmax, cmin):
            Computes the fluxes for the given input state variables and fluxes.
    Args:
        pressnet (nn.Module): A neural network model for predicting pressure.
        raref_solver (callable): A function or module to solve rarefaction waves.
        eos (object): An equation of state object containing thermodynamic properties.
    Forward Method:
        Args:
            P (torch.Tensor): A tensor of primitive variables with shape (ncells, nvars, 2).
                              Contains density, pressure, and velocity for left and right states.
            U (torch.Tensor): A tensor of conserved variables (not used in the current implementation).
            F (torch.Tensor): A tensor of fluxes with shape (ncells, nvars, 2).
            cmax (float): Maximum wave speed (not used in the current implementation).
            cmin (float): Minimum wave speed (not used in the current implementation).
        Returns:
            torch.Tensor: A tensor of computed fluxes with shape (ncells, nvars).
    """
    
    def __init__(self, pressnet, raref_solver, eos):
        super().__init__()
        self.pressnet = pressnet 
        self.raref_solver = raref_solver 
        self.gamma = eos.gamma_th 
        self.eos   = eos 
        
    def forward(self,P,U,F,cmax,cmin):
        # Extract dimensions
        ncells, nvars, _ = F.shape 
        
        # Extract left/right prims
        rhoL = (P[:,0,0])
        rhoR = (P[:,0,1])
        pL   = (P[:,1,0])
        pR   = (P[:,1,1])
        vL   = (P[:,2,0])
        vR   = (P[:,2,1])
        csL  = get_csnd(rhoL,pL,self.gamma)
        
        # Compute the pressure from the MLP 
        prims = torch.stack(
            (torch.log(P[:,0,:]),
            torch.log(P[:,1,:]),
            P[:,2,:]), dim=1 
        )
        mapping = Compactification(method='affine', a=pR, b=pL)
        pressC = mapping.xi_to_x(self.pressnet(prims).squeeze(-1))
        
        # Get the rest of the state 
        rhoCL, _, hCL, csCL, vstarL, _      = get_vel_raref(pressC,rhoL, pL, vL, -1, self.gamma)
        rhoCR, _, hCR, _   , vstarR, vshock = get_vel_shock(pressC,rhoR, pR, vR, +1, self.gamma)
        
        # Contact speed 
        lambdaC = 0.5 * (vstarR + vstarL)
        
        # Rarefaction speeds
        # Tail 
        lambdaRL = (lambdaC - csCL)/(1-lambdaC*csCL)
        # Head 
        lambdaL  = (vL - csL)/(1-vL*csL)
        
        # CL and CR states
        WC = 1/torch.sqrt(1-lambdaC**2)
        
        densCL  = WC * rhoCL 
        densCR  = WC * rhoCR 
        
        tauCL   = densCL * ( WC * hCL - 1 ) * lambdaC 
        tauCR   = densCR * ( WC * hCR - 1 ) * lambdaC 
        
        momCL   = WC**2 * rhoCL * hCL * lambdaC 
        momCR   = WC**2 * rhoCR * hCR * lambdaC 
        
        uCL     = torch.stack([densCL, tauCL, momCL], dim=1)
        uCR     = torch.stack([densCR, tauCR, momCR], dim=1)
        
        FCL     = torch.zeros_like(uCL, device=F.device, dtype=F.dtype)
        FCR     = torch.zeros_like(uCR, device=F.device, dtype=F.dtype)
        
        FCL[:,0] = uCL[:,0] * lambdaC 
        FCR[:,0] = uCR[:,0] * lambdaC 
        
        FCL[:,1] = uCL[:,0] * (WC * hCL - 1) * lambdaC 
        FCR[:,1] = uCR[:,0] * (WC * hCR - 1) * lambdaC 
        
        FCL[:,2] = uCL[:,2]* lambdaC + pressC  
        FCR[:,2] = uCR[:,2]* lambdaC + pressC  
        
        # Get fluxes inside rarefaction 
        rho_RL, p_RL, h_RL, v_RL = self.raref_solver(P[:,:3,0], +1)
        
        WRL = 1/torch.sqrt(1.-v_RL**2)
        
        densRL = WRL * rho_RL 
        
        tauRL  = densRL * ( WRL * h_RL - 1 ) * lambdaRL 
        
        momRL  = WRL**2 * rho_RL * h_RL * v_RL 
        
        uRL     = torch.stack([densRL, tauRL, momRL], dim=1)
        
        FRL     = torch.zeros_like(uCL, device=F.device, dtype=F.dtype)
        
        FRL[:,0] = uRL[:,0] * v_RL 
        
        FRL[:,1] = uRL[:,0] * (WRL * h_RL - 1) * v_RL 
        
        FRL[:,2] = uRL[:,2]* v_RL + p_RL  
        
        # masks 
        mask_L   = (lambdaL >= 0).unsqueeze(1)
        mask_RL  = ((lambdaL<0)  & (lambdaRL>=0)).unsqueeze(1)
        mask_CL  = ((lambdaRL<0) & (lambdaC>0)).unsqueeze(1)
        mask_CR  = ((lambdaC<=0) & (vshock>0)).unsqueeze(1)
        mask_R   = ((vshock<=0)).unsqueeze(1)
        
        # Compute fluxes 
        fluxes = torch.zeros(ncells,nvars, device=F.device, dtype=F.dtype)
        fluxes = torch.where(mask_L , F[:, :, 0], fluxes)
        fluxes = torch.where(mask_RL, FRL, fluxes)
        fluxes = torch.where(mask_CL, FCL, fluxes)
        fluxes = torch.where(mask_CR, FCR, fluxes)
        fluxes = torch.where(mask_R , F[:, :, 1], fluxes)

        # Compute HLLC fluxes
        return fluxes
         

class RiemannSolver(nn.Module):
    """
    RiemannSolver is a neural network-based solver for the Riemann problem in fluid dynamics. 
    It classifies wave patterns and computes fluxes for different cases such as double shocks, 
    double rarefactions, rarefaction-shock combinations, and continuous data.
    Attributes:
        double_shock_solver (callable): Solver for the double shock case.
        double_rarefaction_solver (callable): Solver for the double rarefaction case.
        rarefaction_shock_solver (callable): Solver for the rarefaction-shock case.
        continuity_cutoff (float): Threshold for identifying continuous data.
        gamma (float): Specific heat ratio for the fluid.
        masks (dict): Boolean masks for different wave patterns and classifications.
        fluxes (torch.Tensor): Tensor to store computed fluxes.
        remaining_mask (torch.Tensor): Mask to track unprocessed states.
        hlle_solver (HLLESolver): HLLE solver for continuous data.
    Methods:
        __init__(N, double_shock_solver, double_rarefaction_solver, rarefaction_shock_solver, 
            Initializes the RiemannSolver with the given parameters.
        _insert_fluxes(target_tensor, mask, remaining_mask, new_fluxes):
            Inserts computed fluxes into the target tensor based on masks.
        _zero_out_masks_and_fluxes():
            Resets all masks and fluxes to their initial state.
        forward(P, U, F, cmax, cmin):
            Computes the fluxes for the given primitive variables, conservative variables, 
            fluxes, and wave speeds. Handles different cases such as continuous data, 
            double shocks, double rarefactions, rarefaction-shock combinations, and low-confidence cases.
            Args:
                P (torch.Tensor): Primitive variables (density, pressure, velocity).
                U (torch.Tensor): Conservative variables.
                F (torch.Tensor): Fluxes.
                cmax (torch.Tensor): Maximum wave speeds.
                cmin (torch.Tensor): Minimum wave speeds.
            Returns:
                torch.Tensor: Computed fluxes for all states.
    """
    
    def __init__(self,
                 double_shock_solver, 
                 double_rarefaction_solver, 
                 rarefaction_shock_solver, 
                 continuity_cutoff, gamma, dtype, device):
        super(RiemannSolver,self).__init__()
        self.double_shock_solver    = double_shock_solver
        self.double_rarefaction_solver = double_rarefaction_solver
        self.rarefaction_shock_solver = rarefaction_shock_solver
        self.continuity_cutoff = continuity_cutoff 
        self.hlle_solver = HLLESolver() 
        self.N = None 
        self.masks = {
            "continuous": None,
            "low_conf": None,
            "double_shock": None,
            "double_rarefaction": None,
            "raref_shock": None,
        }
        self.gamma = gamma
        # Create fluxes
        self.fluxes = None
        
        # Create index mask 
        self.remaining_mask = None
        
        self.device = device 
        self.dtype  = dtype 

    
    def _insert_fluxes(self,target_tensor, mask, remaining_mask, new_fluxes):
        indices_in_full = remaining_mask.nonzero()[mask].squeeze()
        target_tensor[indices_in_full, :] = new_fluxes

    def _zero_out_masks_and_fluxes(self):
        self.fluxes[:, :] = 0
        self.remaining_mask[:] = True
        for key in self.masks:
            self.masks[key][:] = False  # Proper in-place reset

    def _allocate_buffers(self, N, nvars):
        device, dtype = self.device, self.dtype
        if N == self.N:
            self._zero_out_masks_and_fluxes()
            return 
        self.fluxes = torch.zeros(N, nvars, device=device, dtype=dtype)
        self.remaining_mask = torch.ones(N, device=device, dtype=torch.bool)
        self.masks = {
            "continuous": torch.zeros(N, dtype=torch.bool, device=device),
            "low_conf": torch.zeros(N, dtype=torch.bool, device=device),
            "double_shock": torch.zeros(N, dtype=torch.bool, device=device),
            "double_rarefaction": torch.zeros(N, dtype=torch.bool, device=device),
            "raref_shock": torch.zeros(N, dtype=torch.bool, device=device),
        }
        self.N = N 
    
    def forward(self, P, U, F, cmax, cmin):
        
        # Extract dimensions
        ncells, nvars, _ = F.shape
        dtype, device = F.dtype, F.device
        
        self._allocate_buffers(ncells, nvars)

        P,U,F = P.clone(), U.clone(), F.clone()
        
        fluxes = self.fluxes 
        remaining_mask = self.remaining_mask
        
        # Flip states where pR > pL to ensure pL >= pR
        flip_mask = (P[:, 1, 1] > P[:, 1, 0])  # Shape: (N,)

        # Flip all variables accordingly
        for i in range(3):  # over variables: rho, press, vel
            P[:, i, :] = torch.where(
                flip_mask.unsqueeze(1),
                torch.stack([
                    P[:, i, 1] * (-1 if i == 2 else 1), # new L  <- old R
                    P[:, i, 0] * (-1 if i == 2 else 1)  # new R  <- old L (flip v)
                ], dim=1),
                P[:, i, :]
            )

            U[:, i, :] = torch.where(
                flip_mask.unsqueeze(1),
                torch.stack([
                    U[:, i, 1] * (-1 if i == 2 else 1), 
                    U[:, i, 0] * (-1 if i == 2 else 1)
                ], dim=1),
                U[:, i, :]
            )

            F[:, i, :] = torch.where(
                flip_mask.unsqueeze(1),
                torch.stack([
                    F[:, i, 1] * (+1 if i==2 else -1), 
                    F[:, i, 0] * (+1 if i==2 else -1),
                ], dim=1),
                F[:, i, :]
            )

        # Flip wave speeds
        #cmin = torch.where(flip_mask, -cmin, cmin)
        #cmax = torch.where(flip_mask, -cmax, cmax)

        # ---------------------- STEP 1: HANDLE CONTINUOUS DATA ----------------------
        continuous_data_mask = torch.max((P[:,:,1] - P[:,:,0]).abs(), dim=1)[0] < self.continuity_cutoff
        if continuous_data_mask.any():
            fluxes_sc = self.hlle_solver(
                P[continuous_data_mask, :, :],
                U[continuous_data_mask, :, :],
                F[continuous_data_mask, :, :],
                cmax[continuous_data_mask], 
                cmin[continuous_data_mask]
            )
            
            # Insert computed fluxes in the tensor 
            self._insert_fluxes(fluxes, continuous_data_mask, remaining_mask, fluxes_sc)
            
            self.masks["continuous"][remaining_mask.clone()] = continuous_data_mask

            # Remove processed data *without copying* by using masks
            remaining_mask[remaining_mask.clone()] &= ~continuous_data_mask
            
            
            P, U, F, cmax, cmin = P[remaining_mask], U[remaining_mask], F[remaining_mask], cmax[remaining_mask], cmin[remaining_mask]
        
        # ---------------------- STEP 2: CLASSIFICATION (ONLY ON REMAINING STATES) ----------------------   
        prims = torch.stack(
            (torch.log(P[:,0,:]),
            torch.log(P[:,1,:]),
            P[:,2,:]), dim=1 
        )
        labels = classify_wave_pattern(prims,self.gamma)
        labels = torch.argmax(labels.to(torch.int64), dim=1)
        
        # ---------------------- STEP 3: HANDLE LOW CONFIDENCE CASES ----------------------
        vacuum_mask = labels == 3
        if vacuum_mask.any():
            fluxes_vac = torch.zeros((vacuum_mask.nonzero().size(0),3), device=P.device, dtype=P.dtype)

            # Insert computed fluxes in the tensor 
            self._insert_fluxes(fluxes, vacuum_mask, remaining_mask, fluxes_vac)


        # ---------------------- STEP 4: HANDLE DOUBLE SHOCK CASE ----------------------
        double_shock_mask = labels == 0
        if double_shock_mask.any():
            fluxes_ds = self.double_shock_solver(
                P[double_shock_mask, :, :],
                U[double_shock_mask, :, :],
                F[double_shock_mask, :, :],
                cmax[double_shock_mask], 
                cmin[double_shock_mask]
            )

            # Store for diagnostics
            self.masks["double_shock"][remaining_mask.clone()] = double_shock_mask
            
            # Insert computed fluxes in the tensor 
            self._insert_fluxes(fluxes, double_shock_mask, remaining_mask, fluxes_ds)


        # ---------------------- STEP 5: HANDLE DOUBLE RAREFACTION CASE ----------------------
        double_rarefaction_mask = labels == 1
        if double_rarefaction_mask.any():
            fluxes_dr = self.double_rarefaction_solver(
                P[double_rarefaction_mask, :, :],
                U[double_rarefaction_mask, :, :],
                F[double_rarefaction_mask, :, :],
                cmax[double_rarefaction_mask], 
                cmin[double_rarefaction_mask]
            )

            # Store for diagnostics
            self.masks["double_rarefaction"][remaining_mask.clone()] = double_rarefaction_mask
            
            # Insert computed fluxes in the tensor 
            self._insert_fluxes(fluxes, double_rarefaction_mask, remaining_mask, fluxes_dr)

        # ---------------------- STEP 5: HANDLE RAREFACTION-SHOCK CASE ----------------------
        rarefaction_shock_mask = labels == 2
        if rarefaction_shock_mask.any():
            fluxes_rs = self.rarefaction_shock_solver(
                P[rarefaction_shock_mask, :, :],
                U[rarefaction_shock_mask, :, :],
                F[rarefaction_shock_mask, :, :],
                cmax[rarefaction_shock_mask], 
                cmin[rarefaction_shock_mask]
            )
            
            # Store for diagnostics
            self.masks["raref_shock"][remaining_mask.clone()] = rarefaction_shock_mask
            
            # Insert computed fluxes in the tensor 
            self._insert_fluxes(fluxes, rarefaction_shock_mask, remaining_mask, fluxes_rs)

        
        # Flip the fluxes back where necessary 
        for i in range(3):  # over variables: rho, press, vel
            fluxes[:, i] = torch.where(
                flip_mask,
                fluxes[:,i] * (+1 if i==2 else -1),
                fluxes[:,i]
            )
            
        return fluxes


def construct_riemann_solver(basepath,device, gamma):
    """
    Constructs and initializes a Riemann solver using pre-trained machine learning models 
    for different components of the solver.
    Args:
        basepath (str): The base directory path where the pre-trained model files and 
                        input normalization tensors are stored.
        device (torch.device): The device (CPU or GPU) on which the models and computations 
                                will be executed.
        gamma (float): The adiabatic index (ratio of specific heats) used in the equation 
                        of state (EOS).
    Returns:
        RiemannSolver: An instance of the RiemannSolver class, initialized with the 
                        pre-trained models and configured for solving Riemann problems.
    Notes:
        - The function loads pre-trained models for different components of the Riemann 
            solver (e.g., shock, rarefaction, and rarefaction-shock solvers) from the 
            specified `basepath`.
        - Input normalization tensors (`input_min` and `input_max`) are also loaded 
            from the `basepath` to normalize the inputs to the models.
        - The function uses the `RootfindMLP` class to define the machine learning models 
            and loads their weights from the corresponding checkpoint files.
        - The `hybrid_eos` function is used to define the equation of state (EOS).
        - The constructed solver includes sub-solvers for shock, rarefaction, and 
            rarefaction-shock interactions, as well as a rarefaction solver for left and 
            right states.
    """
    
    dtype = torch.float64
    try:
        input_min = torch.load(os.path.join(basepath,'input_min.pt'), weights_only=True, map_location=device)
        input_max = torch.load(os.path.join(basepath,'input_max.pt'), weights_only=True, map_location=device)
    except:
        raise FileNotFoundError("Invalid directory passed to constructor, directory must contain max/min files!")
    
    with open(os.path.join(basepath,'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    n_neurons = config['n_neurons']
    n_layers  = config['n_layers']
    
    dshock_model = os.path.join(basepath, "d_shock", "checkpoints", "best_model.pt")
    dshock_mlp =  RootfindMLP( input_dim=6, output_dim=1, d_ff=n_neurons, depth=n_layers, input_max=input_max.to(dtype), input_min=input_min.to(dtype), output_mapping=F.sigmoid).to(device).to(dtype)
    dshock_mlp.load_state_dict(torch.load(dshock_model, weights_only=True,map_location=device))
    
    draref_model = os.path.join(basepath, "d_raref", "checkpoints", "best_model.pt")
    draref_mlp = RootfindMLP( input_dim=6, output_dim=1, d_ff=n_neurons, depth=n_layers, input_max=input_max.to(dtype), input_min=input_min.to(dtype), output_mapping=F.sigmoid).to(device).to(dtype)
    draref_mlp.load_state_dict(torch.load(draref_model, weights_only=True,map_location=device))
    
    rarefshock_model = os.path.join(basepath, "rarefshock", "checkpoints", "best_model.pt")
    rarefshock_mlp =  RootfindMLP( input_dim=6, output_dim=1, d_ff=n_neurons, depth=n_layers, input_max=input_max.to(dtype), input_min=input_min.to(dtype), output_mapping=F.sigmoid).to(device).to(dtype)
    rarefshock_mlp.load_state_dict(torch.load(rarefshock_model, weights_only=True,map_location=device))
    
    raref_left_model = os.path.join(basepath, "raref_left", "checkpoints", "best_model.pt")
    raref_left_mlp = RootfindMLP(input_dim=3,output_dim=1, output_mapping=torch.sigmoid, input_min=input_min, input_max=input_max, d_ff=n_neurons//2, depth=n_layers).to(device).to(dtype)
    raref_left_mlp.load_state_dict(torch.load(raref_left_model, weights_only=True,map_location=device))

    raref_right_model = os.path.join(basepath, "raref_right", "checkpoints", "best_model.pt")
    raref_right_mlp = RootfindMLP(input_dim=3,output_dim=1, output_mapping=torch.sigmoid, input_min=input_min, input_max=input_max, d_ff=n_neurons//2, depth=n_layers).to(device).to(dtype)
    raref_right_mlp.load_state_dict(torch.load(raref_right_model, weights_only=True,map_location=device))
    
    eos = hybrid_eos(0,2,gamma)
    raref_solver = RarefSolver(raref_left_mlp,raref_right_mlp,eos)

    dshock_solver = DShockSolver(dshock_mlp, eos)
    draref_solver = DRarefSolver(draref_mlp, raref_solver, eos)
    rarefshock_solver = RarefShockSolver(rarefshock_mlp, raref_solver, eos)
    
    riemann_solver = RiemannSolver(dshock_solver, draref_solver, rarefshock_solver, 1e-10, gamma, torch.float64, device)
    
    return riemann_solver


def construct_riemann_solver_ensemble(basepaths,device, gamma):
    """
    Constructs and initializes a Riemann solver using pre-trained machine learning models 
    for different components of the solver.
    Args:
        basepath (str): The base directory path where the pre-trained model files and 
                        input normalization tensors are stored.
        device (torch.device): The device (CPU or GPU) on which the models and computations 
                                will be executed.
        gamma (float): The adiabatic index (ratio of specific heats) used in the equation 
                        of state (EOS).
    Returns:
        RiemannSolver: An instance of the RiemannSolver class, initialized with the 
                        pre-trained models and configured for solving Riemann problems.
    Notes:
        - The function loads pre-trained models for different components of the Riemann 
            solver (e.g., shock, rarefaction, and rarefaction-shock solvers) from the 
            specified `basepath`.
        - Input normalization tensors (`input_min` and `input_max`) are also loaded 
            from the `basepath` to normalize the inputs to the models.
        - The function uses the `RootfindMLP` class to define the machine learning models 
            and loads their weights from the corresponding checkpoint files.
        - The `hybrid_eos` function is used to define the equation of state (EOS).
        - The constructed solver includes sub-solvers for shock, rarefaction, and 
            rarefaction-shock interactions, as well as a rarefaction solver for left and 
            right states.
    """
    
    dtype = torch.float64
    dshock_mlps = [] 
    draref_mlps = []
    rarefshock_mlps = []
    raref_left_mlps = []
    raref_right_mlps = []
    for basepath in basepaths:
        try:
            input_min = torch.load(os.path.join(basepath,'input_min.pt'), weights_only=True, map_location=device)
            input_max = torch.load(os.path.join(basepath,'input_max.pt'), weights_only=True, map_location=device)
        except:
            raise FileNotFoundError("Invalid directory passed to constructor, directory must contain max/min files!")
        
        with open(os.path.join(basepath,'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        
        n_neurons = config['n_neurons']
        n_layers  = config['n_layers']
        
        dshock_model = os.path.join(basepath, "d_shock", "checkpoints", "best_model.pt")
        dshock_mlp =  RootfindMLP( input_dim=6, output_dim=1, d_ff=n_neurons, depth=n_layers, input_max=input_max.to(dtype), input_min=input_min.to(dtype), output_mapping=F.sigmoid).to(device).to(dtype)
        dshock_mlp.load_state_dict(torch.load(dshock_model, weights_only=True,map_location=device))
        dshock_mlps.append(dshock_mlp)
        
        draref_model = os.path.join(basepath, "d_raref", "checkpoints", "best_model.pt")
        draref_mlp = RootfindMLP( input_dim=6, output_dim=1, d_ff=n_neurons, depth=n_layers, input_max=input_max.to(dtype), input_min=input_min.to(dtype), output_mapping=F.sigmoid).to(device).to(dtype)
        draref_mlp.load_state_dict(torch.load(draref_model, weights_only=True,map_location=device))
        draref_mlps.append(draref_mlp)
        
        rarefshock_model = os.path.join(basepath, "rarefshock", "checkpoints", "best_model.pt")
        rarefshock_mlp =  RootfindMLP( input_dim=6, output_dim=1, d_ff=n_neurons, depth=n_layers, input_max=input_max.to(dtype), input_min=input_min.to(dtype), output_mapping=F.sigmoid).to(device).to(dtype)
        rarefshock_mlp.load_state_dict(torch.load(rarefshock_model, weights_only=True,map_location=device))
        rarefshock_mlps.append(rarefshock_mlp)
        
        raref_left_model = os.path.join(basepath, "raref_left", "checkpoints", "best_model.pt")
        raref_left_mlp = RootfindMLP(input_dim=3,output_dim=1, output_mapping=torch.sigmoid, input_min=input_min, input_max=input_max, d_ff=n_neurons//2, depth=n_layers).to(device).to(dtype)
        raref_left_mlp.load_state_dict(torch.load(raref_left_model, weights_only=True,map_location=device))
        raref_left_mlps.append(raref_left_mlp)

        raref_right_model = os.path.join(basepath, "raref_right", "checkpoints", "best_model.pt")
        raref_right_mlp = RootfindMLP(input_dim=3,output_dim=1, output_mapping=torch.sigmoid, input_min=input_min, input_max=input_max, d_ff=n_neurons//2, depth=n_layers).to(device).to(dtype)
        raref_right_mlp.load_state_dict(torch.load(raref_right_model, weights_only=True,map_location=device))
        raref_right_mlps.append(raref_right_mlp)
    
    eos = hybrid_eos(0,2,gamma)
    raref_solver = EnsembleRarefSolver(raref_left_mlps,raref_right_mlps,eos)

    dshock_solver = EnsembleDShockSolver(dshock_mlps, eos)
    draref_solver = EnsembleDRarefSolver(draref_mlps, raref_solver, eos)
    rarefshock_solver = EnsembleRarefShockSolver(rarefshock_mlps, raref_solver, eos)
    
    riemann_solver = RiemannSolver(dshock_solver, draref_solver, rarefshock_solver, 1e-10, gamma, torch.float64, device)
    
    return riemann_solver
