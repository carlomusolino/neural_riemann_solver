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

import torch.nn.init as init

from hydro.hlle_solver import HLLESolver
from .weight_initialization import * 

from .MLP import  Compactification

from riemannML.exact.riemann_solver import get_vel_shock, get_vel_raref, get_csnd, get_h, raref_rhs

import os



class EnsembleRarefSolver(nn.Module):
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
    
    def __init__(self, left_nets, right_nets, eos):
        super().__init__() 
        self.nets = {
            +1: left_nets, 
            -1: right_nets
        }
        self.gamma = eos.gamma_th 
    
    def _eval_and_select(self, rhoa, pa, vela, mapping, sign):
        prims = torch.stack(
            (torch.log(rhoa),
            torch.log(pa),
            vela), dim=1 
        )  # shape: (batch_size, 3)
        
        _cs_pred = [] 
        for net in self.nets[sign]:
            _cs_pred.append(mapping.xi_to_x(net(prims).squeeze(-1)))
        
        cs_pred = torch.stack(_cs_pred, dim=1)  # shape: (batch_size, n_nets)
        
        # Evaluate residuals
        residuals = raref_rhs(
            cs_pred,
            rhoa.unsqueeze(1).expand(-1, cs_pred.shape[1]),
            pa.unsqueeze(1).expand(-1, cs_pred.shape[1]),
            vela.unsqueeze(1).expand(-1, cs_pred.shape[1]),
            sign,
            self.gamma
        )  # shape: (batch_size, n_nets)

        # Find best network per batch entry
        best_idx = torch.argmin(torch.abs(residuals), dim=1)  # shape: (batch_size,)

        # Now select the best prediction for each batch entry
        batch_indices = torch.arange(cs_pred.shape[0], device=cs_pred.device)
        best_cs_pred = cs_pred[batch_indices, best_idx]

        return best_cs_pred

    
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
        cs_pred = self._eval_and_select(rhoa,pa,vela,mapping,sign)
        
        vel = sign * cs_pred 
        rho = rhoa * ((cs_pred**2*(self.gamma-1-csa**2))/(csa**2*(self.gamma-1-cs_pred**2)))**(1/(self.gamma-1))
        p   = cs_pred**2 * (self.gamma-1) * rho / (self.gamma - 1 - cs_pred**2) / self.gamma 
        
        h = get_h(rho,p,self.gamma)
        
        return rho, p, h, vel
    
class EnsembleDShockSolver(nn.Module):
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
    def _eval_and_select(self, P):
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
        _pressC = [] 
        for net in self.pressnet:
            _pressC.append(mapping.xi_to_x(net(prims).squeeze(-1)))
        pressC = torch.stack(_pressC, dim=1)  # shape: (batch_size, n_nets)
        # Evaluate residuals
        # Get the rest of the state 
        rhoCL, _, hCL, _, vstarL, vshockL = get_vel_shock(pressC,
                                                          rhoL.unsqueeze(1).expand(-1, pressC.shape[1]),
                                                          pL.unsqueeze(1).expand(-1, pressC.shape[1]),
                                                          vL.unsqueeze(1).expand(-1, pressC.shape[1]), 
                                                          -1, 
                                                          self.gamma)
        rhoCR, _, hCR, _, vstarR, vshockR = get_vel_shock(pressC,
                                                          rhoR.unsqueeze(1).expand(-1, pressC.shape[1]),
                                                          pR.unsqueeze(1).expand(-1, pressC.shape[1]),
                                                          vR.unsqueeze(1).expand(-1, pressC.shape[1]),
                                                          +1, self.gamma)
        residuals = (vstarL-vstarR).abs() 
        # Find best network per batch entry
        best_idx = torch.argmin(residuals, dim=1)  # shape: (batch_size,)
        # Now return 
        batch_indices = torch.arange(pressC.shape[0], device=pressC.device)
        return pressC[batch_indices, best_idx], rhoCL[batch_indices, best_idx], hCL[batch_indices, best_idx], vstarL[batch_indices, best_idx], vshockL[batch_indices, best_idx], \
                rhoCR[batch_indices, best_idx], hCR[batch_indices, best_idx], vstarR[batch_indices, best_idx], vshockR[batch_indices, best_idx]
    def forward(self,P,U,F,cmax,cmin):
        # Extract dimensions
        ncells, nvars, _ = F.shape 
        
        pressC, rhoCL, hCL, vstarL, vshockL, rhoCR, hCR, vstarR, vshockR = self._eval_and_select(P)
        
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
    
    
class EnsembleDRarefSolver(nn.Module):
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
    def _eval_and_select(self, P):
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
        mapping = Compactification(method='affine', a=torch.zeros_like(pL,device=P.device,dtype=P.dtype), b=torch.min(pL,pR))
        _pressC = []
        for net in self.pressnet:
            _pressC.append(mapping.xi_to_x(net(prims).squeeze(-1)))
        pressC = torch.stack(_pressC, dim=1)  # shape: (batch_size, n_nets)
        rhoCL, _, hCL, csCL, vstarL, _ = get_vel_raref(pressC,
                                                       rhoL.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       pL.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       vL.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       -1, self.gamma)
        rhoCR, _, hCR, csCR, vstarR, _ = get_vel_raref(pressC,
                                                       rhoR.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       pR.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       vR.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       +1, self.gamma)
        residuals = (vstarL-vstarR).abs()
        # Find best network per batch entry
        best_idx = torch.argmin(residuals, dim=1)  # shape: (batch_size,)
        # Now return
        batch_indices = torch.arange(pressC.shape[0], device=pressC.device)
        return pressC[batch_indices, best_idx], rhoCL[batch_indices, best_idx], hCL[batch_indices, best_idx], csCL[batch_indices, best_idx], vstarL[batch_indices, best_idx], \
                rhoCR[batch_indices, best_idx], hCR[batch_indices, best_idx], csCR[batch_indices, best_idx], vstarR[batch_indices, best_idx]
        
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
        
        # Compute the pressure from the MLPs
        pressC, rhoCL, hCL, csCL, vstarL, rhoCR, hCR, csCR, vstarR = self._eval_and_select(P)
        
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
    
    
class EnsembleRarefShockSolver(nn.Module):
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
    
    def _eval_and_select(self, P):
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
        mapping = Compactification(method='affine', a=pR, b=pL)
        _pressC = []
        for net in self.pressnet:
            _pressC.append(mapping.xi_to_x(net(prims).squeeze(-1)))
        pressC = torch.stack(_pressC, dim=1)  # shape: (batch_size, n_nets)
        # Evaluate residuals
        rhoCL, _, hCL, csCL, vstarL, _ = get_vel_raref(pressC,
                                                       rhoL.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       pL.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       vL.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       -1, self.gamma)
        rhoCR, _, hCR, _, vstarR, vshock = get_vel_shock(pressC,
                                                       rhoR.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       pR.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       vR.unsqueeze(1).expand(-1,pressC.shape[1]), 
                                                       +1, self.gamma)
        # Get best network per batch entry
        residuals = (vstarL-vstarR).abs()
        best_idx = torch.argmin(residuals, dim=1)  # shape: (batch_size,)
        # Now return
        batch_indices = torch.arange(pressC.shape[0], device=pressC.device)
        return pressC[batch_indices, best_idx], rhoCL[batch_indices, best_idx], hCL[batch_indices, best_idx], csCL[batch_indices, best_idx], vstarL[batch_indices, best_idx], \
                rhoCR[batch_indices, best_idx], hCR[batch_indices, best_idx],  vstarR[batch_indices, best_idx], vshock[batch_indices, best_idx]
    
    def forward(self,P,U,F,cmax,cmin):
        # Extract dimensions
        ncells, nvars, _ = F.shape 
        
        # Extract left/right prims
        rhoL = (P[:,0,0])
        pL   = (P[:,1,0])
        vL   = (P[:,2,0])
        csL  = get_csnd(rhoL,pL,self.gamma)
        
        pressC, rhoCL, hCL, csCL, vstarL, rhoCR, hCR, vstarR, vshock = self._eval_and_select(P)
        
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