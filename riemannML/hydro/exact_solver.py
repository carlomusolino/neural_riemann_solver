# Numpy and matplotlib
import numpy as np 
import matplotlib.pyplot as plt 

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


import torch.nn.init as init

from riemannML.hydro.hlle_solver import HLLESolver


from riemannML.exact.riemann_solver import get_vel_shock, get_vel_raref, get_csnd, classify_wave_pattern, get_csnd, get_h, raref
from riemannML.utilities.rootfinding import bisection_solver

def get_relative_velocity_double_shock(p3, rho1,p1,v1,rho2,p2,v2, gamma):
    _, _, _, _, v3 , _ = get_vel_shock(p3,rho1,p1,v1,-1,gamma)
    _, _, _, _, v3p, _ = get_vel_shock(p3,rho2,p2,v2,+1,gamma)
    return v3-v3p

 
def get_relative_velocity_shock_raref(p3, rho1,p1,v1,rho2,p2,v2, gamma):
    _, _, _, _, v3 , _ = get_vel_raref(p3,rho1,p1,v1,-1,gamma)
    _, _, _, _, v3p, _ = get_vel_shock(p3,rho2,p2,v2,+1,gamma)
    return v3-v3p
    
    
def get_relative_velocity_double_raref(p3, rho1,p1,v1,rho2,p2,v2, gamma):
    _, _, _, _, v3 , _ = get_vel_raref(p3,rho1,p1,v1,-1,gamma)
    _, _, _, _, v3p, _ = get_vel_raref(p3,rho2,p2,v2,+1,gamma)
    return v3-v3p
    
class ExactRiemannSolver(nn.Module):
    """
    ExactRiemannSolver
    This class implements an exact Riemann solver for hydrodynamic problems using PyTorch. 
    It computes the fluxes across interfaces in a computational domain based on the 
    Riemann problem's initial conditions.
    Attributes:
        continuity_cutoff (float): Threshold for determining continuous data.
        gamma (float): Adiabatic index of the gas.
        hlle_solver (HLLESolver): HLLE approximate Riemann solver for fallback cases.
    Methods:
        __init__(gamma):
            Initializes the ExactRiemannSolver with the given adiabatic index.
        _insert_fluxes(target_tensor, mask, remaining_mask, new_fluxes):
            Inserts computed fluxes into the target tensor at specified indices.
        _get_central_states(press_c, rho1, p1, v1, rho2, p2, v2, dshock_mask, draref_mask, rarefshock_mask):
            Computes the central states (density, pressure, velocity, etc.) for different wave patterns.
        _get_frl_frr(p_RL, p_RR, rho_RL, rho_RR, h_RL, h_RR, v_RL, v_RR, draref_mask, shockraref_mask):
            Computes the fluxes for the left and right rarefaction regions.
        _get_fcl_fcr(pressC, rhoCL, rhoCR, hCL, hCR, lambdaC):
            Computes the fluxes for the central left and central right states.
        forward(P, U, F, cmax, cmin):
            Main method to compute the fluxes across interfaces.
            Args:
                P (torch.Tensor): Primitive variables (density, pressure, velocity) for left and right states.
                U (torch.Tensor): Conservative variables for left and right states.
                F (torch.Tensor): Fluxes for left and right states.
                cmax (torch.Tensor): Maximum wave speeds.
                cmin (torch.Tensor): Minimum wave speeds.
            Returns:
                torch.Tensor: Computed fluxes across interfaces.
    """
    
    def __init__(self, gamma):
        
        super().__init__() 
        self.continuity_cutoff = 1e-10
        self.gamma = gamma
        self.hlle_solver = HLLESolver()
    
    def _insert_fluxes(self,target_tensor, mask, remaining_mask, new_fluxes):
        indices_in_full = remaining_mask.nonzero()[mask].squeeze()
        target_tensor[indices_in_full, :] = new_fluxes
    
    def _get_central_states(self, press_c, rho1,p1,v1, rho2,p2,v2, dshock_mask, draref_mask, rarefshock_mask):
        device = press_c.device 
        dtype  = press_c.dtype 
        
        lambda_C  = torch.zeros_like(press_c, device=device, dtype=dtype)
        lambda_CL = torch.zeros_like(press_c, device=device, dtype=dtype)
        lambda_CR = torch.zeros_like(press_c, device=device, dtype=dtype)
        lambda_L  = torch.zeros_like(press_c, device=device, dtype=dtype)
        lambda_R  = torch.zeros_like(press_c, device=device, dtype=dtype)
        rho_CL    = torch.zeros_like(press_c, device=device, dtype=dtype)
        rho_CR    = torch.zeros_like(press_c, device=device, dtype=dtype)
        rho_RL    = torch.zeros_like(press_c, device=device, dtype=dtype)
        rho_RR    = torch.zeros_like(press_c, device=device, dtype=dtype)
        h_CL      = torch.zeros_like(press_c, device=device, dtype=dtype)
        h_CR      = torch.zeros_like(press_c, device=device, dtype=dtype)
        h_RL      = torch.zeros_like(press_c, device=device, dtype=dtype)
        h_RR      = torch.zeros_like(press_c, device=device, dtype=dtype)
        p_RL      = torch.zeros_like(press_c, device=device, dtype=dtype)
        p_RR      = torch.zeros_like(press_c, device=device, dtype=dtype)
        v_RL      = torch.zeros_like(press_c, device=device, dtype=dtype)
        v_RR      = torch.zeros_like(press_c, device=device, dtype=dtype)
        
        cs_L = get_csnd(rho1,p1,self.gamma)
        cs_R = get_csnd(rho2,p2,self.gamma)
        
        rho_CL[dshock_mask], _, h_CL[dshock_mask], _, v3 , lambda_L[dshock_mask] = get_vel_shock(press_c[dshock_mask], 
                                                                                                rho1[dshock_mask],p1[dshock_mask],
                                                                                                v1[dshock_mask],-1,self.gamma)
        rho_CR[dshock_mask], _, h_CR[dshock_mask], _, v3p, lambda_R[dshock_mask] = get_vel_shock(press_c[dshock_mask],
                                                                                                rho2[dshock_mask],p2[dshock_mask],
                                                                                                v2[dshock_mask],+1,self.gamma)
        lambda_C[dshock_mask]  = 0.5 * (v3+v3p)
        lambda_CL[dshock_mask] = lambda_L[dshock_mask]
        lambda_CR[dshock_mask] = lambda_R[dshock_mask]
        
        rho_CL[draref_mask], _, h_CL[draref_mask], cs_CL, v3 , _ = get_vel_raref(press_c[draref_mask],
                                                                                rho1[draref_mask],p1[draref_mask],
                                                                                v1[draref_mask],-1,self.gamma)
        rho_CR[draref_mask], _, h_CR[draref_mask], cs_CR, v3p, _ = get_vel_raref(press_c[draref_mask],
                                                                                rho2[draref_mask],p2[draref_mask],
                                                                                v2[draref_mask],+1,self.gamma)
        lambda_C[draref_mask]  = 0.5 * (v3+v3p)
        lambda_L[draref_mask]  = (v1[draref_mask] - cs_L[draref_mask]) / ( 1 - v1[draref_mask] * cs_L[draref_mask])
        lambda_CL[draref_mask] = (lambda_C[draref_mask] - cs_CL) / ( 1 - lambda_C[draref_mask] * cs_CL)
        lambda_CR[draref_mask] = (lambda_C[draref_mask] + cs_CR) / ( 1 + lambda_C[draref_mask] * cs_CR)
        lambda_R[draref_mask]  = (v2[draref_mask] + cs_R[draref_mask]) / ( 1 + v2[draref_mask] * cs_R[draref_mask])

        rho_CL[rarefshock_mask], _, h_CL[rarefshock_mask], cs_CL, v3 , _ = get_vel_raref(press_c[rarefshock_mask],
                                                                                        rho1[rarefshock_mask],p1[rarefshock_mask], 
                                                                                        v1[rarefshock_mask],-1,self.gamma)
        rho_CR[rarefshock_mask], _, h_CR[rarefshock_mask], _, v3p, lambda_R[rarefshock_mask] = get_vel_shock(press_c[rarefshock_mask],
                                                                                                            rho2[rarefshock_mask],p2[rarefshock_mask],
                                                                                                            v2[rarefshock_mask],+1,self.gamma)
        lambda_C[rarefshock_mask]  = 0.5 * (v3+v3p)
        lambda_L[rarefshock_mask]  = (v1[rarefshock_mask] - cs_L[rarefshock_mask]) / ( 1 - v1[rarefshock_mask] * cs_L[rarefshock_mask])
        lambda_CL[rarefshock_mask] = (lambda_C[rarefshock_mask] - cs_CL) / ( 1 - lambda_C[rarefshock_mask] * cs_CL)
        lambda_CR[rarefshock_mask] = lambda_R[rarefshock_mask]
        if draref_mask.any():
            xi = torch.zeros(draref_mask.sum(), device=device, dtype=dtype)
            # Signs here are NOT a bug, just an unfortunate convention in this function
            rho_RL[draref_mask], p_RL[draref_mask], _, _, _, v_RL[draref_mask] = raref(xi, rho1[draref_mask], p1[draref_mask], v1[draref_mask], self.gamma,+1)
            rho_RR[draref_mask], p_RR[draref_mask], _, _, _, v_RR[draref_mask] = raref(xi, rho2[draref_mask], p2[draref_mask], v2[draref_mask], self.gamma,-1)
            h_RL[draref_mask] = get_h(rho_RL[draref_mask], p_RL[draref_mask], self.gamma)
            h_RR[draref_mask] = get_h(rho_RR[draref_mask], p_RR[draref_mask], self.gamma)
        if rarefshock_mask.any():
            xi = torch.zeros(rarefshock_mask.sum(), device=device, dtype=dtype)
            rho_RL[rarefshock_mask], p_RL[rarefshock_mask], _, _, _, v_RL[rarefshock_mask] = raref(xi, rho1[rarefshock_mask], p1[rarefshock_mask], v1[rarefshock_mask], self.gamma,+1)
            h_RL[rarefshock_mask] = get_h(rho_RL[rarefshock_mask], p_RL[rarefshock_mask], self.gamma)

        return rho_RL, rho_RR, rho_CL, rho_CR, lambda_C, lambda_L, lambda_CL, lambda_CR, lambda_R, h_CL, h_CR, h_RL, h_RR, v_RL, v_RR, p_RL, p_RR
    
    def _get_frl_frr(self,p_RL, p_RR, rho_RL, rho_RR, h_RL, h_RR, v_RL, v_RR, draref_mask, shockraref_mask):
        WCL = 1/torch.sqrt(1-v_RL**2)
        WCR = 1/torch.sqrt(1-v_RR**2)
        densCL  = WCL * rho_RL 
        densCR  = WCR * rho_RR 
        
        tauCL   = densCL * ( WCL * h_RL - 1 ) * v_RL 
        tauCR   = densCR * ( WCR * h_RR - 1 ) * v_RR 
        
        momCL   = WCL**2 * rho_RL * h_RL * v_RL 
        momCR   = WCR**2 * rho_RR * h_RR * v_RR 
        
        uCL     = torch.stack([densCL, tauCL, momCL], dim=1)
        uCR     = torch.stack([densCR, tauCR, momCR], dim=1)
        
        FRL     = torch.zeros_like(uCL, device=v_RL.device, dtype=v_RL.dtype)
        FRR     = torch.zeros_like(uCR, device=v_RL.device, dtype=v_RL.dtype)
        
        FRL[:,0] = uCL[:,0] * v_RL 
        FRR[:,0] = uCR[:,0] * v_RR 
        
        FRL[:,1] = uCL[:,0] * (WCL * h_RL - 1) * v_RL 
        FRR[:,1] = uCR[:,0] * (WCR * h_RR - 1) * v_RR 
        
        FRL[:,2] = uCL[:,2]* v_RL + p_RL  
        FRR[:,2] = uCR[:,2]* v_RR + p_RR 
        return FRL, FRR
    
    def _get_fcl_fcr(self, pressC, rhoCL, rhoCR, hCL, hCR, lambdaC):
        WC = 1/torch.sqrt(1-lambdaC**2)
        densCL  = WC * rhoCL 
        densCR  = WC * rhoCR 
        
        tauCL   = densCL * ( WC * hCL - 1 ) * lambdaC 
        tauCR   = densCR * ( WC * hCR - 1 ) * lambdaC 
        
        momCL   = WC**2 * rhoCL * hCL * lambdaC 
        momCR   = WC**2 * rhoCR * hCR * lambdaC 
        
        uCL     = torch.stack([densCL, tauCL, momCL], dim=1)
        uCR     = torch.stack([densCR, tauCR, momCR], dim=1)
        FCL     = torch.zeros_like(uCL, device=lambdaC.device, dtype=lambdaC.dtype)
        FCR     = torch.zeros_like(uCR, device=lambdaC.device, dtype=lambdaC.dtype)
        
        FCL[:,0] = uCL[:,0] * lambdaC 
        FCR[:,0] = uCR[:,0] * lambdaC 
        
        FCL[:,1] = uCL[:,0] * (WC * hCL - 1) * lambdaC 
        FCR[:,1] = uCR[:,0] * (WC * hCR - 1) * lambdaC 
        
        FCL[:,2] = uCL[:,2]* lambdaC + pressC  
        FCR[:,2] = uCR[:,2]* lambdaC + pressC 
        return FCL, FCR
        
    
    def forward(self, P, U, F, cmax, cmin):
        # Extract dimensions
        ncells, nvars, _ = F.shape
        dtype, device = F.dtype, F.device
        
        fluxes = torch.zeros(ncells,nvars, device=device, dtype=dtype)
        remaining_mask = torch.ones(ncells, device=device, dtype=torch.bool)
        
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
        
        # ---------------------- STEP 3: COMPUTE FLUXES  ----------------------
        # unwrap prims
        rho1 = P[:,0,0]
        rho2 = P[:,0,1]
        p1   = P[:,1,0]
        p2   = P[:,1,1]
        v1   = P[:,2,0]
        v2   = P[:,2,1]
        # get masks 
        dshock_mask     = (labels==0)
        draref_mask     = (labels==1)
        shockraref_mask = (labels==2)
        vacuum_mask     = (labels==3)
        
        f_dshock = lambda p: get_relative_velocity_double_shock(p,
                                                                rho1[dshock_mask],
                                                                p1[dshock_mask],
                                                                v1[dshock_mask],
                                                                rho2[dshock_mask],
                                                                p2[dshock_mask],
                                                                v2[dshock_mask],self.gamma)
        f_draref = lambda p: get_relative_velocity_double_raref(p,
                                                                rho1[draref_mask],
                                                                p1[draref_mask],
                                                                v1[draref_mask],
                                                                rho2[draref_mask],
                                                                p2[draref_mask],
                                                                v2[draref_mask],self.gamma)
        f_rarefshock = lambda p: get_relative_velocity_shock_raref(p,
                                                                   rho1[shockraref_mask],
                                                                   p1[shockraref_mask],
                                                                   v1[shockraref_mask],
                                                                   rho2[shockraref_mask],
                                                                   p2[shockraref_mask],
                                                                   v2[shockraref_mask],self.gamma)
        
        # Get mins and maxs for rootfinding 
        pmin = torch.zeros_like(rho1)
        pmax = torch.zeros_like(rho1)
        
        pmin[dshock_mask]     = torch.max(p1[dshock_mask],p2[dshock_mask]) + 1e-45
        
        pmin[shockraref_mask] = torch.min(p1[shockraref_mask],p2[shockraref_mask])
        pmax[draref_mask]     = torch.min(p1[draref_mask],p2[draref_mask])
        pmax[shockraref_mask] = torch.max(p1[shockraref_mask],p2[shockraref_mask])
        pmin[draref_mask]     = pmax[draref_mask]
        # Find pmax for the double shock case
        pmax[dshock_mask] = pmin[dshock_mask]
        cmask = torch.zeros_like(pmax[dshock_mask], dtype=torch.bool)
        while(True):
            pmax[dshock_mask] = torch.where(cmask, pmax[dshock_mask], 2*pmax[dshock_mask] )
            cmask = f_dshock(pmin[dshock_mask]) * f_dshock(pmax[dshock_mask]) <= 0 
            if torch.all(cmask):
                break
            if torch.any(pmax[dshock_mask]>1e10):
                print(f"Warning {dshock_mask.sum() - cmask.sum()} roots are not bracketed in d_shock")
                break 
        cmask = torch.zeros_like(pmin[draref_mask], dtype=torch.bool)
        while(True):
            pmin[draref_mask] = torch.where(cmask, pmin[draref_mask], 0.5*pmin[draref_mask] )
            cmask = f_draref(pmin[draref_mask]) * f_draref(pmax[draref_mask]) <= 0 
            if torch.all(cmask):
                break
            if torch.any(pmin[draref_mask]<1e-15):
                print(f"Warning {draref_mask.sum() - cmask.sum()}  roots are not bracketed in d_raref")
                break 
        
        press_c = torch.zeros_like(pmax)
        convergence_mask = torch.zeros_like(pmax, dtype=torch.bool)
        
        # Solve for p
        if dshock_mask.any(): press_c[dshock_mask], convergence_mask[dshock_mask] = bisection_solver(f_dshock, pmin[dshock_mask], pmax[dshock_mask], 1e-15) 
        if draref_mask.any(): press_c[draref_mask], convergence_mask[draref_mask] = bisection_solver(f_draref, pmin[draref_mask], pmax[draref_mask], 1e-15)
        if shockraref_mask.any(): press_c[shockraref_mask], convergence_mask[shockraref_mask] = bisection_solver(f_rarefshock, pmin[shockraref_mask], pmax[shockraref_mask], 1e-15)
        press_c[vacuum_mask] = 1e-15 
        convergence_mask[vacuum_mask] = True

        # Solve for P_CL, P_CR
        rho_RL, rho_RR, rho_CL, rho_CR, lambda_C, lambda_L, lambda_CL, lambda_CR, lambda_R, h_CL, h_CR, h_RL, h_RR, v_RL, v_RR, p_RL, p_RR = self._get_central_states(press_c, rho1,p1,v1, rho2,p2,v2, dshock_mask, draref_mask, shockraref_mask)

        # Compute F_CL, F_CR 
        FCL,FCR = self._get_fcl_fcr(press_c, rho_CL, rho_CR, h_CL, h_CR, lambda_C)
        
        FRL, FRR = self._get_frl_frr(p_RL, p_RR, rho_RL, rho_RR, h_RL, h_RR, v_RL, v_RR, draref_mask, shockraref_mask)
        
        # Fill flux tensor
        mask_L   = (lambda_L >= 0).unsqueeze(1)
        mask_RL  = ((lambda_L<0)  & (lambda_CL>0)).unsqueeze(1)
        mask_CL  = ((lambda_CL<=0)  & (lambda_C>0)).unsqueeze(1)
        mask_CR  = ((lambda_C<=0) & (lambda_CR>=0)).unsqueeze(1)
        mask_RR  = ((lambda_CR<0) & (lambda_R>0)).unsqueeze(1)
        mask_R   = ((lambda_R<=0)).unsqueeze(1)
        
            
        fluxes_exact = torch.zeros(mask_L.size(0), 3, device=device, dtype=dtype) 
        fluxes_exact = torch.where(mask_L , F[:, :, 0], fluxes_exact)
        fluxes_exact = torch.where(mask_RL, FRL       , fluxes_exact)
        fluxes_exact = torch.where(mask_CL, FCL       , fluxes_exact)
        fluxes_exact = torch.where(mask_CR, FCR       , fluxes_exact)
        fluxes_exact = torch.where(mask_RR, FRR       , fluxes_exact)
        fluxes_exact = torch.where(mask_R , F[:, :, 1], fluxes_exact)
        
        if convergence_mask.any(): self._insert_fluxes(fluxes, convergence_mask, remaining_mask, fluxes_exact[convergence_mask])
        
        if not convergence_mask.all():
            fluxes_noconv = self.hlle_solver(P[~convergence_mask, :, :], U[~convergence_mask, :, :], F[~convergence_mask, :, :], cmax[~convergence_mask], cmin[~convergence_mask])
            self._insert_fluxes(fluxes, ~convergence_mask, remaining_mask, fluxes_noconv)
        
        # Flip the fluxes back where necessary 
        for i in range(3):  # over variables: rho, press, vel
            fluxes[:, i] = torch.where(
                flip_mask,
                fluxes[:,i] * (+1 if i==2 else -1),
                fluxes[:,i]
            )        
        return fluxes