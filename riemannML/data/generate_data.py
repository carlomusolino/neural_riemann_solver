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
from riemannML.exact.riemann_solver import * 
from riemannML.exact.hybrid_eos import *

from riemannML.exact.riemann_solver import get_limiting_velocity_double_raref, get_limiting_velocity_shock_raref, get_limiting_velocity_double_shock, solve_raref, compute_riemann_invariant

import random

def filter_continuous(prims, min_jump):
    press_jump = (torch.exp(prims[:,1,1]) - torch.exp(prims[:,1,0])).abs()
    rho_jump   = (prims[:,0,1] - prims[:,0,0]).abs()
    vel_jump   = (prims[:,2,1] - prims[:,2,0]).abs()
    mask = (press_jump>min_jump) #| (rho_jump>min_jump) | (vel_jump>min_jump)
    return prims[mask,:,:]

def flip_state(rho,press,vel, dtype=torch.float32):
    # Flip states 
    flip_mask = press[:, 1] > press[:, 0]  # Flip to ensure pL >= pR
    rho_norm = torch.where(flip_mask.unsqueeze(1), rho.flip(dims=[1]), rho)
    press_norm = torch.where(flip_mask.unsqueeze(1), press.flip(dims=[1]), press)
    vel_norm = torch.where(flip_mask.unsqueeze(1), -vel.flip(dims=[1]), vel)
    return torch.stack((rho_norm,press_norm,vel_norm), dim=1).to(dtype)

def generate_dataset(lpressmin,lpressmax,pjumpmin, pjumpmax, lrhomin,lrhomax,velmin,velmax,target, N, device, gamma, dtype, seed=None):
    """
    Generates a training dataset of physical input parameters for a Riemann problem using a Sobol sequence.
    Parameters:
        lpressmin (float): Minimum value for the logarithm of the left pressure (log10(p_L)).
        lpressmax (float): Maximum value for the logarithm of the left pressure (log10(p_L)).
        pjumpmin (float): Minimum value for the logarithm of the pressure jump (log10(p_R - p_L)).
        pjumpmax (float): Maximum value for the logarithm of the pressure jump (log10(p_R - p_L)).
        lrhomin (float): Minimum value for the logarithm of the left and right densities (log10(rho_L), log10(rho_R)).
        lrhomax (float): Maximum value for the logarithm of the left and right densities (log10(rho_L), log10(rho_R)).
        velmin (float): Minimum value for the left velocity (v_L).
        velmax (float): Maximum value for the left velocity (v_L).
        target (str): Target type for the dataset. Must be one of ['d_shock', 'd_raref', 'rarefshock'].
        N (int): Number of samples to generate.
        device (torch.device): Device on which tensors will be allocated (e.g., 'cpu' or 'cuda').
        dtype (torch.dtype): Data type of the tensors (e.g., torch.float32 or torch.float64).
    Returns:
        torch.Tensor: A tensor of shape (N, 3, 2) containing the generated dataset. The dimensions represent:
            - [:, 0, :] -> Left and right densities (rho_L, rho_R).
            - [:, 1, :] -> Left and right pressures (p_L, p_R).
            - [:, 2, :] -> Left and right velocities (v_L, v_R).
    Raises:
        ValueError: If the `target` parameter is not one of ['d_shock', 'd_raref', 'rarefshock'].
    Notes:
        - The dataset is generated using a Sobol sequence for quasi-random sampling.
        - The physical parameters are scaled to their respective ranges defined by the input bounds.
        - The relative velocity is adjusted based on the target type and limiting velocities.
    """
    if not target in ['d_shock', 'd_raref', 'rarefshock']: 
        raise ValueError("target is invalid.")
    
    convfac = np.log10(np.exp(1))
    # NB pR = PL * 10**(x) x in pjumpmin pjumpmax 
    lb = torch.tensor([
    lrhomin/convfac,     # log10(rho_L)
    lrhomin/convfac, # log10(rho_R)
    lpressmin/convfac, # log10(p_L)
    pjumpmin/convfac,   # log10(p_R)
    velmin,      # v_L
    0       # relative velocity
    ], dtype=dtype, device=device)

    ub = torch.tensor([
        lrhomax/convfac,     # log10(rho_L)
        lrhomax/convfac, # log10(rho_R)
        lpressmax/convfac,   # log10(p_L)
        pjumpmax/convfac, # log10(p_R)
        velmax,      # v_L
        1       # relative velocity
    ], dtype=dtype, device=device)

    engine = torch.quasirandom.SobolEngine(dimension=6, scramble=True, seed=seed)
    samples = engine.draw(N).to(device)  # (N, 6)
    phys_vals = lb + (ub - lb) * samples  # scale to physical domain
    
    get_vb = lambda vab, va: (-va+vab)/(va * vab - 1)
    
    rho_L     = phys_vals[:, 0]
    rho_R     = phys_vals[:, 1]
    press_R   = phys_vals[:, 2]
    press_L   = torch.clamp(press_R + phys_vals[:, 3], max=lpressmax/convfac)
    vel_L     = phys_vals[:,4]
    vel_R     = torch.zeros_like(press_L, device=device, dtype=dtype)
    
    v_2S = get_limiting_velocity_double_shock(torch.exp(rho_L),torch.exp(press_L),vel_L,torch.exp(rho_R),torch.exp(press_R),vel_R,gamma)
    v_SR = get_limiting_velocity_shock_raref(torch.exp(rho_L),torch.exp(press_L),vel_L,torch.exp(rho_R),torch.exp(press_R),vel_R,gamma)
    v_2R = get_limiting_velocity_double_raref(torch.exp(rho_L),torch.exp(press_L),vel_L,torch.exp(rho_R),torch.exp(press_R),vel_R,gamma)
    
    if target == 'd_shock':
        vab_min = v_2S 
        vab_max = torch.full_like(vab_min, 0.99, device=device, dtype=dtype)
    elif target == 'rarefshock':
        vab_min = v_SR 
        vab_max = v_2S 
    elif target == 'd_raref':
        vab_min = v_2R 
        vab_max = v_SR 
    
    phys_vals[:,5] = vab_min + (vab_max-vab_min) * phys_vals[:,5]
    vel_R = torch.clamp(get_vb(phys_vals[:,5], vel_L), min=-0.99, max=0.99)
    
    return torch.stack(
        (torch.stack((rho_L,rho_R), dim=1), 
        torch.stack((press_L,press_R), dim=1), 
        torch.stack((vel_L,vel_R), dim=1)), dim=1
    ).to(device).to(dtype)
    
def generate_dataset_raref(lpressmin,lpressmax,lrhomin,lrhomax,velmin,velmax, N, device, gamma, dtype, seed=None):
    """
    Generates a dataset of left-going and right-going rarefaction wave states 
    for a Riemann problem in relativistic hydrodynamics.
    Parameters:
        lpressmin (float): Minimum value of the logarithm of pressure (log10(p_L)).
        lpressmax (float): Maximum value of the logarithm of pressure (log10(p_L)).
        lrhomin (float): Minimum value of the logarithm of density (log10(rho_L)).
        lrhomax (float): Maximum value of the logarithm of density (log10(rho_L)).
        velmin (float): Minimum value of velocity (v_L).
        velmax (float): Maximum value of velocity (v_L).
        N (int): Number of samples to generate.
        device (torch.device): Device to perform computations on (e.g., 'cpu' or 'cuda').
        gamma (float): Adiabatic index of the fluid.
        dtype (torch.dtype): Data type for tensors (e.g., torch.float32).
        seed (int, optional): Seed for the Sobol sequence generator. Default is None.
    Returns:
        tuple: A tuple containing:
            - (state_L, cs_L): Left-going rarefaction states and corresponding sound speeds.
                - state_L (torch.Tensor): Tensor of shape (M, 3) containing valid left-going states 
                    [log(rho_L), log(p_L), v_L].
                - cs_L (torch.Tensor): Tensor of shape (M,) containing sound speeds for left-going states.
            - (state_R, cs_R): Right-going rarefaction states and corresponding sound speeds.
                - state_R (torch.Tensor): Tensor of shape (M, 3) containing valid right-going states 
                    [log(rho_R), log(p_R), v_R].
                - cs_R (torch.Tensor): Tensor of shape (M,) containing sound speeds for right-going states.
    Notes:
        - The function uses a Sobol sequence to generate quasi-random samples in the parameter space.
        - Filters are applied to ensure the generated states satisfy physical constraints for rarefaction waves.
        - The function assumes the existence of helper functions `get_csnd`, `compute_riemann_invariant`, 
            and `solve_raref` for computing sound speed, Riemann invariants, and solving rarefaction conditions, respectively.
    """
    
    convfac = np.log(np.exp(1))

    lb = torch.tensor([
    lrhomin/convfac,     # log10(rho_L)
    lpressmin/convfac,   # log10(p_L)
    velmin,              # v_L
    ], dtype=dtype, device=device)

    ub = torch.tensor([
        lrhomax/convfac,     # log10(rho_L)
        lpressmax/convfac,   # log10(p_L)
        velmax,              # v_L
    ], dtype=dtype, device=device)
    
    
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    samples = engine.draw(N).to(device)   # (N, 6)
    phys_vals = lb + (ub - lb) * samples  # scale to physical domain

    prims = phys_vals.clone()
    rho   = torch.exp(phys_vals[:,0])
    press = torch.exp(phys_vals[:,1])
    vel   = phys_vals[:,2]

    cs    = get_csnd(rho,press,gamma)

    # Left going, filter valid states

    # Get head speed
    xih_L = (vel-cs)/(1-vel*cs)
    # Filter supersonic states
    mask1_L = xih_L < 0 
    # Compute Riemann invariant 
    Ja_L = compute_riemann_invariant(rho[mask1_L],press[mask1_L], vel[mask1_L], +1, gamma)
    # Get lower limit of xi 
    xi0_L = (Ja_L-1)/(Ja_L+1)
    # Find states that can be connected with xi = 0 
    mask2_L = xi0_L >= 0
    rho_L   = rho[mask1_L][mask2_L]
    press_L = press[mask1_L][mask2_L]
    vel_L   = vel[mask1_L][mask2_L]
    cs_L    = solve_raref(rho_L,press_L,vel_L,+1,gamma)

    state_L = torch.stack((
        torch.log(rho_L),
        torch.log(press_L),
        vel_L
    ), dim=1)

    # Right going, filter valid states

    # Get head speed
    xih_R = (vel+cs)/(1+vel*cs)
    # Filter supersonic states
    mask1_R = xih_R > 0 
    # Compute Riemann invariant 
    Ja_R = compute_riemann_invariant(rho[mask1_R],press[mask1_R], vel[mask1_R], -1, gamma)
    # Get lower limit of xi 
    xi0_R = (Ja_R-1)/(Ja_R+1)
    # Find states that can be connected with xi = 0 
    mask2_R = xi0_R <= 0
    rho_R   = rho[mask1_R][mask2_R]
    press_R = press[mask1_R][mask2_R]
    vel_R   = vel[mask1_R][mask2_R]
    cs_R    = solve_raref(rho_R,press_R,vel_R,-1,gamma)

    state_R = torch.stack((
        torch.log(rho_R),
        torch.log(press_R),
        vel_R
    ), dim=1)
    
    return (state_L,cs_L), (state_R, cs_R)
    
def get_coverage(prims, bins=10):
    """
    Computes the coverage of a 6-dimensional space by mapping input samples to discrete bins.

    Args:
        prims (torch.Tensor): A tensor of shape (B, 3, 2) representing the input samples, 
                              where B is the batch size.
        bins (int, optional): The number of bins to divide each feature into. Defaults to 10.

    Returns:
        float: The coverage ratio, defined as the number of unique bins hit divided by the 
               total number of bins in the 6-dimensional space.
    """
    # Flatten (B, 3, 2) → (B, 6)
    samples = prims.flatten(start_dim=1)

    # Normalize to [0, 1] per feature
    min_vals = samples.min(dim=0, keepdim=True)[0]
    max_vals = samples.max(dim=0, keepdim=True)[0]
    normed = (samples - min_vals) / (max_vals - min_vals + 1e-8)

    # Digitize into [0, bins-1]
    digitized = (normed * bins).long().clamp(0, bins - 1)

    # Map 6D bin index to unique scalar (like base-bins positional encoding)
    indices = digitized[:, 0]
    for i in range(1, 6):
        indices = indices * bins + digitized[:, i]

    # Count unique bins hit
    unique_bins = indices.unique().numel()
    total_bins = bins**6

    coverage = unique_bins / total_bins
    return coverage