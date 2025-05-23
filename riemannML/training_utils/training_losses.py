import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from riemannML.exact.riemann_solver import get_vel_shock, get_vel_raref, raref_rhs, get_csnd
from riemannML.solvers.MLP import Compactification

def compute_loss__d_shock(model, prims, pstar_true, gamma):
    """
    Computes the loss for the double shock configuration in a Riemann problem.
    This function calculates two loss components:
    1. The smooth L1 loss between the predicted velocities on the left and right sides of the shock.
    2. The smooth L1 loss between the logarithm of the predicted pressure and the logarithm of the true pressure.
    Args:
        model (torch.nn.Module): The neural network model used to predict the compactified pressure.
        prims (torch.Tensor): A tensor containing the primitive variables (log-density, log-pressure, velocity) 
                              for the left and right states. Shape: [batch_size, 3, 2].
        pstar_true (torch.Tensor): The true post-shock pressure values. Shape: [batch_size].
        gamma (float): The adiabatic index (ratio of specific heats).
    Returns:
        list: A list containing two loss values:
              - The velocity consistency loss (smooth L1 loss between left and right velocities).
              - The pressure prediction loss (smooth L1 loss between predicted and true pressures).
    """
    out  = model(prims)
    
    rhoL = torch.exp(prims[:,0,0])
    rhoR = torch.exp(prims[:,0,1])
    pL = torch.exp(prims[:,1,0])
    pR = torch.exp(prims[:,1,1])
    vL = prims[:,2,0]
    vR = prims[:,2,1]
    
    # out here is exponentiated, p* >= max(pL,pR) for consistency with double shock configuration
    xi_out = out.squeeze(-1)
    mapping = Compactification(method="compact", a=torch.max(pL,pR))
    press_out = mapping.xi_to_x(xi_out)
    
    _, _, _, _, vstarL_pred, _ = get_vel_shock(press_out,rhoL,pL,vL,-1,gamma)
    _, _, _, _, vstarR_pred, _ = get_vel_shock(press_out,rhoR,pR,vR,+1,gamma)

    return  [F.smooth_l1_loss(vstarL_pred, vstarR_pred, beta=1.0, reduction='mean'), F.smooth_l1_loss(torch.log(press_out), torch.log(pstar_true), beta=1.0, reduction='mean')]

def compute_loss__rarefshock(model, prims, pstar_true, gamma):
    """
    Computes the loss for a rarefaction-shock configuration in a Riemann problem.
    This function evaluates two loss components:
    1. The mismatch between the predicted velocities at the left and right states.
    2. The mismatch between the predicted and true pressure in logarithmic space.
    Args:
        model (torch.nn.Module): The neural network model used to predict the compactified pressure.
        prims (torch.Tensor): A tensor of primitive variables with shape (batch_size, 3, 2).
                              The tensor contains:
                              - prims[:, 0, 0]: log-density on the left state (rhoL).
                              - prims[:, 0, 1]: log-density on the right state (rhoR).
                              - prims[:, 1, 0]: log-pressure on the left state (pL).
                              - prims[:, 1, 1]: log-pressure on the right state (pR).
                              - prims[:, 2, 0]: velocity on the left state (vL).
                              - prims[:, 2, 1]: velocity on the right state (vR).
        pstar_true (torch.Tensor): The true pressure at the star region (p*).
        gamma (float): The adiabatic index of the gas.
    Returns:
        list: A list containing two loss components:
              - Smooth L1 loss for the velocity mismatch between the left and right states.
              - Smooth L1 loss for the logarithmic pressure mismatch between the predicted and true values.
    """
    out  = model(prims)
    
    rhoL = torch.exp(prims[:,0,0])
    rhoR = torch.exp(prims[:,0,1])
    pL = torch.exp(prims[:,1,0])
    pR = torch.exp(prims[:,1,1])
    vL = prims[:,2,0]
    vR = prims[:,2,1]
    
    # out here is exponentiated, p* >= max(pL,pR) for consistency with double shock configuration
    xi_out = out.squeeze(-1)
    mapping = Compactification(method="affine", a=pR, b=pL)
    press_out = mapping.xi_to_x(xi_out)
    
    _, _, _, _, vstarL_pred, _ = get_vel_raref(press_out,rhoL,pL,vL,-1,gamma)
    _, _, _, _, vstarR_pred, _ = get_vel_shock(press_out,rhoR,pR,vR,+1, gamma)

    return [F.smooth_l1_loss(vstarL_pred, vstarR_pred, beta=1.0, reduction='mean'), F.smooth_l1_loss(torch.log(press_out), torch.log(pstar_true), beta=1.0, reduction='mean')]

def compute_loss__d_raref(model, prims, pstar_true, gamma):
    """
    Computes the loss for a model predicting the pressure in a Riemann problem 
    with rarefaction waves. The loss consists of two components:
    1. The smooth L1 loss between the predicted velocities on the left and right sides.
    2. The smooth L1 loss between the logarithm of the predicted pressure and the true pressure.
    Args:
        model (torch.nn.Module): The neural network model used to predict the output.
        prims (torch.Tensor): A tensor containing primitive variables. 
                              Shape: [batch_size, 3, 2], where:
                              - prims[:, 0, :] contains log(density) for left and right states.
                              - prims[:, 1, :] contains log(pressure) for left and right states.
                              - prims[:, 2, :] contains velocity for left and right states.
        pstar_true (torch.Tensor): The true pressure values (p*). Shape: [batch_size].
        gamma (float): The adiabatic index (specific heat ratio).
    Returns:
        list: A list containing two loss components:
              - The first element is the smooth L1 loss between the predicted velocities 
                on the left and right sides of the rarefaction wave.
              - The second element is the smooth L1 loss between the logarithm of the 
                predicted pressure and the logarithm of the true pressure.
    """
    out  = model(prims)
    
    rhoL = torch.exp(prims[:,0,0])
    rhoR = torch.exp(prims[:,0,1])
    pL = torch.exp(prims[:,1,0])
    pR = torch.exp(prims[:,1,1])
    vL = prims[:,2,0]
    vR = prims[:,2,1]
    
    # out here is exponentiated, p* >= max(pL,pR) for consistency with double shock configuration
    xi_out = out.squeeze(-1)
    mapping = Compactification(method="affine", a=torch.zeros_like(pL,device=pL.device,dtype=pL.dtype), b=torch.min(pL,pR))
    
    press_out = mapping.xi_to_x(xi_out)
    
    _, _, _, _, vstarL_pred, _ = get_vel_raref(press_out,rhoL,pL,vL,-1,gamma)
    _, _, _, _, vstarR_pred, _ = get_vel_raref(press_out,rhoR,pR,vR,+1, gamma)
    

    return [F.smooth_l1_loss(vstarL_pred, vstarR_pred, beta=1.0, reduction='mean'), F.smooth_l1_loss(torch.log(press_out), torch.log(pstar_true), beta=1.0, reduction='mean')]


def compute_loss__raref_solver(model, state, cs_true, sign, gamma):
    """
    Computes the loss for a rarefaction solver using a neural network model.

    Args:
        model (torch.nn.Module): The neural network model used to predict xi values.
        state (torch.Tensor): Input tensor representing the state variables. 
                              Shape: [batch_size, 3], where the columns represent 
                              log-density (log(rho)), log-pressure (log(press)), and velocity (vel).
        cs_true (torch.Tensor): Ground truth speed of sound values. Shape: [batch_size].
        sign (float): Sign parameter indicating the direction of the rarefaction wave.
        gamma (float): Specific heat ratio (adiabatic index) of the gas.

    Returns:
        list: A list containing two loss values:
            - loss_root (torch.Tensor): Smooth L1 loss between predicted and true speed of sound.
            - loss_func (torch.Tensor): Smooth L1 loss for the rarefaction wave equation residual.
    """
    batch = cs_true.size(0)
    rho   = torch.exp(state[:,0])
    press = torch.exp(state[:,1])
    vel   = state[:,2]
    csa   = get_csnd(rho,press,gamma)
    xi_pred = model(state).squeeze(-1)
    mapping = Compactification(method="affine", a=torch.zeros(batch,device=cs_true.device,dtype=cs_true.dtype), b=csa)
    cs_pred = mapping.xi_to_x(xi_pred)
    loss_root = F.smooth_l1_loss(cs_pred,cs_true)
    loss_func = F.smooth_l1_loss(raref_rhs(cs_pred,rho,press,vel,sign,gamma), torch.zeros_like(cs_true,device=cs_true.device,dtype=cs_true.dtype))
    return [loss_func, loss_root]
