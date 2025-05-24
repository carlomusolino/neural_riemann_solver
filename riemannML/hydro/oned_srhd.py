# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torch.nn.functional as F
import numpy as np

from riemannML.exact.hybrid_eos import hybrid_eos
from riemannML.utilities.rootfinding import bisection_solver

from .ppm import shock_indicator

from contextlib import contextmanager

import time

RHO = 0 
PRESS = 1 
VEL = 2
EPS = 3
TEMP =4 

DENS = 0 
TAU  = 1
S    = 2 

import torch 

class RusanovSolver(nn.Module):
    """
    A PyTorch module implementing the Rusanov (or local Lax-Friedrichs) solver for 
    solving hyperbolic partial differential equations. This solver computes the 
    numerical flux at cell interfaces using the Rusanov flux formula.
    Methods:
        forward(P, U, F, c):
            Computes the Rusanov flux for the given input parameters.
    Args:
        P (torch.Tensor): A tensor representing primitive variables (not used in the current implementation).
        U (torch.Tensor): A tensor of conserved variables with shape (batch_size, num_vars, num_cells).
        F (torch.Tensor): A tensor of fluxes with shape (batch_size, num_vars, num_cells).
        c (torch.Tensor): A tensor of maximum signal speeds with shape (batch_size, num_cells).
    Returns:
        torch.Tensor: The computed Rusanov flux with shape (batch_size, num_vars, num_interfaces).
    """
    
    def __init__(self):
        super().__init__() 
        
    def forward(self, P, U ,F, c):
        return 0.5 * ( F[:,:,0] + F[:,:,1] - c[:,None] * (U[:,:,1] - U[:,:,0]) ) 
    
class SRHD1D(nn.Module):
    """
    SRHD1D: Special Relativistic Hydrodynamics in 1D
    This class implements a 1D solver for special relativistic hydrodynamics (SRHD) 
    using PyTorch. It includes methods for boundary condition application, 
    reconstruction, flux computation, and time-stepping.
    Attributes:
        ngz (int): Number of ghost zones.
        nx (int): Number of cells in the computational domain.
        xmin (float): Minimum x-coordinate of the domain.
        xmax (float): Maximum x-coordinate of the domain.
        h (float): Cell width.
        x (torch.Tensor): Cell-centered coordinates.
        eos (object): Equation of state object.
        rho, press, vel, eps, temp, zvec (torch.Tensor): Primitive variables.
        D, tau, S (torch.Tensor): Conserved variables.
        fluxes (torch.Tensor): Fluxes at cell interfaces.
        u_l, u_r (torch.Tensor): Reconstructed primitive variables at left and right interfaces.
        c_l, c_r (torch.Tensor): Conserved variables at left and right interfaces.
        f_l, f_r (torch.Tensor): Fluxes at left and right interfaces.
        gz_mask (torch.Tensor): Mask for ghost zones.
        interior_mask (torch.Tensor): Mask for interior cells.
        device (torch.device): Device for computation (CPU or GPU).
        dtype (torch.dtype): Data type for tensors.
        limiter (callable): Limiter function for slope-limited reconstruction.
        recon (str): Reconstruction method ("slope_limited", "godunov", "weno").
        riemann_solver (callable): Riemann solver function.
        rho_atm, temp_atm (float): Atmospheric density and temperature.
        press_atm, eps_atm (float): Atmospheric pressure and specific internal energy.
        _timers_on (bool): Flag to enable/disable timers.
        _timers (dict): Dictionary to store timing information.
        _calls (dict): Dictionary to store call counts for timed methods.
        _on_gpu (bool): Flag indicating if computation is on GPU.
        _bc_type (str): Boundary condition type ("outgoing", "reflect", "periodic").
    Methods:
        timers_on(): Enable timers for performance measurement.
        timers_off(): Disable timers for performance measurement.
        __isfinite(u): Check if all elements in tensor `u` are finite.
        _apply_bcs_transparent(u): Apply transparent boundary conditions.
        _apply_bcs_reflect(u): Apply reflective boundary conditions.
        _apply_bcs_periodic(u): Apply periodic boundary conditions.
        _apply_bcs(u): Apply boundary conditions based on `_bc_type`.
        _start_timer(): Start a timer for performance measurement.
        _stop_timer(start, end, name): Stop a timer and record elapsed time.
        _timer(name): Context manager for timing a block of code.
        _conservs_to_prims(u): Convert conserved variables to primitive variables.
        _conservs_to_prims_poly(u): Convert conserved variables to primitive variables using a polynomial method.
        _conservs_to_prims_analytic(u): Convert conserved variables to primitive variables using an analytic method.
        _W__z(z): Compute Lorentz factor W from z.
        _weno_recon(u, idx): Perform WENO reconstruction for variable `u`.
        _slope_limited_recon(u, idx): Perform slope-limited reconstruction for variable `u`.
        _zeroth_order_recon(u, idx): Perform zeroth-order (Godunov) reconstruction for variable `u`.
        _compute_cp_cm(v, cs2, W): Compute the left and right characteristic speeds.
        reconstruct(): Perform reconstruction of primitive variables.
        getfluxes(): Compute fluxes at cell interfaces using the Riemann solver.
        forward(t, u): Compute the time derivative of conserved variables `u` at time `t`.
    """
    
    
    def __init__(self, domain, ncells, nghost, eos, riemann_solver, recon_method, limiter, use_flux_limiter, bc_type, device, dtype):
        
        # Initialize super class 
        super().__init__() 
        
        # Domain
        self.ngz = nghost 
        self.nx  = ncells 
         
        self.xmin = domain[0]
        self.xmax = domain[1]
        
        self.h   = (domain[1]-domain[0])/(ncells)
        self.x   = torch.linspace(
            domain[0] + ( 1/2 - nghost ) * self.h, 
            domain[1] + ( nghost - 1/2) * self.h, 
            ncells + 2*nghost, dtype=dtype
            ).to(device)
        
        # EOS 
        self.eos = eos 
        
        # Primitives
        self.rho = torch.zeros_like(self.x, dtype=dtype).to(device)
        self.press = torch.zeros_like(self.x, dtype=dtype).to(device)
        self.vel  = torch.zeros_like(self.x, dtype=dtype).to(device)
        self.eps  = torch.zeros_like(self.x, dtype=dtype).to(device)
        self.temp = torch.zeros_like(self.x, dtype=dtype).to(device)
        self.zvec = torch.zeros_like(self.x, dtype=dtype).to(device)
        
        # Conserved 
        self.D = torch.zeros_like(self.x, dtype=dtype).to(device)
        self.tau = torch.zeros_like(self.x, dtype=dtype).to(device)
        self.S = torch.zeros_like(self.x, dtype=dtype).to(device)
        
        # Fluxes 
        self.fluxes = torch.zeros((ncells+1, 3), dtype=dtype).to(device)
        
        # Reconstructed vars 
        self.u_l = torch.zeros((ncells+1, 5), dtype=dtype).to(device)
        self.u_r = torch.zeros((ncells+1, 5), dtype=dtype).to(device)
        
        self.c_l = torch.zeros((ncells+1,3), dtype=dtype).to(device)
        self.c_r = torch.zeros((ncells+1,3), dtype=dtype).to(device)
        
        self.f_l = torch.zeros((ncells+1,3), dtype=dtype).to(device)
        self.f_r = torch.zeros((ncells+1,3), dtype=dtype).to(device)
        
        # Masks 
        self.gz_mask       = ( self.x < domain[0]) | ( self.x > domain[1])
        self.interior_mask = ~self.gz_mask 
        
        # Helpers 
        self.device = device
        self.dtype  = dtype 
        
        # Stuff 
        self.limiter = limiter 
        self.recon   = recon_method
        
        if (self.recon == "slope_limited") and (self.limiter is None): 
            raise ValueError("When using slope-limited recon, a valid limiter must be provided")
        
        self.riemann_solver = riemann_solver
        
        self.rho_atm = 1e-14 
        self.temp_atm = 0 
        self.press_atm, self.eps_atm  = self.eos.press_eps__temp_rho(
            torch.tensor(self.temp_atm, device=device, dtype=dtype),
            torch.tensor(self.rho_atm, device=device, dtype=dtype) )
        
        self._timers_on = False 
        
        self._timers = {
            "c2p" : 0,
            "recon": 0,
            "flux" : 0,
            "riemann": 0
        }
        
        self._calls = {
            "c2p" : 0,
            "recon": 0,
            "flux" : 0,
            "riemann": 0
        }
        
        self._on_gpu = ( device != torch.device('cpu') )
        
        self._bc_type = bc_type 
        
        self._use_flux_limiter = use_flux_limiter
        if self._use_flux_limiter and (self.ngz < 3):
            raise ValueError("Flux limiter needs 3 ghostzones") 
        
        # Fixed for now, values come from Marti Mueller 96 JCP 
        self._eps_flattening = 0.9 
        self._omega_1_flattening = 0.52 
        self._omega_2_flattening = 10.0 
    
    def get_timers(self):
        return self._timers 
    def get_counters(self):
        return self._calls
    def timers_on(self):
        self._timers_on = True
    
    def timers_off(self):
        self._timers_on = False
    
    def __isfinite(self, u):
        return torch.isfinite(u).all()
    
    def _apply_bcs_transparent(self, u):
        u[0:self.ngz, :] = u[self.ngz, :]
        u[-self.ngz:, :] = u[-self.ngz-1, :] 
        
    def _apply_bcs_reflect(self, u):
        u[0:self.ngz, :] = u[self.ngz, :]
        u[-self.ngz:, :] = u[-self.ngz-1, :]
        u[-self.ngz:, 2] *= -1 
        u[0:self.ngz, 2] *= -1
    
    def _apply_bcs_periodic(self, u):
        u[0:self.ngz, :] = u[-self.ngz-1:-1, :]
        u[-self.ngz:, :] = u[1:self.ngz+1, :]
    
    def _apply_bcs(self, u):
        if self._bc_type == "outgoing":
            self._apply_bcs_transparent(u)
        elif self._bc_type == "reflect":
            self._apply_bcs_reflect(u)
        elif self._bc_type == "periodic":
            self._apply_bcs_periodic(u)
        else:
            raise ValueError("Boundary condition type not implemented")
    
    def _start_timer(self):
        if self._timers_on:
            if self._on_gpu:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                return start, end
            else:
                start = time.time()
                return start, None
        else:
            return None, None
    
    def _stop_timer(self, start, end, name):
        if self._timers_on:
            if self._on_gpu:
                end.record()
                torch.cuda.synchronize()
                self._timers[name] += start.elapsed_time(end)
                self._calls[name] += 1
            else:
                end = time.time()
                self._timers[name] += end - start
                self._calls[name] += 1
        else:
            return 
    
    @contextmanager 
    def _timer(self,name):
        start, end = self._start_timer()
        try: yield
        finally: self._stop_timer(start,end,name)
    
    def _conservs_to_prims(self, u):
        # Convert to double no matter what
        Snorm = u[:,2].abs() 
        u[:,2] = torch.where( Snorm > u[:,0] + u[:,1], 0.999 * (u[:,0] + u[:,1]) * torch.sign(u[:,2]), u[:,2])
        t = u[:,0].double()
        q = u[:,1].double() / t 
        r = u[:,2].abs().double() / t 
        k = r / ( 1 + q )
        
        Wtilde = lambda z: torch.sqrt(1+z**2)
        rhotilde = lambda z: t/Wtilde(z)
        epstilde = lambda z: Wtilde(z) * q - r * z + z**2/(1+Wtilde(z))
        ptilde = lambda z: rhotilde(z) * epstilde(z) * (self.eos.gamma_th - 1 )
        atilde = lambda z: ptilde(z)/(rhotilde(z)*(1+epstilde(z)))
        htilde = lambda z: (1+epstilde(z)) * (1+atilde(z))
        
        f = lambda z: z - r/htilde(z)
        
        zm = 0.5 * k  / torch.sqrt(1-(0.5*k)**2)
        zp = 1e-6 + k/torch.sqrt(1-k**2)
        
        zeta, mask = bisection_solver(f,zm,zp,1e-15)
        W = Wtilde(zeta)
        self.rho = (t/W ).to(self.dtype)
        self.eps = epstilde(zeta).to(self.dtype)
        self.press = ptilde(zeta).to(self.dtype)
        h = htilde(zeta)
        self.vel  = u[:,2]/t/h/W 
        self.zvec  = (W * self.vel).to(self.dtype)    
        self.temp = (self.eps  * (self.eos.gamma_th-1)).to(self.dtype)
        
        atm_mask = (self.rho < (1 + 1e-3) * self.rho_atm) | ~mask 
        self.rho = torch.where(atm_mask, torch.full_like(self.rho,self.rho_atm, device=u.device,dtype=u.dtype), self.rho)
        self.eps = torch.where(atm_mask, torch.full_like(self.rho,self.eps_atm, device=u.device,dtype=u.dtype), self.eps)
        self.press = torch.where(atm_mask, torch.full_like(self.rho,self.press_atm, device=u.device,dtype=u.dtype), self.press)
        self.vel = torch.where(atm_mask, torch.zeros_like(self.rho, device=u.device,dtype=u.dtype), self.vel)
        self.zvec = torch.where(atm_mask, torch.zeros_like(self.rho, device=u.device,dtype=u.dtype), self.zvec)
        self.temp = torch.where(atm_mask, torch.zeros_like(self.rho, device=u.device,dtype=u.dtype), self.temp)
    
    def _conservs_to_prims_poly(self, u):
        # Convert to double no matter what
        D = u[:,0].double()
        E = u[:,1].double() + D 
        M = u[:,2].double().abs()
        v0_mask = (M < 1e-10)
        gamma = self.eos.gamma_th
        if not torch.all(v0_mask):
            E_ = E[~v0_mask]
            R_ = D[~v0_mask]
            M_ = M[~v0_mask]
            
            f0 = lambda v: ( gamma * v * ( E_ - M_ * v ) - M_ * ( 1 - v**2 ) )**2 - ( 1 - v**2 ) * v**2 * (gamma-1)**2 * R_**2
            
            vL = 1/(2*M_*(gamma-1)) * ( gamma * E_ - torch.sqrt( (gamma*E_)**2 - 4 * (gamma-1) * M_**2 )  )
            vu = torch.min(torch.ones_like(E_, device=self.device, dtype=torch.float64), M_/E_ + 1e-6)
            
            v_, _ = bisection_solver(f0, vL, vu, tol=1e-9, max_iter=100)
            self.vel[~v0_mask] = v_
        
        self.vel[v0_mask] = 0 
        
        W = 1./torch.sqrt(1 - self.vel**2)
        self.rho = (D / W).to(self.dtype)
        
        vel_sign = torch.where( v0_mask, torch.zeros_like(M, device=self.device, dtype=self.dtype), torch.sign(u[:,2]))
        self.vel = (self.vel * vel_sign).to(self.dtype)

        self.press = ((gamma - 1) * ( E - M * self.vel - self.rho )).to(self.dtype)
        self.eps  = (self.press / self.rho / (gamma - 1)).to(self.dtype)
        self.temp = (self.eps  * (gamma-1)).to(self.dtype)
        self.zvec = W * self.vel 
    
    def _conservs_to_prims_analytic(self, u):
        # Convert to double no matter what
        D = u[:,0].double()
        E = u[:,1].double() + D 
        M = u[:,2].double().abs()

        # If M is zero, v is zero as well
        v0_mask = (M < 1e-10)
        
        gamma = self.eos.gamma_th
        
        denom =  (gamma-1)**2 * ( M**2 + D**2 ) 
        b1 = - 2 * gamma * ( gamma - 1) * M * E / denom
        b2 = (gamma**2*E**2 + 2*(gamma-1)*M**2 - (gamma-1)**2*D**2 ) / denom
        b3 = - 2 * gamma * M * E / denom 
        b4 = M**2 / denom 

        a1 = - b2 
        a2 = b1*b3 - 4 * b4 
        a3 = 4 * b2*b4 - b3**2 - b1**2 * b4 
        
        R = (9*a1*a2 - 27 * a3 - 2 * a1**3 ) / 54 
        S = (3*a2 - a1**2) / 9 
        T =  R**2 + S**3
        
        x1 = (R + torch.sqrt(T))**(1/3) + (R - torch.sqrt(T))**(1/3) - a1 / 3
        
        B = 0.5 * ( b1 + torch.sqrt(b1**2 - 4 * b2 + 4 * x1) )
        C = 0.5 * ( x1 - torch.sqrt(x1**2 - 4 * b4 ) ) 
        
        vel = torch.where(v0_mask, torch.zeros_like(B, device=self.device), (- B + torch.sqrt(B**2 - 4 * C)) * 0.5 )
  
        W =   1./torch.sqrt(1 - vel**2) 
        self.rho = (D / W).to(self.dtype)
        vel_sign = torch.where( v0_mask, torch.zeros_like(M, device=self.device, dtype=self.dtype), torch.sign(u[:,2]))
        
        self.vel = (vel * vel_sign).to(self.dtype)

        self.press = ((gamma - 1) * ( E - M * self.vel - self.rho )).to(self.dtype)
        self.eps  = (self.press / self.rho / (gamma - 1)).to(self.dtype)
        self.temp = (self.eps  * (gamma-1)).to(self.dtype)
        self.zvec = W * self.vel 
        
    
    def _W__z(self,z):
        return torch.sqrt(z**2 + 1)
    
    def _weno_recon(self,u,idx):
        #------------------------------------------------------------------#
        d0 = 1./3. 
        d1 = 2./3.
        #------------------------------------------------------------------#
        ngz = self.ngz 
        nx = self.nx
    
        u0  = u[ngz:nx+ngz+1]
        um1 = u[ngz-1:nx+ngz]
        um2 = u[ngz-2:nx+ngz-1]
        up1 = u[ngz+1:nx+ngz+2]
        
        beta0 = (um1-um2)**2 
        beta1 = (u0-um1)**2 
        beta2 = (up1-u0)**2
        
        alphaL0 = d0 / (1e-15 + beta0)**2
        alphaL1 = d1 / (1e-15 + beta1)**2
        
        alphaR0 = d0 / (1e-15 + beta2)**2
        alphaR1 = d1 / (1e-15 + beta1)**2
        
        wL = 1. / ( alphaL0 + alphaL1)
        wR = 1. / ( alphaR0 + alphaR1)
        
        self.u_r[:,idx] = 0.5 * wR * (alphaR0 * ( 3*u0 - up1) + alphaR1 * ( um1 + u0 ) )
        self.u_l[:,idx] = 0.5 * wL * (alphaL1 * ( u0 + um1) + alphaL0 * ( 3*um1 - um2 ) )        
        #------------------------------------------------------------------#
    
    def _slope_limited_recon(self, u, idx):
        #------------------------------------------------------------------#
        ngz = self.ngz 
        nx = self.nx
        # U[i] - U[i-1]
        sL = u[ngz:nx+ngz+1] - u[ngz-1:nx+ngz]
        # U[i+1] - U[i]
        sR = u[ngz+1:nx+ngz+2] - u[ngz:nx+ngz+1]
        # Get UR: U[i] - 0.5 * limiter
        self.u_r[:, idx] = u[ngz:nx+ngz+1] - 0.5 * self.limiter(sL, sR)
        #------------------------------------------------------------------#
        #------------------------------------------------------------------#
        # U[i-1] - U[i-2]
        sL = u[ngz-1:nx+ngz] - u[ngz-2:nx+ngz-1]
        # U[i] - U[i-1]
        sR = u[ngz:nx+ngz+1] - u[ngz-1:nx+ngz]
        # Get UL: U[i-1] + 0.5 * limiter
        self.u_l[:, idx] = u[ngz-1:nx+ngz] + 0.5 * self.limiter(sL, sR)
        #------------------------------------------------------------------#
    
    def _zeroth_order_recon(self,u,idx):
        #------------------------------------------------------------------#
        ngz = self.ngz 
        nx = self.nx
        # UL[i] = U[i-1+1/2] = U[i-1]
        self.u_l[:, idx] = u[ngz-1:nx+ngz]
        # UR[i] = U[i-1/2] = U[i]
        self.u_r[:, idx] = u[ngz:nx+ngz+1]
        
    def _compute_cp_cm_old(self, v, cs2, W):
        u0_sq = W**2 
        A = u0_sq * (1 - cs2) + cs2 
        B = 2 * ( cs2 - u0_sq * v * ( 1 - cs2 ) ) 
        C = u0_sq * v**2 * (1-cs2) - cs2 
        det = torch.sqrt(torch.max(torch.zeros_like(B, device=self.device, dtype=self.dtype),B**2 - 4 * A * C))
        c1 = 0.5 * ( -B + det ) / A
        c2 = 0.5 * ( -B - det ) / A
        return torch.max(c1,c2), torch.min(c1,c2)
    
    def _compute_cp_cm(self, v, cs2, W):
        sigma_s = cs2 / (W**2 * (1-cs2))
        det = torch.sqrt(torch.max(torch.zeros_like(v, device=self.device, dtype=self.dtype),sigma_s*(1-v**2+sigma_s)))
        c1 = ( v + det ) / (1+sigma_s)
        c2 = ( v - det ) / (1+sigma_s)
        return c1, c2
    
    def reconstruct(self):
        if self.recon == "slope_limited":
            self._slope_limited_recon(self.rho, RHO)
            self._slope_limited_recon(self.temp, TEMP)
            self._slope_limited_recon(self.zvec, VEL)
        elif self.recon == "godunov":
            self._zeroth_order_recon(self.rho, RHO)
            self._zeroth_order_recon(self.temp, TEMP)
            self._zeroth_order_recon(self.zvec, VEL)
        elif self.recon == "weno":
            self._weno_recon(self.rho, RHO)
            self._weno_recon(self.temp, TEMP)
            self._weno_recon(self.zvec, VEL)
        else:
            raise ValueError("Reconstruction method not implemented")
    
    def getfluxes(self, u):
        
        # Compute W
        Wl = self._W__z(self.u_l[:,VEL]) 
        Wr = self._W__z(self.u_r[:,VEL]) # Shape: [ncells+1]
        
        # Compute pressure and eps from reconstructed rho and T
        self.u_l[:,PRESS], self.u_l[:,EPS] = self.eos.press_eps__temp_rho(self.u_l[:,TEMP], self.u_l[:,RHO])
        self.u_r[:,PRESS], self.u_r[:,EPS] = self.eos.press_eps__temp_rho(self.u_r[:,TEMP], self.u_r[:,RHO])
        
        # Convert zvec to v in reconstructed vars
        self.u_l[:,VEL] /= Wl
        self.u_r[:,VEL] /= Wr 
        
        # Get sound speed 
        cs2l = self.eos.cs2__eps_rho(self.u_l[:,EPS], self.u_l[:,RHO])
        cs2r = self.eos.cs2__eps_rho(self.u_r[:,EPS], self.u_r[:,RHO])
        
        # Compute c+ and c-
        cpl, cml = self._compute_cp_cm(self.u_l[:,VEL], cs2l, Wl)
        cpr, cmr = self._compute_cp_cm(self.u_r[:,VEL], cs2r, Wr)
        
        # Compute cmax and cmin
        cmax  = torch.max(cpr, cpl)
        cmin  = torch.min(cml, cmr)
        vfloor_mask = (cmax.abs() < 1e-12) & (cmin.abs() < 1e-12)
        cmax = torch.where( vfloor_mask, torch.ones_like(cmax, device=self.device, dtype=self.dtype), cmax)
        cmin = torch.where( vfloor_mask, -torch.ones_like(cmin, device=self.device, dtype=self.dtype), cmin)
        
        # Compute C_L and C_R 
        self.c_l[:,DENS] = Wl * self.u_l[:,RHO]
        self.c_r[:,DENS] = Wr * self.u_r[:,RHO]
        
        h_l = 1 + self.u_l[:,EPS] + self.u_l[:,PRESS]/self.u_l[:,RHO]
        h_r = 1 + self.u_r[:,EPS] + self.u_r[:,PRESS]/self.u_r[:,RHO]
        
        self.c_l[:,TAU] = self.c_l[:,DENS] * ( Wl * h_l - 1 ) - self.u_l[:,PRESS]
        self.c_r[:,TAU] = self.c_r[:,DENS] * ( Wr * h_r - 1 ) - self.u_r[:,PRESS]
        
        self.c_l[:,S] = Wl**2 * self.u_l[:,RHO] * h_l * self.u_l[:,VEL] 
        self.c_r[:,S] = Wr**2 * self.u_r[:,RHO] * h_r * self.u_r[:,VEL]
        
        # Compute F_L and F_R
        self.f_l[:,DENS] = self.c_l[:,DENS] * self.u_l[:,VEL]
        self.f_r[:,DENS] = self.c_r[:,DENS] * self.u_r[:,VEL]
        
        self.f_l[:,TAU]  = self.c_l[:,DENS] * ( Wl * h_l - 1 ) * self.u_l[:,VEL]
        self.f_r[:,TAU]  = self.c_r[:,DENS] * ( Wr * h_r - 1 ) * self.u_r[:,VEL]
        
        self.f_l[:,S]    = self.c_l[:,S] * self.u_l[:,VEL] + self.u_l[:,PRESS]
        self.f_r[:,S]    = self.c_r[:,S] * self.u_r[:,VEL] + self.u_r[:,PRESS]
        
        # Compute fluxes
        with self._timer('riemann'):
            self.fluxes = self.riemann_solver(
                torch.cat((self.u_l.unsqueeze(-1), self.u_r.unsqueeze(-1)), dim=2),
                torch.cat((self.c_l.unsqueeze(-1), self.c_r.unsqueeze(-1)), dim=2),
                torch.cat((self.f_l.unsqueeze(-1), self.f_r.unsqueeze(-1)), dim=2),
                cmax, cmin
            )
        
        if self._use_flux_limiter:
            # Shock indicator
            theta = shock_indicator(
                self.press,
                self.vel, 
                self.nx, 
                self.ngz, 
                self._eps_flattening, 
                self._omega_1_flattening, 
                self._omega_2_flattening 
            )
            
            # Godunov recon for LO flux
            c_l_LO = u[self.ngz-1:self.ngz-1+self.nx+1]
            c_r_LO = u[self.ngz:self.ngz+self.nx+1]
            
            # Low order flux
            F_LO = RusanovSolver()(
                    torch.cat((self.u_l.unsqueeze(-1), self.u_r.unsqueeze(-1)), dim=2),
                    torch.cat((c_l_LO.unsqueeze(-1), c_r_LO.unsqueeze(-1)), dim=2),
                    torch.cat((self.f_l.unsqueeze(-1), self.f_r.unsqueeze(-1)), dim=2),
                    torch.max(cmax.abs(), cmin.abs())
                )
            
            # Apply flux limiter 
            self.fluxes = theta[:,None] * F_LO + (1-theta[:,None]) * self.fluxes 
            
            # Save the indicator 
            self.theta = theta
            
            self.cmax = cmax 
            self.cmin = cmin
        
    def forward(self, t, u):
        
        # First we apply boundary conditions 
        self._apply_bcs(u)
        
        # Then we do the c2p 
        with self._timer('c2p'):
            self._conservs_to_prims(u)
        
        # Then we reconstruct
        with self._timer('recon'):
            self.reconstruct()
        
        # Then we compute fluxes
        with self._timer('flux'):
            self.getfluxes(u) 
        
        # Finally we assemble the RHS 
        rhs = torch.zeros_like(u, device=self.device, dtype=self.dtype)
        rhs[self.interior_mask, :] = -(self.fluxes[1:, :] - self.fluxes[:-1, :]) / self.h
        rhs[self.gz_mask      , :] = 0 
        
        # And return it! 
        return rhs       
        