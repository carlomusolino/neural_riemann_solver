import torch 
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import newton
from riemannML.utilities.rootfinding import newton_solver, brent_solver, bisection_solver 

DSHOCK     = 0 
DRAREF     = 1 
RAREFSHOCK = 2 # Shock to the right (-->) raref to the left (<--)
SHOCKRAREF = 3 # Raref to the right (-->) shock to the left (<--)

def get_eps(rho,p,gamma):
    return p/(gamma-1.)/rho 
def get_h(rho,p,gamma):
    eps = get_eps(rho,p,gamma)
    return 1. + eps + p/rho
def get_csnd(rho,p,gamma):
    h = get_h(rho,p,gamma)
    return torch.sqrt(gamma*p/rho/h)
def get_w(vel):
    return 1./torch.sqrt(1-vel**2)


def get_vel(p, rhoa, pa, vela, sign, gamma, verbose=False):
    '''
    Compute the flow velocity behind a rarefaction or shock in terms of 
    post-wave pressure for a given state ahead of the wave.
    
    p -> post-wave pressure (tensor)
    rhoa -> density ahead of the wave (tensor)
    pa -> pressure ahead of the wave (tensor)
    vela -> flow velocity ahead of the wave (tensor)
    sign -> +1 for right-going wave, -1 for left-going
    gamma -> adiabatic index (scalar or tensor)
    
    Output: (rho, eps, h, csnd, vel, vshock)
    '''
    
    # Internal energy
    epsa = get_eps(rhoa, pa, gamma)
    
    # Specific enthalpy
    ha = 1. + epsa + pa / rhoa
    
    # Sound speed ahead of the wave
    csa = torch.sqrt(gamma * pa / (rhoa * ha))
    
    # Lorentz factor of the flow
    wa = get_w(vela)

    # Shock case: p > pa
    shock_mask = p > pa

    ### SHOCK SOLUTION
    # Compute shock relations where p > pa
    a_shock = 1 + (gamma - 1) * (pa - p) / (gamma * p)
    b_shock = 1 - a_shock
    c_shock = ha * (pa - p) / rhoa - ha**2

    h_shock = (-b_shock + torch.sqrt(b_shock**2 - 4 * a_shock * c_shock)) / (2 * a_shock)
    rho_shock = gamma * p / ((gamma - 1) * (h_shock - 1))
    eps_shock = p / ((gamma - 1) * rho_shock)
    j_shock = sign * torch.sqrt((p - pa) / (ha / rhoa - h_shock / rho_shock))

    a_vshock = j_shock**2 + (rhoa * wa)**2
    b_vshock = -vela * rhoa**2 * wa**2
    vshock_shock = (-b_vshock + sign * j_shock**2 * torch.sqrt(1 + rhoa**2 / j_shock**2)) / a_vshock

    wshock = get_w(vshock_shock)

    a_vel = wshock * (p - pa) / j_shock + ha * wa * vela
    b_vel = ha * wa + (p - pa) * (wshock * vela / j_shock + 1 / (rhoa * wa))
    vel_shock = a_vel / b_vel
    if( torch.any(shock_mask) and verbose ) :
        print(f"p rhoa pa vela {p} {rhoa} {pa} {vela} {sign}")
        print(f"h_shock {h_shock}")
        print(f"a,b,c shock {a_shock}, {b_shock}, {c_shock}")
        print(f"vshock_shock: {vshock_shock}")
        print(f"In shock computation: a_vel {a_vel} b_vel {b_vel} v_shock {vel_shock}")
    cs_shock = get_csnd(rho_shock, p, gamma)

    ### RAREFACTION SOLUTION
    # Compute rarefaction relations where p <= pa
    k_raref = pa / rhoa**gamma
    rho_raref = (p / k_raref) ** (1.0 / gamma)
    eps_raref = p / ((gamma - 1) * rho_raref)
    cs_raref = torch.sqrt(gamma * p / (rho_raref + gamma * p / (gamma - 1)))

    sqgl1 = np.sqrt(gamma - 1)
    a_raref = (1 + vela) / (1 - vela) * ((sqgl1 + csa) / (sqgl1 - csa) * (sqgl1 - cs_raref) / (sqgl1 + cs_raref)) ** (-sign * 2 / sqgl1)
    vel_raref = (a_raref - 1) / (a_raref + 1)

    h_raref = get_h(rho_raref, p, gamma)

    # Combine shock and rarefaction solutions using `torch.where`
    rho = torch.where(shock_mask, rho_shock, rho_raref)
    eps = torch.where(shock_mask, eps_shock, eps_raref)
    h = torch.where(shock_mask, h_shock, h_raref)
    cs = torch.where(shock_mask, cs_shock, cs_raref)
    vel = torch.where(shock_mask, vel_shock, vel_raref)
    vshock = torch.where(shock_mask, vshock_shock, torch.zeros_like(p, device=p.device))  # No shock velocity in rarefaction

    return rho, eps, h, cs, vel, vshock

def get_vel_shock(p, rhoa, pa, vela, sign, gamma, verbose=False):
    dtype = p.dtype
    p,rhoa,pa,vela = map(lambda x: x.double(), [p,rhoa,pa,vela ])
    tiny = 1e-42
    # Internal energy
    epsa = get_eps(rhoa, pa, gamma)
    
    # Specific enthalpy
    ha = 1. + epsa + pa / rhoa
    
    # Lorentz factor of the flow
    wa = get_w(vela)

    ### SHOCK SOLUTION
    # Compute shock relations where p > pa
    a_shock = 1 + (gamma - 1) * (pa - p) / (gamma * p)
    b_shock = 1 - a_shock
    c_shock = ha * (pa - p) / rhoa - ha**2
    # Clamp to ensure we're not fcked by roundoff 
    h_shock = torch.clamp((-b_shock + torch.sqrt(b_shock**2 - 4 * a_shock * c_shock)) / (2 * a_shock), min=1.0 + tiny)
    rho_shock = gamma * p / ((gamma - 1) * (h_shock - 1))
    eps_shock = p / ((gamma - 1) * rho_shock)

    j_shock = sign * torch.sqrt((p - pa) / (ha / rhoa - h_shock / rho_shock +tiny))
    if verbose:
        print(f" ha {ha} rhoa {rhoa} ha/rhoa {ha/rhoa}")
        print(f" hshock {h_shock} rhoshock {rho_shock} hs/rhos {h_shock/rho_shock}")
    
        print(f"j_shock {j_shock}, rho_shock {rho_shock}, p-pa {p-pa+tiny} ha / rhoa - h_shock / rho_shock {ha / rhoa - h_shock / rho_shock}")
    
    a_vshock = j_shock**2 + (rhoa * wa)**2
    b_vshock = -vela * rhoa**2 * wa**2
          
    # Like above, we clamp! 
    vshock_shock = torch.clamp((-b_vshock + sign * j_shock**2 * torch.sqrt(1 + rhoa**2 / ( tiny+ j_shock**2) )) / a_vshock, min=-(1-1e-10), max=(1-1e-10))

    wshock = get_w(vshock_shock)
    if verbose:
        print(f"a_vshock {a_vshock}, b_vshock {b_vshock}")
        print(f"v_shock {vshock_shock}, w_shock {wshock}")
    
    a_vel = wshock * (p-pa) / (j_shock+tiny) + ha * wa * vela
    b_vel = ha * wa + (p-pa) * (wshock * vela / (j_shock+tiny) + 1 / (rhoa * wa))
    vel_shock = a_vel / b_vel

    cs_shock = get_csnd(rho_shock, p, gamma)
    if verbose:
        print(f"a_vel {a_vel}, b_vel {b_vel}")
        print(f"v_shock {vshock_shock}, w_shock {wshock}")
    #print(f"a_vel {a_vel}, b_vel {b_vel}")
    
    return rho_shock.to(dtype), eps_shock.to(dtype), h_shock.to(dtype), cs_shock.to(dtype), vel_shock.to(dtype), vshock_shock.to(dtype)


def get_vel_raref(p, rhoa, pa, vela, sign, gamma):
    dtype = p.dtype
    p,rhoa,pa,vela = map(lambda x: x.double(), [p,rhoa,pa,vela ])
    # Internal energy
    epsa = get_eps(rhoa, pa, gamma)
    
    # Specific enthalpy
    ha = 1. + epsa + pa / rhoa
    
    # Sound speed ahead of the wave
    csa = torch.sqrt(gamma * pa / (rhoa * ha))
    
    # Lorentz factor of the flow
    wa = get_w(vela)
    ### RAREFACTION SOLUTION
    # Compute rarefaction relations where p <= pa
    k_raref = pa / rhoa**gamma
    rho_raref = (p / k_raref) ** (1.0 / gamma)
    eps_raref = p / ((gamma - 1) * rho_raref)
    cs_raref = torch.sqrt(gamma * p / (rho_raref + gamma * p / (gamma - 1)))

    sqgl1 = np.sqrt(gamma - 1)
    a_raref = (1 + vela) / (1 - vela) * ((sqgl1 + csa) / (sqgl1 - csa) * (sqgl1 - cs_raref) / (sqgl1 + cs_raref)) ** (-sign * 2 / sqgl1)
    vel_raref = (a_raref - 1) / (a_raref + 1)

    h_raref = get_h(rho_raref, p, gamma)
    return rho_raref.to(dtype), eps_raref.to(dtype), h_raref.to(dtype), cs_raref.to(dtype), vel_raref.to(dtype), torch.zeros_like(rho_raref, dtype=dtype, device=p.device)

def get_p_star(vel_func,prims,gamma,tol=1e-07):
    pL   = 10**prims[:,1,0]
    pR   = 10**prims[:,1,1]
    rhoL = 10**prims[:,0,0]
    rhoR = 10**prims[:,0,1]
    vL   = prims[:,2,0]
    vR   = prims[:,2,1]
    
    def func(p):
        _,_,_,_,v1,_ = vel_func(p,rhoL,pL,vL,-1,gamma)
        _,_,_,_,v2,_ = vel_func(p,rhoR,pR,vR,+1,gamma)
        return v1 - v2 

    # Find the range for the pressure root-find
    pmin = (pR + pL) * 0.5
    pmax = pmin.clone()

    # Vectorized root-finding initialization
    mask = func(pmin) * \
           func(pmax) > 0

    # Iteratively refine pmin and pmax in parallel
    for _ in range(10000):  # Prevent infinite loops
        pmin = torch.where(mask, torch.clamp(0.5 * pmin, min=0), pmin)
        pmax = torch.where(mask, 2*pmax, pmax)

        fmin = func(pmin)
        fmax = func(pmax)

        # Ensure no NaNs appear
        fmin = torch.nan_to_num(fmin, nan=0.0)
        fmax = torch.nan_to_num(fmax, nan=0.0)

        mask = (fmin * fmax) > 0  # True if root is still not bracketed

        if not torch.any(mask):  # Exit early if all cases are bracketed
            break
    if torch.any(mask):
        print("Warning: Some roots are not bracketed!")
    
    return brent_solver(func,pmin,pmax,tol=tol,max_iter=100)

def get_dvel(p, rhol, pl, vell, rhor, pr, velr, gamma, verbose=False):
    '''
    Get difference in flow speed between left and right intermediate states
    given left and right states and pressure in the intermediate state.
    
    Inputs:
    - p: Intermediate pressure (tensor)
    - rhol, pl, vell: Left state (density, pressure, velocity)
    - rhor, pr, velr: Right state (density, pressure, velocity)
    - gamma: Adiabatic index
    
    Output:
    - Difference in velocity between left and right post-wave states
    '''
    
    # Compute post-wave states for left-going and right-going waves
    rhols, epsls, hls, csls, vells, vshockl = get_vel(p, rhol, pl, vell, -1, gamma, verbose)
    rhors, epsrs, hrs, csrs, velrs, vshockr = get_vel(p, rhor, pr, velr, +1, gamma, verbose)
    if verbose: print(f"In dvel: vel1 {vells} vel2 {velrs} dvel {vells-velrs}")
    # Compute velocity difference
    return vells - velrs


def get_p(pmin, pmax, tol, rhol, pl, vell, rhor, pr, velr, gamma, dtype):
    '''
    Finds the pressure in the intermediate state of a Riemann problem using Brent’s method.

    Inputs:
    - pmin, pmax: Initial pressure brackets (tensors)
    - tol: Tolerance for convergence
    - rhol, pl, vell: Left state (density, pressure, velocity)
    - rhor, pr, velr: Right state (density, pressure, velocity)
    - gamma: Adiabatic index

    Output:
    - p_c: Intermediate pressure estimate
    '''
    rhol, pl, vell, rhol, pr, velr, pmin, pmax = map(lambda x: x.double(), [rhol, pl, vell, rhol, pr, velr, pmin, pmax])
    # Define the function to be solved
    def func(p):
        return get_dvel(p, rhol, pl, vell, rhor, pr, velr, gamma)

    # Solve using Brent’s method
    p_c, mask = bisection_solver(func, pmin, pmax, tol)
    
    return p_c.to(dtype), mask



def classify_riemann_problem(rho, press, vel, eos, dtype=torch.float32, tol=1e-7):
    gamma = eos.gamma_th
    #press, eps = eos.press_eps__temp_rho(rho, temp)

    h = get_h(rho, press, gamma)
    cs = get_csnd(rho, press, gamma)
    w = get_w(vel)
    
    # Find the range for the pressure root-find
    pmin = (press[:,0] + press[:,1]) * 0.5
    pmax = pmin.clone()

    # Vectorized root-finding initialization
    mask = get_dvel(pmin, rho[:,0], press[:,0], vel[:,0], rho[:,1], press[:,1], vel[:,1], gamma) * \
           get_dvel(pmax, rho[:,0], press[:,0], vel[:,0], rho[:,1], press[:,1], vel[:,1], gamma) > 0

    # Iteratively refine pmin and pmax in parallel
    while(True):  # Prevent infinite loops
        pmin = torch.where(mask, torch.clamp(0.5 * pmin, min=0), pmin)
        pmax = torch.where(mask, 2*pmax, pmax)

        fmin = get_dvel(pmin, rho[:,0], press[:,0], vel[:,0], rho[:,1], press[:,1], vel[:,1], gamma)
        fmax = get_dvel(pmax, rho[:,0], press[:,0], vel[:,0], rho[:,1], press[:,1], vel[:,1], gamma)

        # Ensure no NaNs appear
        fmin = torch.nan_to_num(fmin, nan=0.0)
        fmax = torch.nan_to_num(fmax, nan=0.0)

        mask = (fmin * fmax) > 0  # True if root is still not bracketed

        if not torch.any(mask):  # Exit early if all cases are bracketed
            break
    print(f"Max {torch.max(pmax)}, Min {torch.min(pmin)}")
    # Solve for press_c using root-finding (parallelized version)
    press_c, convergence_mask = get_p(pmin, pmax, 
                                      tol, 
                                      rho[:,0], press[:,0], vel[:,0], 
                                      rho[:,1], press[:,1], vel[:,1], 
                                      gamma, dtype )

    # Initialize labels as boolean tensors
    labels = torch.zeros((rho.shape[0], 3), dtype=torch.bool, device=rho.device)

    # Classification conditions
    labels[:, DSHOCK] = (press_c >= press[:,0]) & (press_c >= press[:,1])
    labels[:, DRAREF] = (press_c < press[:,0]) & (press_c < press[:,1])
    labels[:, RAREFSHOCK] = (press_c < press[:,0]) & (press_c >= press[:,1]) # Shock to the right raref to the left 
    
    return labels, convergence_mask, press_c

def raref(xi, rhoa, pa, vela, gamma, sign):
    '''
    Compute the flow state in a rarefaction wave given the pre-wave state.
    
    Inputs:
    - xi: Self-similarity variable (tensor)
    - rhoa: Pre-wave density (tensor)
    - pa: Pre-wave pressure (tensor)
    - vela: Pre-wave velocity (tensor)
    - gamma: Adiabatic index (tensor or scalar)
    - sign: +1 for right-going wave, -1 for left-going wave (tensor)
    
    Outputs:
    - rho: Density inside the rarefaction wave
    - p: Pressure inside the rarefaction wave
    - eps: Specific internal energy
    - h: Specific enthalpy
    - cs: Sound speed
    - vel: Flow velocity inside the rarefaction wave
    '''
    
    # Compute initial conditions
    epsa = get_eps(rhoa, pa, gamma)   
    ha = get_h(rhoa, pa, gamma)
    csa = get_csnd(rhoa, pa, gamma)
    wa = get_w(vela)

    # Compute transformation constants
    b = np.sqrt(gamma - 1)
    c = (b + csa) / (b - csa)
    d = -sign * b / 2.0
    k = (1.0 + xi) / (1.0 - xi)
    l = c * k**d
    v = ((1.0 - vela) / (1.0 + vela))**d

    # Define the function to solve for cs2 (sound speed inside rarefaction)
    def func(cs):
        return l * v * (1 + sign * cs)**d * (cs - b) + (1 - sign * cs)** d * (cs + b)

    def dfunc(cs):
        return (
            l * v * (1 + sign * cs)** d * (1 + sign * d * (cs - b) / (1 + sign * cs)) +
            (1 - sign * cs)** d * (1 - sign * d * (cs + b) / (1 - sign * cs))
        )

    # Solve for cs2 using Newton’s method
    cs2 = newton_solver(func, dfunc, csa, tol=5e-7)
    #cs2 = brent_solver(func, torch.zeros_like(csa), torch.ones_like(csa), max_iter=1000)
    # Compute velocity inside the rarefaction wave
    vel = (xi + sign * cs2) / (1.0 + sign * cs2 * xi)

    # Compute density inside the rarefaction wave
    rho = rhoa * ((cs2**2 * (gamma - 1 - csa**2)) / (csa**2 * (gamma - 1 - cs2**2)))** (1.0 / (gamma - 1))

    # Compute pressure inside the rarefaction wave
    p = cs2**2 * (gamma - 1) * rho / (gamma - 1 - cs2**2) / gamma
    eps = p / ((gamma - 1) * rho)
    
    # Compute enthalpy and sound speed
    h = get_h(rho, p, gamma)
    cs = get_csnd(rho, p, gamma)
    
    return rho, p, eps, h, cs, vel

def solve_riemann_problem(rho, temp, vel, eos, x, t, xc=0.0, dtype=torch.float32, verbose=False):
    """
    Vectorized relativistic Riemann solver.

    Inputs:
    - rho: Initial density (N,2)
    - temp: Initial temperature (N,2)
    - vel: Initial velocity (N,2)
    - eos: Equation of state (object with gamma_th)
    - x: Array of spatial positions
    - t: Time at which the solution is computed
    - xc: Initial position of the discontinuity

    Outputs:
    - out_data: Solution array (x, rho, press, eps, vel, csnd)
    """

    gamma = eos.gamma_th
    press, eps = eos.press_eps__temp_rho(temp, rho)

    h = get_h(rho, press, gamma)
    cs = get_csnd(rho, press, gamma)
    w = get_w(vel)

    # Find the range for the pressure root-find
    pmin = (press[:,0] + press[:,1]) * 0.5
    pmax = pmin.clone()

    # Vectorized root-finding initialization
    mask = get_dvel(pmin, rho[:,0], press[:,0], vel[:,0], rho[:,1], press[:,1], vel[:,1], gamma) * \
           get_dvel(pmax, rho[:,0], press[:,0], vel[:,0], rho[:,1], press[:,1], vel[:,1], gamma) > 0
    
    for i in range(10000):  # Prevent infinite loops
        pmin = torch.where(mask, torch.clamp(0.5 * pmin, min=0), pmin)
        pmax = torch.where(mask, 2*pmax, pmax)

        fmin = get_dvel(pmin, rho[:,0], press[:,0], vel[:,0], rho[:,1], press[:,1], vel[:,1], gamma, verbose=False)
        fmax = get_dvel(pmax, rho[:,0], press[:,0], vel[:,0], rho[:,1], press[:,1], vel[:,1], gamma, verbose=False)

        if verbose: print(f"Iteration {i}, pmax, pmin {pmax}, {pmin}, dvels: {fmin}, {fmax}")
        
        # Ensure no NaNs appear
        fmin = torch.nan_to_num(fmin, nan=0.0)
        fmax = torch.nan_to_num(fmax, nan=0.0)

        mask = (fmin * fmax) > 0  # True if root is still not bracketed
        
        if not torch.any(mask):  # Exit early if all cases are bracketed
            break
    if torch.any(mask):
        print("Warning: Some roots are not bracketed!")
        print("Some states where we failed to find a root:")
        print(f"{rho[mask,:]}, {press[mask,:]}, {vel[mask,:]}")
        
    # Solve for intermediate pressure (vectorized)
    press_c, convergence_mask = get_p(pmin, pmax, 1e-10, rho[:,0], press[:,0], vel[:,0], rho[:,1], press[:,1], vel[:,1], gamma, dtype)
    rho = rho[convergence_mask,:].clone()
    press = press[convergence_mask,:].clone()
    vel = vel[convergence_mask,:].clone()
    press_c = press_c[convergence_mask].clone()
    # Compute left and right post-wave states
    (rhols, epsls, hls, csls, vells, vshockl) = get_vel(press_c, rho[:,0], press[:,0], vel[:,0], -1, gamma)
    (rhors, epsrs, hrs, csrs, velrs, vshockr) = get_vel(press_c, rho[:,1], press[:,1], vel[:,1], +1, gamma)

    # Compute velocity of the contact discontinuity
    vels = (vells + velrs) / 2.0

    # Compute wave positions (vectorized)
    x1 = torch.where(press[:,0] > press_c, (vel[:,0] - cs[:,0]) / (1 - vel[:,0] * cs[:,0]) * t, xc + vshockl * t)
    x2 = torch.where(press[:,0] > press_c, (vels - csls) / (1 - vels * csls) * t, x1)
    x3 = xc + vels * t
    x4 = torch.where(press[:,1] > press_c, xc + (vels + csrs) / (1 + vels * csrs) * t, xc + vshockr * t)
    x5 = torch.where(press[:,1] > press_c, xc + (vel[:,1] + cs[:,1]) / (1 + vel[:,1] * cs[:,1]) * t, x4)

    # Compute xi = (x - xc) / t
    xi = (x - xc) / t

    # Allocate state variables as PyTorch tensors
    rho_out = torch.zeros_like(x, device=x.device)
    press_out = torch.zeros_like(x, device=x.device)
    eps_out = torch.zeros_like(x, device=x.device)
    csnd_out = torch.zeros_like(x, device=x.device)
    vel_out = torch.zeros_like(x, device=x.device)

    # Vectorized piecewise assignment using `torch.where`
    rho_out = torch.where(x <= x1  , rho[:, 0].squeeze(), rho_out)
    press_out = torch.where(x <= x1, press[:, 0].squeeze(), press_out)
    eps_out = torch.where(x <= x1  , eps[:, 0].squeeze(), eps_out)
    vel_out = torch.where(x <= x1  , vel[:, 0].squeeze(), vel_out)
    csnd_out = torch.where(x <= x1 , cs[:, 0].squeeze(), csnd_out)
    # Rarefaction wave (left)
    rho_raref, press_raref, eps_raref, _, csnd_raref, vel_raref = raref(xi, rho[:, 0], press[:, 0], vel[:, 0], gamma, +1)
    rho_out = torch.where((x > x1) & (x <= x2), rho_raref, rho_out)
    press_out = torch.where((x > x1) & (x <= x2), press_raref, press_out)
    eps_out = torch.where((x > x1) & (x <= x2), eps_raref, eps_out)
    vel_out = torch.where((x > x1) & (x <= x2), vel_raref, vel_out)
    csnd_out = torch.where((x > x1) & (x <= x2), csnd_raref, csnd_out)
    # Left of contact wave 
    rho_out = torch.where((x>x2)&(x<=x3), rhols, rho_out)
    press_out = torch.where((x>x2)&(x<=x3), press_c, press_out)
    eps_out = torch.where((x>x2)&(x<=x3), epsls, eps_out)
    vel_out = torch.where((x>x2)&(x<=x3), vels, vel_out)
    csnd_out = torch.where((x>x2)&(x<=x3), csls, csnd_out)
    # Right of contact wave 
    rho_out = torch.where((x>x3)&(x<=x4), rhors, rho_out)
    press_out = torch.where((x>x3)&(x<=x4), press_c, press_out)
    eps_out = torch.where((x>x3)&(x<=x4), epsrs, eps_out)
    vel_out = torch.where((x>x3)&(x<=x4), vels, vel_out)
    csnd_out = torch.where((x>x3)&(x<=x4), csrs, csnd_out)
    # Rarefaction wave (right)
    rho_raref, press_raref, eps_raref, _, csnd_raref, vel_raref = raref(xi, rho[:, 1], press[:, 1], vel[:, 1], gamma, -1)
    rho_out = torch.where((x>x4)&(x<=x5), rho_raref, rho_out)
    press_out = torch.where((x>x4)&(x<=x5), press_raref, press_out)
    eps_out = torch.where((x>x4)&(x<=x5), eps_raref, eps_out)
    vel_out = torch.where((x>x4)&(x<=x5), vel_raref, vel_out)
    csnd_out = torch.where((x>x4)&(x<=x5), csnd_raref, csnd_out)
    # Right unperturbed state
    rho_out = torch.where(x>=x5, rho[:, 1].squeeze(), rho_out)
    press_out = torch.where(x>=x5, press[:, 1].squeeze(), press_out)
    eps_out = torch.where(x>=x5, eps[:, 1].squeeze(), eps_out)
    vel_out = torch.where(x>=x5, vel[:, 1].squeeze(), vel_out)
    csnd_out = torch.where(x>=x5, cs[:, 1].squeeze(), csnd_out)
    # Convert back to NumPy for easy export
    out_data = torch.column_stack((
    x.squeeze(),
    rho_out.squeeze(),
    press_out.squeeze(),
    eps_out.squeeze(),
    vel_out.squeeze(),
    csnd_out.squeeze()
    ))


    return out_data

def print_label(label):
    if  label.ndim>1:
        raise ValueError("Can only print a single label")
    elif not ( torch.sum(label) == 1 ) :   
        raise ValueError("Invalid label provided, either empty or multiple predictions")
    if label[0]:
        print("Double shock (<->)")
    elif label[1]:
        print("Double rarefaction (<->)")
    elif label[2]:
        print("Rarefaction to the left (<-) shock to the right (->)")
    else:
        print("Shock to the left (<-) rarefaction to the right (->)")




def get_limiting_velocity_double_shock(rho1,p1,v1,rho2,p2,v2, gamma):
    h2 = get_h(rho2,p2,gamma)
    a = (1+(gamma-1)*(p2-p1)/(gamma*p1))
    b = -(gamma-1)*(p2-p1)/(gamma*p1)
    c = h2 * ( p2 - p1 ) / rho2 - h2**2
    
    h_hat = torch.where(
        p1>p2, (-b + torch.sqrt(b**2-4*a*c))/(2*a), (-b - torch.sqrt(b**2-4*a*c))/(2*a)
    )
    
    e_hat = h_hat * ( gamma * p1 ) / ( (gamma-1)*(h_hat-1) ) - p1 

    e2 = rho2 * ( get_eps(rho2, p2, gamma) + 1 )
    
    return torch.sqrt((p1-p2)*(e_hat-e2)/((e_hat+p2)*(e2+p1)))


def get_limiting_velocity_shock_raref(rho1,p1,v1,rho2,p2,v2, gamma):
    sqgl1 = np.sqrt(gamma-1) 
    K = p1/rho1**gamma 
    rho_raref = (p2/K)**(1/gamma)
    cs2   = get_csnd(rho_raref,p2,gamma)
    cs1   = get_csnd(rho1,p1,gamma)
    A_plus = ((sqgl1-cs2)/(sqgl1+cs2) * (sqgl1+cs1)/(sqgl1-cs1))**(2/sqgl1)
    
    return ((1-A_plus)/(1+A_plus))

def get_limiting_velocity_double_raref(rho1,p1,v1,rho2,p2,v2, gamma):
    sqgl1 = np.sqrt(gamma-1) 
    cs2   = get_csnd(rho2,p2,gamma)
    cs1   = get_csnd(rho1,p1,gamma)
    S1    = ((sqgl1+cs1)/(sqgl1-cs1))**(2/sqgl1)
    S2    = ((sqgl1+cs2)/(sqgl1-cs2))**(-2/sqgl1)
    return -(S1-S2)/(S1+S2)        
        
def classify_wave_pattern(prims,gamma):
    rhoL = torch.exp(prims[:,0,0])
    rhoR = torch.exp(prims[:,0,1])
    pL = torch.exp(prims[:,1,0])
    pR = torch.exp(prims[:,1,1])
    vL = (prims[:,2,0])
    vR = (prims[:,2,1])
    
    v_2S = get_limiting_velocity_double_shock(rhoL,pL,vL,rhoR,pR,vR,gamma)
    v_SR = get_limiting_velocity_shock_raref(rhoL,pL,vL,rhoR,pR,vR,gamma)
    v_2R = get_limiting_velocity_double_raref(rhoL,pL,vL,rhoR,pR,vR,gamma)
    
    v12 = (vL-vR)/(1-vL*vR)
    
    labels = torch.zeros((prims.size(0), 4), dtype=torch.bool, device=prims.device)
    
    labels[:,0] = v12 > v_2S                   # Double shock 
    labels[:,2] = (v12 <= v_2S) & (v12 > v_SR) # Shock Rarefaction 
    labels[:,1] = (v12 <= v_SR) & (v12 > v_2R) # Double rarefaction
    labels[:,3] = (v12 <= v_2R)                # 2 raref with vacuum
    
    return labels


def solve_raref(rhoa,pa,vela,sign,gamma):
    
    csa  = get_csnd(rhoa,pa,gamma)
    
    b = np.sqrt(gamma-1)
    c = (b+csa)/(b-csa)
    d = -sign * b/2.
    l = c
    
    v = ((1.-vela)/(1.+vela))**d
        
    func = lambda cs: l*v*(1+sign*cs)**d*(cs-b) + (1.-sign*cs)**d*(cs+b)
    dfunc = lambda cs: l*v*(1+sign*cs)**d*(1+sign*d*(cs-b)/(1+sign*cs))+(1-sign*cs)**d*(1.-sign*d*(cs+b)/(1.-sign*cs))

    return newton_solver(func, dfunc, csa, tol=1e-15)

def raref_rhs(cs, rhoa,pa,vela,sign,gamma):
    csa  = get_csnd(rhoa,pa,gamma)
    
    b = np.sqrt(gamma-1)
    c = (b+csa)/(b-csa)
    d = -sign * b/2.
    l = c
    
    v = ((1.-vela)/(1.+vela))**d
        
    return l*v*(1+sign*cs)**d*(cs-b) + (1.-sign*cs)**d*(cs+b)

def compute_riemann_invariant(rhoa, pa, vela, sign, gamma):
    csa = get_csnd(rhoa,pa,gamma)
    sqgl1 = np.sqrt(gamma-1)
    return (1+vela)/(1-vela) * ((sqgl1+csa)/(sqgl1-csa))**(sign*2/sqgl1)