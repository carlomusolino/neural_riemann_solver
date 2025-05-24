import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import numpy as np

from .initial_data import oned_shocktube 

from riemannML.exact.riemann_numpy import solve_riemann_problem

from riemannML.utilities.error_measures import compute_L1_error, compute_L2_error

from .oned_srhd import SRHD1D

from riemannML.exact.hybrid_eos import hybrid_eos

import os 
import pickle

id = {
    "prob_1": {
        "rhol": 1,
        "rhor": 1,
        "pl"  : 1,
        "pr"  : 10,
        "vl"  : 0.9,
        "vr"  : 0.
    },
    
    "prob_2": {
        "rhol": 1,
        "rhor": 10,
        "pl"  : 10,
        "pr"  : 20,
        "vl"  : -0.6,
        "vr"  : 0.5
    },
    
    "prob_3": {
        "rhol": 10,
        "rhor": 1,
        "pl"  : 40/3,
        "pr"  : 1e-6,
        "vl"  : 0.,
        "vr"  : 0.
    },
    
    "prob_4": {
        'rhol': 1,
        'rhor': 1,
        'pl': 1e03,
        'pr': 1e-2,
        'vl': 0.0,
        'vr': 0.0
    }    
}

def run_shocktube(id, solver, gamma, t_end=0.4, CFL=0.8, tstep='euler', device=None):
    
    dtype = torch.float64
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Initial condition: Problem 1 (Mignone-Bodo)
    rhol, rhor = id['rhol'], id['rhor']
    pl, pr = id['pl'], id['pr']
    vl, vr = id['vl'], id['vr']
    U0, _ = oned_shocktube(rhol, rhor, pl, pr, vl, vr, solver.x, solver.nx, solver.ngz, gamma, device, dtype)
    
    dt = CFL * solver.h
    t = torch.linspace(0, t_end, int(t_end / dt) + 1, device=device, dtype=dtype)
    
    with torch.no_grad(): solution = odeint(solver, U0, t, method=tstep)
    
    solver._conservs_to_prims(solution[-1])
    
    x = np.linspace(solver.xmin, solver.xmax, solver.nx)
    exact_solution = solve_riemann_problem(rhol, pl, vl, rhor, pr, vr, gamma, x, t_end, 0., False)
    
    results = {
        'rho': solver.rho[solver.ngz:-solver.ngz].cpu().numpy(),
        'press': solver.press[solver.ngz:-solver.ngz].cpu().numpy(),
        'vel': solver.vel[solver.ngz:-solver.ngz].cpu().numpy()
    }
    
    return x, results, exact_solution, compute_L1_error(solver,exact_solution,solver.ngz,'rho'), compute_L2_error(solver,exact_solution,solver.ngz)

def run_test_battery(
    id, resolutions, 
    solvers, model_names,
    gamma, 
    domain=[-.5,.5], ngz=2, recon="godunov", limiter=None, bc_type='outgoing', use_flux_limiter=True, log_timers=False,
    t_end=0.4, CFL=0.8, tstep='euler', save_results_path=None, device=None, dtype = torch.float64):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eos = hybrid_eos(0,2,gamma)
    
    results = dict() 
    timers = dict()
    l1_errors = dict() 
    l2_errors = dict() 

    for model in model_names:
        l1_errors[model] = [] 
        l2_errors[model] = [] 
    
    for res in resolutions:
        results[res] = dict() 
        timers[res] = dict() 
        for solver, name in zip(solvers, model_names):
            print(f"Running resolution {res}, model {name}...")
            hydro_solver = SRHD1D(domain,res,ngz,eos,solver,recon,limiter,use_flux_limiter,bc_type,device,dtype)
            if log_timers: hydro_solver.timers_on() 
            else: hydro_solver.timers_off() # unnecessary but won't hurt..
            x, solution, exact_solution, l1, l2 = run_shocktube(id,hydro_solver,gamma,t_end,CFL,tstep,device)
            l1_errors[name].append(l1)
            l2_errors[name].append(l2)
            results[res][name] = solution 
            if log_timers: 
                timers[res][name] = {
                    'timers':  hydro_solver.get_timers(),
                    'counters': hydro_solver.get_counters() 
                }
                
        results[res]['x'] = x
    results['reference'] = exact_solution 
    results['x_reference'] = x 
    
    if save_results_path is not None:
        os.makedirs(save_results_path, exist_ok=True)
        with open(os.path.join(save_results_path,'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        errors = {'l1': l1_errors, 'l2': l2_errors, 'resolutions': resolutions }
        with open(os.path.join(save_results_path,'errors.pkl'), 'wb') as f:
            pickle.dump(errors, f)
        if log_timers: 
            with open(os.path.join(save_results_path,'timers.pkl'), 'wb') as f:
                pickle.dump(timers, f)
    