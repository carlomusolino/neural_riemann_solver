# PyTorch
import torch

from riemannML.solvers.solver import construct_riemann_solver
from riemannML.hydro.hlle_solver import HLLESolver
from riemannML.hydro.hllc_solver import HLLCSolver
from riemannML.hydro.exact_solver import ExactRiemannSolver

from riemannML.hydro.simulation_utils import run_test_battery, id


import os 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(seed, n_dsets, n_layers, n_neurons, basedir):
    
    bdir = os.path.join(basedir, str(n_layers)+"_layers_"+str(n_neurons)+"_neurons", str(n_dsets)+"_samples")
    
    path = os.path.join(bdir, str(seed))
    riemann_solver = construct_riemann_solver(path, device, 5/3)
    hlle_riemann_solver    = HLLESolver() 
    hllc_riemann_solver    = HLLCSolver() 
    exact_riemann_solver   = ExactRiemannSolver(5/3)
    
    # Run test batteries 
    solvers = [hlle_riemann_solver, hllc_riemann_solver, riemann_solver, exact_riemann_solver]
    names = ['hlle','hllc','ml','exact']
    names_noexact = ['hlle','hllc','ml']
    solvers_noexact = [hlle_riemann_solver, hllc_riemann_solver, riemann_solver]
    resolutions = [50,100,200,400,800,1600,3200]
    for prob in ['prob_1', 'prob_2', 'prob_3', 'prob_4']:
        run_test_battery(id[prob], 
                         resolutions, solvers, names, 5/3, 
                         device=device, use_flux_limiter=False, 
                         log_timers=True, save_results_path=os.path.join(bdir, str(seed), prob+"_fo"))
        run_test_battery(id[prob], 
                         resolutions, solvers_noexact, names_noexact, 5/3, 
                         device=device, use_flux_limiter=True, recon='weno', ngz=3, tstep='midpoint',
                         log_timers=False, save_results_path=os.path.join(bdir, str(seed), prob+"_so"))