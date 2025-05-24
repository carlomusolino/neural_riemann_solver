# PyTorch
import torch

from riemannML.solvers.solver import construct_riemann_solver
from riemannML.hydro.hlle_solver import HLLESolver
from riemannML.hydro.hllc_solver import HLLCSolver
from riemannML.hydro.exact_solver import ExactRiemannSolver

from riemannML.hydro.simulation_utils import run_test_battery, id


import os 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_sims(seed, 
             n_dsets,
             n_layers,
             n_neurons,
             do_first_order,
             do_second_order,
             resolutions, 
             solvers,
             basedir):
    
    bdir = os.path.join(basedir, str(n_layers)+"_layers_"+str(n_neurons)+"_neurons", str(n_dsets)+"_samples")
    
    path = os.path.join(bdir, str(seed))
    riemann_solver = construct_riemann_solver(path, device, 5/3)
    hlle_riemann_solver    = HLLESolver() 
    hllc_riemann_solver    = HLLCSolver() 
    exact_riemann_solver   = ExactRiemannSolver(5/3)

    names = list(dict.fromkeys(solvers))
    solver_map = {
        'ml': riemann_solver,
        'hlle': hlle_riemann_solver,
        'hllc': hllc_riemann_solver,
        'exact': exact_riemann_solver
    }
    riemann_solvers = [] 
    for name in names:
        riemann_solvers.append(solver_map[name])

    for prob in ['prob_1', 'prob_2', 'prob_3', 'prob_4']:
        if do_first_order:
            run_test_battery(id[prob], 
                            resolutions, riemann_solvers, names, 5/3, 
                            device=device, use_flux_limiter=False, 
                            log_timers=True, save_results_path=os.path.join(bdir, str(seed), prob+"_fo"))
        if do_second_order:
            run_test_battery(id[prob], 
                            resolutions, riemann_solvers, names, 5/3, 
                            device=device, use_flux_limiter=True, recon='weno', ngz=3, tstep='midpoint',
                            log_timers=False, save_results_path=os.path.join(bdir, str(seed), prob+"_so"))
