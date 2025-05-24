
from riemannML.solvers.solver import construct_riemann_solver_ensemble
from riemannML.hydro.hlle_solver import HLLESolver
from riemannML.hydro.hllc_solver import HLLCSolver


from riemannML.hydro.simulation_utils import run_test_battery, id


import torch
import os 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(n_dsets,n_layers,n_neurons, seed_combination, basedir):
    bdir = os.path.join(basedir, str(n_layers)+"_layers_"+str(n_neurons)+"_neurons", str(n_dsets)+"_samples")
    model_dirs = [] 
    for seed in seed_combination:
        model_dirs.append(os.path.join(bdir, str(seed)))
    riemann_solver = construct_riemann_solver_ensemble(model_dirs, device, 5/3)
    hlle_riemann_solver    = HLLESolver()
    hllc_riemann_solver    = HLLCSolver()
    names = ['ml', 'hlle', 'hllc']
    solvers = [riemann_solver, hlle_riemann_solver, hllc_riemann_solver]
    resolutions = [800]
    for prob in ['prob_1', 'prob_2', 'prob_3', 'prob_4']:
        outdir = os.path.join(bdir, str(seed_combination), prob+"_fo")
        os.makedirs(outdir, exist_ok=True)
        run_test_battery(id[prob], resolutions, solvers, names, 5/3, device=device, use_flux_limiter=False, log_timers=True, save_results_path=outdir)