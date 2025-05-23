# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from riemannML.solvers.solver import construct_riemann_solver
from riemannML.hydro.hlle_solver import HLLESolver 
from riemannML.hydro.hllc_solver import HLLCSolver 

from riemannML.hydro.simulation_utils import run_test_battery, id

import os 

from riemannML.training_utils.train_models import * 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(seed, n_dsets, batch, n_layers, n_neurons, learning_rate, basedir='../models'):
    bdir = os.path.join(basedir, str(n_layers)+"_layers_"+str(n_neurons)+"_neurons", str(n_dsets)+"_samples")
    train_models(
        seed=seed,n_neurons=n_neurons, n_layers=n_layers, gamma=5/3,N=n_dsets,batch_size=batch, lr=learning_rate, models_basepath=bdir,lpressmax=3.5, device=device
    )
    
    path = os.path.join(bdir, str(seed))
    riemann_solver = construct_riemann_solver(path, device, 5/3)
    
    names = ['ml']
    resolutions = [800]
    for prob in ['prob_1', 'prob_2', 'prob_3', 'prob_4']:
        run_test_battery(id[prob], resolutions, [riemann_solver], names, 5/3, device=device, use_flux_limiter=False, log_timers=True, save_results_path=os.path.join(bdir, str(seed), prob+"_fo"))
    