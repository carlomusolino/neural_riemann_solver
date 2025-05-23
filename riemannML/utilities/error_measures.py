import torch

def compute_L1_error(solver, exact, ngz, var):
    device = solver.device 
    dtype  = solver.dtype 
    
    
    exact = torch.from_numpy(exact).to(device)
    if var == 'rho':
        l1_err = (exact[:,1] - solver.rho[ngz:-ngz]).abs().mean() 
    elif var == 'press':
        l1_err = (exact[:,2] - solver.press[ngz:-ngz]).abs().mean() 
    else:
        l1_err = (exact[:,4] - solver.vel[ngz:-ngz]).abs().mean() 
    
    return l1_err.item() 

def compute_L2_error(solver, exact, ngz):
    device = solver.device 
    exact = torch.from_numpy(exact).to(device)

    rho_err = ((exact[:,1] - solver.rho[ngz:-ngz])**2).mean()
    press_err = ((exact[:,2] - solver.press[ngz:-ngz])**2).mean()
    vel_err = ((exact[:,4] - solver.vel[ngz:-ngz])**2).mean()
    
    l2_err = ((rho_err + press_err + vel_err)/3).sqrt()
    return l2_err.item()