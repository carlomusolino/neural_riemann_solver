import torch 
from riemannML.exact.riemann_solver import get_h 

def setup_initial_state_shocktube(rhol,rhor,pl,pr,vell,velr,x,ncells,nghost,gamma,device,dtype):
    U0 = torch.zeros(ncells+2*nghost, 3, device=device, dtype=dtype)
    P0 = torch.zeros_like(U0, device=device, dtype=dtype)
    
    P0[:,0] = torch.where( x <= 0., rhol * torch.ones_like(x, device=device), rhor * torch.ones_like(x, device=device)) 
    P0[:,2] = torch.where( x <= 0., vell * torch.ones_like(x, device=device), velr * torch.ones_like(x, device=device))
    P0[:,1] = torch.where( x <= 0., pl * torch.ones_like(x, device=device),   pr * torch.ones_like(x, device=device))

    W = 1./torch.sqrt(1-P0[:,2]**2)
    U0[:,0] = W * P0[:,0]
    h = get_h(P0[:,0], P0[:,1], gamma)
    U0[:,1] = W**2*P0[:,0] * h - P0[:,1] - U0[:,0]
    U0[:,2] = W**2*P0[:,0] * h * P0[:,2]
    return U0, P0