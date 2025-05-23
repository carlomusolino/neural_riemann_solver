import torch
import torch.nn.functional as F
from .limiters import minmod 

def slope(aj,ajp1,ajm1):
    daj  = 0.5 * (ajp1-aj) + 0.5 * ( aj - ajm1 )
    s    = daj.sign()
    cond = (ajp1-aj) * (aj-ajm1) > 0 
    return torch.where(
        cond,
        torch.min(
            torch.min(daj.abs(), 2*(aj-ajm1).abs()),
            2*(ajp1-aj).abs()
        ) * s,
        torch.zeros_like(aj,device=aj.device,dtype=aj.dtype)
        )

def interp_to_faces(P, nx, ngz):
    device = P.device 
    dtype  = P.dtype     
    _, nv  = P.shape
    
    
    aj   = P[ngz-1:nx+ngz, :]
    ajp1 = P[ngz:nx+ngz+1, :]
    ajp2 = P[ngz+1:nx+ngz+2]
    ajm1 = P[ngz-2:nx+ngz-1, :]
    ajm2 = P[ngz-3:nx+ngz-1, :]
    

    daj = slope(aj, ajp1, ajm1)
    dajp1 = slope(ajp1, ajp2, aj)
    
    return 0.5 * (ajp1+aj) + 1./6. * (daj+dajp1)

def shock_indicator( press, vel, nx, ngz, eps, omega1, omega2):
    
    device = press.device 
    dtype  = press.dtype
    
    pjp1 = press[ngz:nx+ngz+2]
    pjm1 = press[ngz-2:nx+ngz]
    pjp2 = press[ngz+1:nx+ngz+3]
    pjm2 = press[ngz-3:nx+ngz-1]
    
    sj = (pjp1-pjm1).sign().to(torch.int64)
    
    vjp1 = vel[ngz:nx+ngz+2]
    vjm1 = vel[ngz-2:nx+ngz]
    
    shock_cond = ((pjp1-pjm1).abs()/torch.min(pjp1,pjm1) > eps) & ( vjm1 > vjp1 )
    
    ftilde = torch.min(
        torch.ones_like(pjp1, device=device, dtype=dtype),
        shock_cond.to(dtype) * torch.max(
            ((pjp1-pjm1)/(pjp2-pjm2+1e-45) - omega1) * omega2,
            torch.zeros_like(pjp1, device=device, dtype=dtype)
        )
    )
    # We pad ftilde in both directions, total size is ncells + 3 
    ftilde = F.pad(ftilde,(1,1), mode='constant', value=0.0)
    
    # Ignore the ghostzones, just pretend that we have no shocks there (should be fine)
    idxpsj = torch.arange(1, ftilde.size(0)-1, device=device) + sj
    
    # f has size ncells + 2 
    f = torch.max(ftilde[1:-1], ftilde[idxpsj]).clamp(0,1)
    return f[:-1]

def shock_flattening(uL, uR, a, press, vel, nx, ngz, eps, omega1, omega2):
    
    device = uL.device 
    dtype  = uL.dtype
    
    pjp1 = press[ngz:nx+ngz+2]
    pjm1 = press[ngz-2:nx+ngz]
    pjp2 = press[ngz+1:nx+ngz+3]
    pjm2 = press[ngz-3:nx+ngz-1]
    
    sj = (pjp1-pjm1).sign().to(torch.int64)
    
    vjp1 = vel[ngz:nx+ngz+2]
    vjm1 = vel[ngz-2:nx+ngz]
    
    shock_cond = ((pjp1-pjm1).abs()/torch.min(pjp1,pjm1) > eps) & ( vjm1 > vjp1 )
       
    ftilde = torch.min(
        torch.ones_like(pjp1, device=device, dtype=dtype),
        shock_cond.to(dtype) * torch.max(
            ((pjp1-pjm1)/(pjp2-pjm2+1e-45) - omega1) * omega2,
            torch.zeros_like(pjp1, device=device, dtype=dtype)
        )
    )
    # We pad ftilde in both directions, total size is ncells + 3 
    ftilde = F.pad(ftilde,(1,1), mode='constant', value=0.0)
    
    # Ignore the ghostzones, just pretend that we have no shocks there (should be fine)
    idxpsj = torch.arange(1, ftilde.size(0)-1, device=device) + sj
    
    # f has size ncells + 2 
    f = torch.max(ftilde[1:-1], ftilde[idxpsj]).clamp(0,1)
    

    uL = f[:-1,None] * a[ngz-1 : nx+ngz, :]   + (1 - f[:-1,None]) * uL.clone()
    uR = f[1:,None]  * a[ngz   : nx+ngz+1, :] + (1 - f[1:,None])  * uR.clone()
    
    return uL, uR 

def monotonize(aL,aR,a, nx, ngz):
    
    device = a.device 
    dtype  = a.dtype
    
    aj = a[ngz-1:nx+ngz]
    
    mask1 = (aR - aj) * (aj - aL) <= 0 
    mask2 = (aR-aL)*(aj - (aL+aR)/2) > (aR-aL)**2/6
    mask3 = (aL-aR)*(aj-(aL+aR)/2) > (aR-aL)**2/6 
    
    aL = torch.where(mask1, aj, aL)
    aR = torch.where(mask1, aj, aR)
    
    aL = torch.where(mask2, 3*aj - 2*aR, aL)
    aR = torch.wnere(mask3, 3*aj - 2*aL, aR)
    
    return aL, aR 
    
    
    