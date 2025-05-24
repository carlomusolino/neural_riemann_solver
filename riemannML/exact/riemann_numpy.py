import numpy as np
from scipy.optimize import brentq
from scipy.optimize import newton


def get_eps(rho,p,gamma):
    return p/(gamma-1.)/rho 
def get_h(rho,p,gamma):
    eps = get_eps(rho,p,gamma)
    return 1. + eps + p/rho
def get_csnd(rho,p,gamma):
    h = get_h(rho,p,gamma)
    return np.sqrt(gamma*p/rho/h)
def get_w(vel):
    return 1./np.sqrt(1-vel**2)


def get_vel(p,rhoa,pa,vela,sign,gamma):
    '''
    Compute the flow velocity behind a rarefaction or shock in terms of 
    post-wave pressure for a given state ahead of the wave.
    p -> post-wave pressure
    rhoa -> density ahead of the wave 
    pa -> pressure ahead of the wave 
    vela -> flow vel ahead of wave
    sign -> +1 for right-going wave, -1 for left-going
    Output: (rho,eps,h,csnd,vel,vshock)
    '''
    epsa = get_eps(rhoa,pa,gamma)
    ha = 1. + epsa + pa/rhoa 
    csa = np.sqrt(gamma*pa/rhoa/ha)
    wa = get_w(vela)
    if p>pa:
        # shock 
        a = 1+(gamma-1.)*(pa-p)/gamma/p
        b = 1-a 
        c = ha * (pa-p)/rhoa - ha**2
        
        h = (-b+np.sqrt(b**2-4*a*c))/2./a 
        rho = gamma*p/(gamma-1)/(h-1)
        eps = p/(gamma-1.)/rho
        j = sign * np.sqrt((p-pa)/(ha/rhoa-h/rho))
        
        a = j**2+(rhoa*wa)**2
        b = -vela*rhoa**2*wa**2
        
        vshock = (-b+sign*j**2*np.sqrt(1+rhoa**2/j**2))/a
        
        wshock = get_w(vshock)
        
        a =  wshock*(p-pa)/j+ha*wa*vela
        b = ha*wa+(p-pa)*(wshock*vela/j+1/rhoa/wa)
        
        vel = a/b 
        
        cs = get_csnd(rho,p,gamma)
        
        return (rho,eps,h,cs,vel,vshock)
    else:
        # Rarefaction
        k = pa/rhoa**gamma 
        
        rho = (p/k)**(1./gamma)
        
        eps = p/(gamma-1)/rho 
        cs = np.sqrt(gamma*p/(rho+gamma*p/(gamma-1)))
        
        sqgl1 = np.sqrt(gamma-1)
        a = (1+vela)/(1-vela)*((sqgl1+csa)/(sqgl1-csa)*(sqgl1-cs)/(sqgl1+cs))**(-sign*2/sqgl1)
        
        vel = (a-1.)/(a+1)
        
        h = get_h(rho,p,gamma)
        return (rho,eps,h,cs,vel,0.)


def get_dvel(p,rhol,pl,vell,rhor,pr,velr,gamma):
    '''
    Get difference in flow speed between left and right intermediate states
    given left and right states and pressure in the intermediate state 
    '''
    
    (rhols,epsls,hls,csls,vells,vshockl) = get_vel(p,rhol,pl,vell,-1,gamma)
    
    (rhors,epsrs,hrs,csrs,velrs,vshockr) = get_vel(p,rhor,pr,velr,+1,gamma)
    
    return (vells-velrs)

def get_p(pmin,pmax,tol,rhol,pl,vell,rhor,pr,velr,gamma):
    '''
    Find the pressure in intermediate state of a Riemann problem of SRHYDRO.
    '''
    func = lambda x: get_dvel(x,rhol,pl,vell,rhor,pr,velr,gamma)
    
    return brentq(func,pmin,pmax,xtol=tol)


def raref_new(xi,rhoa,pa,vela,gamma,sign):
    
    # sign = +1 --> left 
    # sign = -1 --> right 
    
    csa  = get_csnd(rhoa,pa,gamma)
    B = np.sqrt(gamma-1)
    D = sign * 2/B 
    A = ((1+vela)/(1-vela)) 
    C = ((B+csa)/(B-csa))**D
    X = ((1-xi)/(1+xi))
    K = C * A * X 
    # this is just (1-v)/(1+v)
    
    func = lambda cs: K * (1-sign*cs) * (B-cs)**D - (1+sign*cs) * (cs+B)**D
    
    def dfunc(cs):
        return - sign * K * (B-cs)**D - sign * (cs+B)**D - D * K * (B-cs)**(D-1) * ( 1- sign*cs) - D * ( B + cs)**(D-1)*(1+sign*cs)

    
    cs = newton(func, csa, fprime=dfunc, tol = 1e-15)
    
    vel = ( xi + sign*cs ) / (1. + sign*cs*xi) 
    
    rho = rhoa*((cs**2*(gamma-1-csa**2))/(csa**2*(gamma-1-cs**2)))**(1./(gamma-1))
    
    p = cs**2*(gamma-1)*rho/(gamma-1-cs**2)/gamma
    eps = p/(gamma-1)/rho
    h = get_h(rho,p,gamma)
    cs = get_csnd(rho,p,gamma)
    
    return (rho,p,eps,h,cs,vel)
    
def raref(xi,rhoa,pa,vela,gamma,sign):
    '''
    Compute the flow state in a rarefaction wave given pre-wave state.
    '''
    epsa = get_eps(rhoa,pa,gamma)   
    ha   = get_h(rhoa,pa,gamma)
    csa  = get_csnd(rhoa,pa,gamma)
    wa   = get_w(vela)
    
    b = np.sqrt(gamma-1)
    c = (b+csa)/(b-csa)
    d = -sign * b/2.
    k = (1.+xi)/(1.-xi)
    l = c*k**d
    
    
    v = ((1.-vela)/(1.+vela))**d
    #v = ((1.+vela)/(1.-vela))**d
        
    func = lambda cs: l*v*(1+sign*cs)**d*(cs-b) + (1.-sign*cs)**d*(cs+b)
    dfunc = lambda cs: l*v*(1+sign*cs)**d*(1+sign*d*(cs-b)/(1+sign*cs))+(1-sign*cs)**d*(1.-sign*d*(cs+b)/(1.-sign*cs))
    
    
    cs2 = newton(func, csa, fprime=dfunc, tol = 5e-07)
    
    vel = ( xi + sign*cs2 ) / (1. + sign*cs2*xi) 
    
    rho = rhoa*((cs2**2*(gamma-1-csa**2))/(csa**2*(gamma-1-cs2**2)))**(1./(gamma-1))
    
    p = cs2**2*(gamma-1)*rho/(gamma-1-cs2**2)/gamma
    eps = p/(gamma-1)/rho
    h = get_h(rho,p,gamma)
    cs = get_csnd(rho,p,gamma)
    
    return (rho,p,eps,h,cs,vel)

def get_wave_structure(rhol,pl,vell,rhor,pr,velr,gamma):
    epsl = get_eps(rhol,pl,gamma)   
    hl   = get_h(rhol,pl,gamma)
    csl  = get_csnd(rhol,pl,gamma)
    wl   = get_w(vell)
    
    epsr = get_eps(rhor,pr,gamma)   
    hr   = get_h(rhor,pr,gamma)
    csr  = get_csnd(rhor,pr,gamma)
    wr   = get_w(velr)
    
    dvel1 = -1
    dvel2 = -1 
    pmin = (pl+pr)/2.
    pmax = pmin
    while(get_dvel(pmin,rhol,pl,vell,rhor,pr,velr,gamma)*get_dvel(pmax,rhol,pl,vell,rhor,pr,velr,gamma) > 0):
        pmin = 0.5*max(pmin,0.)
        pmax = 2.*pmax 
    print(pmin,pmax)
    p, r = get_p(pmin,pmax,1e-10,rhol,pl,vell,rhor,pr,velr,gamma)
    print(f"Found pressure {p}, iterations {r.iterations}")
    (rhols,epsls,hls,csls,vells,vshockl) = get_vel(p,rhol,pl,vell,-1,gamma)
    (rhors,epsrs,hrs,csrs,velrs,vshockr) = get_vel(p,rhor,pr,velr,+1,gamma)   
    
    vels = (vells+velrs) / 2.
    
    if ( (p > pr) and (p<=pl)):
        print("Shock rarefaction")
    if ( (p>pr) and (p>pl) ): 
        print("double shock")
    if ( (p<=pr) and (p<=pl)):
        print("double raref")

def solve_riemann_problem(rhol,pl,vell,rhor,pr,velr,gamma,x,t,xc=0.,save_to_file=True,outfname="riemann.dat"):
    
    epsl = get_eps(rhol,pl,gamma)   
    hl   = get_h(rhol,pl,gamma)
    csl  = get_csnd(rhol,pl,gamma)
    wl   = get_w(vell)
    
    epsr = get_eps(rhor,pr,gamma)   
    hr   = get_h(rhor,pr,gamma)
    csr  = get_csnd(rhor,pr,gamma)
    wr   = get_w(velr)
    
    dvel1 = -1
    dvel2 = -1 
    pmin = (pl+pr)/2.
    pmax = pmin
    while(get_dvel(pmin,rhol,pl,vell,rhor,pr,velr,gamma)*get_dvel(pmax,rhol,pl,vell,rhor,pr,velr,gamma) > 0):
        pmin = 0.5*max(pmin,0.)
        pmax = 2.*pmax 
    p = get_p(pmin,pmax,1e-10,rhol,pl,vell,rhor,pr,velr,gamma)
    print(f"Found pressure {p}")
    if ( (p > pr) and (p<=pl)):
        print("Shock rarefaction")
    if ( (p>pr) and (p>pl) ): 
        print("double shock")
    if ( (p<=pr) and (p<=pl)):
        print("double raref")
    (rhols,epsls,hls,csls,vells,vshockl) = get_vel(p,rhol,pl,vell,-1,gamma)
    (rhors,epsrs,hrs,csrs,velrs,vshockr) = get_vel(p,rhor,pr,velr,+1,gamma)   
    
    vels = (vells+velrs) / 2.

    if( pl > p):
        x1 = xc + (vell-csl)/(1.-vell*csl)*t 
        x2 = xc + (vels-csls)/(1.-vels*csls)*t
    else:
        x1 = xc + vshockl*t 
        x2 = x1 
    
    x3 = xc + vels*t 
    
    if ( pr > p ):
        x4 = xc + (vels+csrs)/(1.+vels*csrs)*t 
        x5 = xc + (velr+csr) /(1+velr*csr)*t 
    else :
        x4 = xc + vshockr*t
        x5 = x4 
    
    rho = np.zeros(x.shape)
    press = np.zeros(x.shape)
    eps = np.zeros(x.shape)
    csnd = np.zeros(x.shape)
    vel = np.zeros(x.shape)
    
    for i in range(len(x)):
        xi = (x[i]-xc)/t 
        xx = x[i]
        if  xx <= x1 :
            press[i] = pl 
            rho[i] = rhol 
            vel[i] = vell 
            eps[i] = epsl 
            csnd[i] = csl 
        elif xx <= x2 :
            (rho[i],press[i],eps[i],d1,csnd[i],vel[i]) = raref(xi,rhol,pl,vell,gamma,+1)
        elif xx <= x3 :
            press[i] = p 
            rho[i]   = rhols
            eps[i]   = epsls
            vel[i]   = vels
            csnd[i]  = csls
        elif xx <= x4 :
            press[i] = p 
            rho[i]   = rhors
            eps[i]   = epsrs
            vel[i]   = vels
            csnd[i]  = csrs
        elif xx<=x5 :
            (rho[i],press[i],eps[i],d1,csnd[i],vel[i]) = raref(xi,rhor,pr,velr,gamma,-1)
        else:
            press[i] = pr
            rho[i] = rhor
            vel[i] = velr 
            eps[i] = epsr 
            csnd[i] = csr
    out_data = np.column_stack((x,rho,press,eps,vel,csnd))
    if save_to_file:
        np.savetxt(outfname,out_data)
    return out_data