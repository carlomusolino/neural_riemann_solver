import torch 

def newton_solver(func, dfunc, x0, tol=5e-7, max_iter=20):
    '''
    Newton's method solver for root-finding in PyTorch (batch-compatible).
    
    Inputs:
    - func: Function whose root we seek
    - dfunc: Derivative of func
    - x0: Initial guess (tensor)
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Outputs:
    - Root estimate (tensor)
    '''
    x = x0.clone()
    
    for _ in range(max_iter):
        fx = func(x)
        dfx = dfunc(x)

        # Ensure no division by zero
        dfx = torch.where(torch.abs(dfx) > 1e-12, dfx, torch.ones_like(dfx) * 1e-12)

        x_new = x - fx / dfx
        
        # Check convergence
        if torch.max(torch.abs(x_new - x)) < tol:
            break

        x = x_new

    return x

def bisection_solver(func, pmin, pmax, tol=1e-7):
    orig_dtype = pmin.dtype 
    
    # Inside the solver we always want doubles
    pmin = pmin.to(torch.float64)
    pmax = pmax.to(torch.float64)
    
    # Compute function values at bracketing points
    fmin = func(pmin)
    fmax = func(pmax)

    # Mask for valid root-bracketed cases
    mask = (fmin * fmax < 0)
    invalid_mask = (fmin * fmax) > 0 
    nanmask = torch.isnan(fmin) | torch.isnan(fmax)
    
    root_found_mask_L = fmin == 0 
    root_found_mask_R = fmax == 0 

    if torch.all(invalid_mask):
        raise ValueError("No valid roots: Function values must bracket the root.")
    if torch.any(invalid_mask):
        print(f"WARNING: In bisection {torch.sum(~mask)}/{pmin.size(0)} roots are not bracketed")
    if nanmask.any():
        print(f"WARNING: {torch.isnan(fmin).sum() + torch.isnan(fmax).sum()} nan entries")
        print(f"NaN index {nanmask.nonzero()}")
        print(f"NaN pmin/pmax {pmin[nanmask]}/{pmax[nanmask]}")
        print(f"NaN pmin/pmax {fmin[nanmask]}/{fmax[nanmask]}")
    
    # Create filtered tensors for valid cases
    p0, p1 = pmin.clone(), pmax.clone()
    f0, f1 = fmin.clone(), fmax.clone()
    # Compute function values
    f0 = func(p0)
    f1 = func(p1)
    
    macheps = torch.finfo(p0.dtype).eps
    
    while(True):
        t = 2 * macheps * torch.abs(p1) + tol
        # Bisection step as fallback
        p_new = (p0 + p1) / 2  
        
        # Compute function value at new point
        f_new = func(p_new)
        
        # Update brackets
        update_mask = (f0 * f_new < 0)
        p1 = torch.where(update_mask, p_new, p1)
        f1 = torch.where(update_mask, f_new, f1)
        p0 = torch.where(update_mask, p0, p_new)
        f0 = torch.where(update_mask, f0, f_new)

        # Check convergence ( excluding non-bracketed roots )
        convergence_mask = ((p1-p0).abs()<t) | (torch.abs(f_new) == 0) | invalid_mask | nanmask
        if torch.all(convergence_mask):
            break
        
    # Create full-sized output tensor, filling invalid cases with NaN
    p_root = torch.full_like(pmin, torch.nan)
    if root_found_mask_L.any(): p_root[root_found_mask_L] = pmin[root_found_mask_L]
    if root_found_mask_R.any(): p_root[root_found_mask_R] = pmax[root_found_mask_R]
    p_root[mask] = p1[mask]

    full_convergence_mask = torch.zeros_like(pmin, dtype=torch.bool)
    if root_found_mask_L.any(): full_convergence_mask[root_found_mask_L] = True 
    if root_found_mask_R.any(): full_convergence_mask[root_found_mask_R] = True 
    full_convergence_mask[mask] = convergence_mask[mask]
    # We cast back to original type here
    return p_root.to(orig_dtype), full_convergence_mask
        

def brent_solver(func, pmin, pmax, tol=1e-7, max_iter=50):
    """
    Implements Brent's root-finding method in PyTorch (batch-compatible).
    This version filters out invalid cases where the root is not bracketed.

    Inputs:
    - func: Function for which we want to find the root
    - pmin, pmax: Initial bracketing values (tensors)
    - tol: Convergence tolerance
    - max_iter: Maximum number of iterations

    Output:
    - p_root: Estimated root tensor (with NaN for invalid cases)
    - convergence_mask: Boolean tensor indicating which roots successfully converged
    """
    
    # Compute function values at bracketing points
    fmin = func(pmin)
    fmax = func(pmax)

    # Mask for valid root-bracketed cases
    mask = (fmin * fmax < 0)

    if not torch.any(mask):
        raise ValueError("No valid roots: Function values must bracket the root.")

    # Create filtered tensors for valid cases
    p0, p1 = pmin.clone(), pmax.clone()
    f0, f1 = fmin.clone(), fmax.clone()

    macheps = torch.finfo(torch.float64).eps

    for _ in range(max_iter):
        t = 2 * macheps * torch.abs(p1) + tol

        # Compute function values
        f0 = func(p0)
        f1 = func(p1)

        # Swap if necessary to ensure |f1| < |f0|
        swap_mask = torch.abs(f1) < torch.abs(f0)
        p0, p1 = torch.where(swap_mask, p1, p0), torch.where(swap_mask, p0, p1)
        f0, f1 = torch.where(swap_mask, f1, f0), torch.where(swap_mask, f0, f1)

        # Bisection step as fallback
        p_new = (p0 + p1) / 2  

        # Secant step (faster)
        secant_step = p1 - f1 * (p1 - p0) / (f1 - f0)

        # Use secant if valid, otherwise use bisection
        p_new = torch.where(torch.abs(f1 - f0) > 1e-12, secant_step, p_new)

        # Compute function value at new point
        f_new = func(p_new)

        # Update brackets
        update_mask = (f_new * f1 < 0)
        p0, f0 = torch.where(update_mask, p1, p0), torch.where(update_mask, f1, f0)
        p1, f1 = p_new, f_new

        # Check convergence
        convergence_mask = (torch.abs(p_new - p0) < t) | (torch.abs(f_new) == 0) | (~mask)
        if torch.all(convergence_mask):
            break

    # Create full-sized output tensor, filling invalid cases with NaN
    p_root = torch.full_like(pmin, torch.nan)
    p_root[mask] = p1[mask]

    full_convergence_mask = torch.zeros_like(pmin, dtype=torch.bool)
    full_convergence_mask[mask] = convergence_mask[mask]

    return p_root, full_convergence_mask  # Root estimates and convergence info
