import torch 

def minmod(a, b):
    """
    Computes the minmod function, which is a slope limiter used in numerical methods
    for solving hyperbolic partial differential equations. The minmod function selects
    the smallest magnitude value with the same sign if both inputs have the same sign,
    otherwise it returns zero.

    Args:
        a (torch.Tensor): The first input tensor.
        b (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: A tensor containing the result of the minmod function applied
        element-wise to the inputs.
    """
    cond = (a*b > 0)
    return torch.where(cond, torch.sign(a)*torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))

def mc2(a, b):
    """
    Computes the MC2 limiter function, commonly used in numerical methods for solving 
    hyperbolic partial differential equations to limit oscillations near discontinuities.

    The function takes two input tensors `a` and `b`, and applies a combination of 
    minmod and other limiting techniques to produce a result that ensures stability 
    and avoids spurious oscillations.

    Args:
        a (torch.Tensor): A tensor representing the first input values.
        b (torch.Tensor): A tensor representing the second input values.

    Returns:
        torch.Tensor: A tensor containing the limited values, computed based on the 
        MC2 limiter formula.

    Notes:
        - The function assumes that `a` and `b` are tensors of the same shape and 
          reside on the same device.
        - The limiter ensures that the output is non-negative and bounded by the 
          inputs in a way that preserves monotonicity.
    """
    c = 0.5 * ( a + b )
    signb = torch.sign(b)
    return signb * torch.max(
        torch.zeros_like(a, device=a.device, dtype=a.dtype),
        torch.min(
            2 * torch.min(torch.abs(b), signb * a),
            signb * c
        )
    )
