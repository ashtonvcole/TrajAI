# Preprocess data
# Iterate to read in files
# Consider trajectory of each runner
# (Only use segments with all)
# Compute v, a
# Write states to npy files, organize appropriately
# State should be in normalized local coordinate system

import numpy as np

def get_states(trajs: list) -> np.ndarray:
    # Debating if list or ndarray
    # Define 3d state array (time, runner, state var)
    pass

def get_edges(trajs: list) -> np.ndarray:
    # Debating if list or ndarray
    # Define 3d edge array (time, edge, edge var)
    # Every runner connected to every, I assume
    pass

def get_Dxs(xs: np.ndarray, dt=1) -> np.ndarray:
    """Compute derivative along a fixed-interval trajectory.
    
    Second-order-accurate finite differences are used to compute the first derivative. Central differences are used in the middle of the sequence. One-sided stencils are used for the first and last values.

    Input:
        xs (np.ndarray): A list of scalar function values of length at least 3.

    Output:
        Dxs (np.ndarray): A list of derivatives of the same length, corresponding to the respective values.
    """
    if len(xs) < 3:
        raise ValueError("Input array must have a length of at least 3 for these stencils.")
    Dxs = np.zeros(xs.shape)
    Dxs[0] = (-3/2 * xs[0] + 2 * xs[1] - 1/2 * xs[2]) / dt
    Dxs[1:-1] = (xs[2:] - xs[0:-2]) / (2 * dt)
    Dxs[-1] = (1/2 * xs[-3] - 2 * xs[-2] + 3/2 * xs[-1]) / dt
    return Dxs

def get_D2xs(xs: np.ndarray, dt=1) -> np.ndarray:
    """Compute second derivative along a fixed-interval trajectory.
    
    Second-order-accurate finite differences are used to compute the second derivative. Central differences are used in the middle of the sequence. One-sided stencils are used for the beginning and end values.

    Input:
        xs (np.ndarray): A list of scalar function values of length at least 4.

    Output:
        D2xs (np.ndarray): A list of derivatives of the same length, corresponding to the respective values.
    """
    if len(xs) < 4:
        raise ValueError("Input array must have a length of at least 3 for these stencils.")
    D2xs = np.zeros(xs.shape)
    D2xs[0] = (2 * xs[0] - 5 * xs[1] + 4 * xs[2] - xs[3]) / dt ** 2
    D2xs[1:-1] = (xs[2:] - 2 * xs[1:-1] +  xs[0:-2]) / dt ** 2
    D2xs[-1] = (-xs[-4] + 4 * xs[-3] - 5 * xs[-2] + 2 * xs[-1]) / dt ** 2
    return D2xs