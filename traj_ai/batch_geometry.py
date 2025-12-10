import torch
import torch.nn as nn



def rotate_2D(vec: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate a batch of 2D vectors by an angle. This is equivalent to rotating the frame by -theta.

    Arguments:
        vec (torch.Tensor): A list of reference vectors, of dimension (n, 2).
        theta (torch.Tensor): A list of angles, of dimension n or (n, 1).

    Returns:
        torch.Tensor: A list of rotated vectors.
    """
    if theta.ndim == 1:
        theta = theta.unsqueeze(-1)
    elif theta.ndim == 2:
        pass
    else:
        raise ValueError('Argument theta should be a 1 or 2 dimensional tensor.')
    # Build rotation matrices
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.stack([
        torch.stack([c.squeeze(-1), -s.squeeze(-1)], dim=-1),
        torch.stack([s.squeeze(-1), c.squeeze(-1)], dim=-1) 
    ], dim=1).transpose(1, 2)
    return torch.bmm(vec.unsqueeze(1), R).squeeze(1)



def get_angle_2D(ref: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Get the angle of a vector relative to a reference vector.

    The formula is atan2(ref x vec . e3, ref . vec)

    Arguments:
        ref (torch.Tensor): A list of reference vectors, of dimension (n, 2).
        vec (torch.Tensor): A list of comparison vectors, of dimension (n, 2).

    Returns:
        torch.Tensor: The signed angle relative to the reference vector, of dimension (n, 1).
    """
    cross = ref[:, 0] * vec[:, 1] - ref[:, 1] * vec[:, 0]
    dot = ref[:, 0] * vec[:, 0] + ref[:, 1] * vec[:, 1]
    return torch.atan2(cross, dot).unsqueeze(-1)



def to_frame_2D(vec: torch.Tensor, ref_from: torch.Tensor, ref_to: torch.Tensor) -> torch.Tensor:
    """Rotate a batch of 2D vectors from one coordinate frame to another. Rotating a vector by theta is equivalent to rotating the frame by -theta.

    Arguments:
        vec (torch.Tensor): A list of vectors, of dimension (n, 2).
        ref_from (torch.Tensor): A list of reference primary axis vectors for the current coordinate frame, of dimension n or (n, 2). These need not be normalized, though it is recommended.
        vec (torch.Tensor): A list of reference primary axis vectors for the desired coordinate frame, of dimension n or (n, 2). These need not be normalized, though it is recommended. Whether this is right- or left- handed must be consistent with ref_from, i.e., no reflections.

    Returns:
        torch.Tensor: The vectors in the new coordinate frame.
    """
    # Expand reference vectors if necessary
    if ref_from.shape[0] == 1 and vec.shape[0] > 1:
        ref_from = ref_from.repeat(vec.shape[0], 1)
    if ref_to.shape[0] == 1 and vec.shape[0] > 1:
        ref_to = ref_to.repeat(vec.shape[0], 1)
    theta = -get_angle_2D(ref_from, ref_to) # Angle between normal-tangential and global coordinate systems
    return rotate_2D(vec, theta) # Rotate displacements from global to normal-tangential frame