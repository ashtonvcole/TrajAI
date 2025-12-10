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