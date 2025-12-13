from .batch_geometry import to_frame_2D
import torch
import torch.nn as nn



class NormalTangentialDistanceObjectiveStateRelater(nn.Module):
    """
    Get a frame-invariant particle-particle relationship, from the perspective of the second, i.e., influenced, particle.

    state1 = [x1[n], y1[n], ..., x1[n - num_past], y1[n - num_past],
              u1[n], v1[n], ..., u1[n - num_past], v1[n - num_past],
              additional, attributes, here, ...]

    state2 = [x2[n], y2[n], ..., x2[n - num_past], y2[n - num_past],
              u2[n], v2[n], ..., u2[n - num_past], v2[n - num_past],
              additional, attributes, here, ...]

    become

    relation12 = [dxt12[n], dxn12[n], ..., dxt12[n - num_past], dxn12[n - num_past],
                  dvt12[n], dvn12[n], ..., dvt12[n - num_past], dvn12[n - num_past],
                  d12[n], ..., d12[n - num_past]]
    
    dxt12[i]: The tangential component of the displacement of 2 from 1 at frame i.
    dxn12[i]: The normal component of the displacement of 2 from 1 at frame i.
    dvt12[i]: The tangential component of the velocity difference of 2 from 1 at frame i.
    dvn12[i]: The normal component of the velocity difference of 2 from 1 at frame i.
    (from the normal-tangential frame of 2, the influenced object)

    Attributes:
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
        epsilon (float): Adjustment facor used to handle zero velocities and displacements. This should be a few orders of magnitude smaller than typical velocities.
    """

    def __init__(self, num_past: int, epsilon: float = 1e-6) -> None:
        """Constructor for a normal-tangential based objective relation between particle states.

        Arguments:
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
            epsilon (float, optional): Threshold for zero velocities, below which an e1 = (0, 1) tangent is assumed. This should be a few orders of magnitude smaller than typical velocities.

        Returns:
        """
        super(NormalTangentialDistanceObjectiveStateRelater, self).__init__()
        self.num_past = num_past
        self.epsilon = epsilon # To handle zero velocity/displacement, note that this is much smaller than a pixel

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Determine the dynamics relationship between two particle states.

        Arguments:
            x_i (torch.Tensor): A tensor of influencer particle states, of dimension (num_particles, dim_particle_state).
            x_j (torch.Tensor): A tensor of influenced particle states, of dimension (num_particles, dim_particle_state).

        Returns:
            torch.Tensor: A tensor of corresponding influencer-influenced relationship vectors, of dimension (num_particles, 10 * (num_past + 1).
        """
        # Reference indices
        POS_START = 0
        POS_DIM = 2
        VEL_START = POS_START + (self.num_past + 1) * POS_DIM
        VEL_DIM = 2
        ATT_START = VEL_START + (self.num_past + 1) * VEL_DIM
        DX_START = 0
        DV_START = DX_START + (self.num_past + 1) * POS_DIM
        D_START = DV_START + (self.num_past + 1) * VEL_DIM
        D_DIM = 1
        R_DIM = (self.num_past + 1) * (POS_DIM + VEL_DIM + D_DIM)

        # Create relation tensor
        r = torch.zeros((x_i.shape[0], R_DIM), device=x_i.device)

        dx = x_j[:, POS_START:VEL_START] - x_i[:, POS_START:VEL_START] # Displacement at num_past + 1 steps
        dv = x_j[:, VEL_START:ATT_START] - x_i[:, VEL_START:ATT_START] # Velocity at num_past + 1 steps
        e1 = torch.tensor([1.0, 0.0], dtype=x_j.dtype, device=x_j.device).repeat(x_j.shape[0], 1) # Global frame reference vector

        # Iteratively reframe displacements, velocities, and distances
        for i in range(self.num_past + 1):
            POS_LOCAL = POS_START + i * POS_DIM
            VEL_LOCAL = VEL_START + i * VEL_DIM
            DX_LOCAL = DX_START + i * POS_DIM
            DV_LOCAL = DV_START + i * VEL_DIM
            D_LOCAL = D_START + i * D_DIM
            DX2_START = i * POS_DIM # For dx tensor, not relater
            DV2_START = i * VEL_DIM # For dv tensor, not relater

            # Get tangent from the influenced particle
            ref = x_j[:, VEL_LOCAL:(VEL_LOCAL + VEL_DIM)]
            speeds = torch.norm(ref, dim=1, keepdim=True)
            stable_mask = (speeds > self.epsilon).squeeze(-1) # Turn into bool vector of dimension (num_particles)
            ref[stable_mask, :] = ref[stable_mask, :] / (speeds[stable_mask]) # Normalize reference vector where speed is sufficiently large
            ref[~stable_mask, :] = e1[~stable_mask, :] # Otherwise just use e1
            
            r[:, DX_LOCAL:(DX_LOCAL + POS_DIM)] = to_frame_2D(dx[:, DX2_START:(DX2_START + POS_DIM)], e1, ref) # Rotate displacements from global to normal-tangential frame
            r[:, DV_LOCAL:(DV_LOCAL + VEL_DIM)] = to_frame_2D(dv[:, DV2_START:(DV2_START + VEL_DIM)], e1, ref) # Rotate velocities from global to normal-tangential frame

            # Distance
            r[:, D_LOCAL:(D_LOCAL + D_DIM)] = torch.norm(dx[:, DX2_START:(DX2_START + POS_DIM)], dim=1, keepdim=True)

        # Use other parts of state as needed
        # Nothing for now, since the dynamics is what we're interested in!
        return r