import torch
import torch.nn as nn



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



class InvariantStateTranscoder(nn.Module):
    """
    Convert particle state to frame-indifferent form.

    The state of a particle is assumed to be composed of a series of positions and velocities, as well as other data. It is then translated into a new state which is invariant to coordinate frame translations or rotations.

    state = [x[n], y[n], ..., x[n - num_past], y[n - num_past],
             u[n], v[n], ..., u[n - num_past], v[n - num_past],
             additional, attributes, here, ...]

    state_invariant = [r[1], theta[1], ..., r[num_past], theta[num_past]
                       R[1], Theta[1], ..., r[num_past], theta[num_past],
                       additional, attributes, here, ...]

    r[i]: The norm of the displacement between frames n and n - i.
    theta[i]: The angle between the two vectors in radians.
    R[i]: The norm of the vector velocity difference between frames n and n - i.
    Theta[i]: The angle between the two vectors in radians.

    Attributes:
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
    """

    def __init__(self, num_past: int, epsilon: float = 1e-6) -> None:
        """Constructor for translationally and rotationally invariant node transcoder.

        Arguments:
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
            eps (float): Adjustment facor used to handle zero velocities and displacements. This should be a few orders of magnitude smaller than typical velocities.
        
        Returns:
            None
        """
        super(InvariantStateTranscoder, self).__init__()
        self.num_past = num_past
        self.epsilon = epsilon # To handle zero velocity/displacement, note that this is much smaller than a pixel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert particle states to a translationally and rotationally invariant form. 
        
        Arguments:
            x (torch.Tensor): A tensor of particle states, of dimension (num_particles, dim_particle_state).

        Returns:
            torch.Tensor: A tensor of invariant particle states, of dimension (num_particles, dim_particle_state - 2).
        """
        # Reference indices
        POS_START = 0
        POS_DIM = 2
        VEL_START = POS_START + (self.num_past + 1) * POS_DIM
        VEL_DIM = 2
        ATT_START = VEL_START + (self.num_past + 1) * VEL_DIM
        P_START = 0
        V_START = P_START + self.num_past * POS_DIM
        A_START = V_START + self.num_past * VEL_DIM
        # Create new state
        x_new = torch.zeros((x.shape[0], x.shape[1] - 2), device=x.device)
        # Get present position and velocity
        pos_present = x[:, POS_START:(POS_START + POS_DIM)]
        vel_present = x[:, VEL_START:(VEL_START + VEL_DIM)]
        for i in range(self.num_past):
            POS_PAST_START = POS_START + (i + 1) * POS_DIM
            VEL_PAST_START = VEL_START + (i + 1) * VEL_DIM
            P_PAST_START = P_START + i * POS_DIM
            V_PAST_START = V_START + i * VEL_DIM
            # Get past position and velocity
            pos_past = x[:, POS_PAST_START:(POS_PAST_START + POS_DIM)]
            vel_past = x[:, VEL_PAST_START:(VEL_PAST_START + VEL_DIM)]
            # Invariant position
            x_new[:, P_PAST_START:(P_PAST_START + POS_DIM)] = self._get_invariant_position(pos_present, pos_past, vel_present)
            # Invariant velocity
            x_new[:, V_PAST_START:(V_PAST_START + VEL_DIM)] = self._get_invariant_velocity(vel_present, vel_past)
        # Copy additional attributes
        x_new[:, A_START:] = x[:, ATT_START:]
        return x_new

    def _get_invariant_position(self, pos_present: torch.Tensor, pos_past: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Get relative position in polar coordinates (distance, angle_relative_to_tangent).

        Arguments:
            pos_present (torch.Tensor): The current position, of dimension (num_particles, 2).
            pos_past (torch.Tensor): The past position, of dimension (num_particles, 2).
            ref (torch.Tensor): A reference direction for the local coordinate system, e.g., the tangent (normalized velocity).

        Returns:
            torch.Tensor: Rotation-invariant displacement (distance, angle_relative_to_tangent), of dimension (num_particles, 2)
        """
        ref = ref / (torch.norm(ref, dim=1, keepdim=True) + self.epsilon) # Normalize reference vector
        rt = torch.zeros(pos_present.shape, device=pos_present.device)
        disp = pos_present - pos_past
        dist = torch.norm(disp, dim=1)
        rt[:, 0] = dist # Distance
        rt[:, 1] = get_angle_2D(ref, disp) # Angle
        return rt

    def _get_invariant_velocity(self, vel_present: torch.Tensor, vel_past: torch.Tensor) -> torch.Tensor:
        """Get relative velocity change in polar coordinates (norm_of_difference, angle).

        Arguments:
            vel_present (torch.Tensor): The current velocity, of dimension (num_particles, 2).
            vel_past (torch.Tensor): The past velocity, of dimension (num_particles, 2).

        Returns:
            torch.Tensor: Rotation-invariant velocity change (norm_of_difference, angle), of dimension (num_particles, 2)
        """
        rt = torch.zeros(vel_present.shape, device=vel_present.device)
        rt[:, 0] = torch.norm(vel_present - vel_past, dim=1) # Norm of velocity change (displacement in velocity space)
        rt[:, 1] = get_angle_2D(vel_present, vel_past) # Angle
        return rt