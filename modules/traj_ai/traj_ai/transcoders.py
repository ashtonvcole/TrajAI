from .batch_geometry import get_angle_2D, rotate_2D, to_frame_2D
import torch
import torch.nn as nn



class NormalTangentialObjectiveStateTranscoder(nn.Module):
    """
    Convert particle state to frame-indifferent form.

    The state of a particle is assumed to be composed of a series of positions and velocities, as well as other data. It is then translated into a new state which is invariant to coordinate frame translations or rotations.

    state = [x[n], y[n], ..., x[n - num_past], y[n - num_past],
             u[n], v[n], ..., u[n - num_past], v[n - num_past],
             additional, attributes, here, ...]

    state_invariant = [xt[1], xn[1], ..., xt[num_past], xn[num_past],
                       v,
                       vt[1], vn[1], ..., vt[num_past], vn[num_past],
                       additional, attributes, here, ...]

    xt[i]: The tangential component of the displacement between frames n and n - i.
    xn[i]: The normal component of the displacement between frames n and n - i.
    v: The velocity magnitude at frame n.
    vt[i]: The tangential component of the velocity at frame n - i.
    vn[i]: The normal component of the velocity at frame n - i.

    Attributes:
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
        epsilon (float): Adjustment facor used to handle zero velocities and displacements. This should be a few orders of magnitude smaller than typical velocities.
    """

    def __init__(self, num_past: int, epsilon: float = 1e-6) -> None:
        """Constructor for a normal-tangential translationally and rotationally invariant node transcoder.

        Arguments:
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
            epsilon (float, optional): Adjustment facor used to handle zero velocities and displacements. This should be a few orders of magnitude smaller than typical velocities. Default is 1e-6.
        
        Returns:
            None
        """
        super(NormalTangentialObjectiveStateTranscoder, self).__init__()
        self.num_past = num_past
        self.epsilon = epsilon # To handle zero velocity/displacement, note that this is much smaller than a pixel
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert particle states to a translationally and rotationally invariant form. 
        
        Arguments:
            x (torch.Tensor): A tensor of particle states, of dimension (num_particles, dim_particle_state).

        Returns:
            torch.Tensor: A tensor of invariant particle states, of dimension (num_particles, dim_particle_state - 3).
        """
        # Reference indices
        POS_START = 0
        POS_DIM = 2
        VEL_START = POS_START + (self.num_past + 1) * POS_DIM
        VEL_DIM = 2
        ATT_START = VEL_START + (self.num_past + 1) * VEL_DIM
        ATT_LEN = x.shape[1] - ATT_START
        P_START = 0
        V_START = P_START + self.num_past * POS_DIM
        A_START = V_START + 1 + self.num_past * VEL_DIM # Since keep v[n]
        XN_DIM = self.num_past * POS_DIM + 1 + self.num_past * VEL_DIM + ATT_LEN
        
        # Create new state
        x_new = torch.zeros((x.shape[0], XN_DIM), device=x.device)
        
        # Get present position and velocity
        pos_present = x[:, POS_START:(POS_START + POS_DIM)]
        vel_present = x[:, VEL_START:(VEL_START + VEL_DIM)]
        x_new[:, V_START] = torch.norm(vel_present)
        
        for i in range(self.num_past):
            POS_PAST_START = POS_START + (i + 1) * POS_DIM
            VEL_PAST_START = VEL_START + (i + 1) * VEL_DIM
            P_PAST_START = P_START + i * POS_DIM
            V_PAST_START = V_START + 1 + i * VEL_DIM # Since keep v[n]
            
            # Get past position and velocity
            pos_past = x[:, POS_PAST_START:(POS_PAST_START + POS_DIM)]
            vel_past = x[:, VEL_PAST_START:(VEL_PAST_START + VEL_DIM)]
            
            # Objective position
            x_new[:, P_PAST_START:(P_PAST_START + POS_DIM)] = self._get_objective_position(pos_present, pos_past, vel_present)
            # Objective velocity
            
            x_new[:, V_PAST_START:(V_PAST_START + VEL_DIM)] = self._get_objective_velocity(vel_present, vel_past)
        
        # Copy additional attributes
        x_new[:, A_START:] = x[:, ATT_START:]
        return x_new

    def _get_objective_position(self, pos_present: torch.Tensor, pos_past: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Get relative position in normal-tangential coordinates (dt, dn).

        Arguments:
            pos_present (torch.Tensor): The current position, of dimension (num_particles, 2).
            pos_past (torch.Tensor): The past position, of dimension (num_particles, 2).
            ref (torch.Tensor): A reference direction for the local coordinate system, e.g., the tangent (normalized velocity).

        Returns:
            torch.Tensor: Rotation-invariant displacement (displacement_tangential, displacement_normal), of dimension (num_particles, 2)
        """
        ref = ref / (torch.norm(ref, dim=1, keepdim=True) + self.epsilon) # Normalize reference vector
        e1 = torch.tensor([1.0, 0.0], dtype=ref.dtype, device=ref.device).repeat(ref.shape[0], 1) # Global frame reference vector
        return to_frame_2D(pos_present - pos_past, e1, ref) # Rotate displacements from global to normal-tangential frame

    def _get_objective_velocity(self, vel_present: torch.Tensor, vel_past: torch.Tensor) -> torch.Tensor:
        """Get velocity in normal-tangential coordinates (vt, vn).

        Arguments:
            vel_present (torch.Tensor): The current (reference) velocity, of dimension (num_particles, 2).
            vel_past (torch.Tensor): The past velocity, of dimension (num_particles, 2).

        Returns:
            torch.Tensor: Rotation-invariant velocity (velocity_tangential, velocity_normal), of dimension (num_particles, 2)
        """
        ref = vel_present / (torch.norm(vel_present, dim=1, keepdim=True) + self.epsilon) # Normalize reference vector
        e1 = torch.tensor([1.0, 0.0], dtype=ref.dtype, device=ref.device).repeat(ref.shape[0], 1) # Global frame reference vector
        return to_frame_2D(vel_past, e1, ref) # Rotate displacements from global to normal-tangential frame



class PolarObjectiveStateTranscoder(nn.Module):
    """
    Convert particle state to frame-indifferent form.

    The state of a particle is assumed to be composed of a series of positions and velocities, as well as other data. It is then translated into a new state which is invariant to coordinate frame translations or rotations.

    state = [x[n], y[n], ..., x[n - num_past], y[n - num_past],
             u[n], v[n], ..., u[n - num_past], v[n - num_past],
             additional, attributes, here, ...]

    state_invariant = [r[1], theta[1], ..., r[num_past], theta[num_past],
                       v,
                       R[1], Theta[1], ..., R[num_past], Theta[num_past],
                       additional, attributes, here, ...]

    r[i]: The norm of the displacement between frames n and n - i.
    theta[i]: The angle between the two vectors in radians.
    v: The velocity magnitude at frame n.
    R[i]: The norm of the vector velocity difference between frames n and n - i.
    Theta[i]: The angle between the two vectors in radians.

    Attributes:
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
        epsilon (float): Adjustment facor used to handle zero velocities and displacements. This should be a few orders of magnitude smaller than typical velocities.
    """

    def __init__(self, num_past: int, epsilon: float = 1e-6) -> None:
        """Constructor for a polar translationally and rotationally invariant node transcoder.

        Arguments:
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
            epsilon (float, optional): Adjustment facor used to handle zero velocities and displacements. This should be a few orders of magnitude smaller than typical velocities. Default is 1e-6.
        
        Returns:
            None
        """
        super(PolarObjectiveStateTranscoder, self).__init__()
        self.num_past = num_past
        self.epsilon = epsilon # To handle zero velocity/displacement, note that this is much smaller than a pixel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert particle states to a translationally and rotationally invariant form. 
        
        Arguments:
            x (torch.Tensor): A tensor of particle states, of dimension (num_particles, dim_particle_state).

        Returns:
            torch.Tensor: A tensor of invariant particle states, of dimension (num_particles, dim_particle_state - 3).
        """
        # Reference indices
        POS_START = 0
        POS_DIM = 2
        VEL_START = POS_START + (self.num_past + 1) * POS_DIM
        VEL_DIM = 2
        ATT_START = VEL_START + (self.num_past + 1) * VEL_DIM
        ATT_LEN = x.shape[1] - ATT_START
        P_START = 0
        V_START = P_START + self.num_past * POS_DIM
        A_START = V_START + self.num_past * VEL_DIM
        XN_DIM = self.num_past * POS_DIM + 1 + self.num_past * VEL_DIM + ATT_LEN
        
        # Create new state
        x_new = torch.zeros((x.shape[0], XN_DIM), device=x.device)
        
        # Get present position and velocity
        pos_present = x[:, POS_START:(POS_START + POS_DIM)]
        vel_present = x[:, VEL_START:(VEL_START + VEL_DIM)]
        x_new[:, V_START] = torch.norm(vel_present)
        
        for i in range(self.num_past):
            POS_PAST_START = POS_START + (i + 1) * POS_DIM
            VEL_PAST_START = VEL_START + (i + 1) * VEL_DIM
            P_PAST_START = P_START + i * POS_DIM
            V_PAST_START = V_START + 1 + i * VEL_DIM # Since keep v[n]
            
            # Get past position and velocity
            pos_past = x[:, POS_PAST_START:(POS_PAST_START + POS_DIM)]
            vel_past = x[:, VEL_PAST_START:(VEL_PAST_START + VEL_DIM)]
            
            # Objective position
            x_new[:, P_PAST_START:(P_PAST_START + POS_DIM)] = self._get_objective_position(pos_present, pos_past, vel_present)
            # Objective velocity
            
            x_new[:, V_PAST_START:(V_PAST_START + VEL_DIM)] = self._get_objective_velocity(vel_present, vel_past)
            
        # Copy additional attributes
        x_new[:, A_START:] = x[:, ATT_START:]
        return x_new

    def _get_objective_position(self, pos_present: torch.Tensor, pos_past: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Get relative position in polar coordinates (distance, angle_relative_to_tangent).

        Arguments:
            pos_present (torch.Tensor): The current position, of dimension (num_particles, 2).
            pos_past (torch.Tensor): The past position, of dimension (num_particles, 2).
            ref (torch.Tensor): A reference direction for the local coordinate system, e.g., the tangent (normalized velocity).

        Returns:
            torch.Tensor: Rotation-invariant displacement (distance, angle_relative_to_tangent), of dimension (num_particles, 2)
        """
        ref = ref / (torch.norm(ref, dim=1, keepdim=True) + self.epsilon) # Normalize reference vector
        rt = torch.zeros_like(pos_present)
        disp = pos_present - pos_past
        dist = torch.norm(disp, dim=1)
        rt[:, 0] = dist # Distance
        rt[:, 1] = get_angle_2D(ref, disp) # Angle
        return rt

    def _get_objective_velocity(self, vel_present: torch.Tensor, vel_past: torch.Tensor) -> torch.Tensor:
        """Get relative velocity change in polar coordinates (norm_of_difference, angle).

        Arguments:
            vel_present (torch.Tensor): The current (reference) velocity, of dimension (num_particles, 2).
            vel_past (torch.Tensor): The past velocity, of dimension (num_particles, 2).

        Returns:
            torch.Tensor: Rotation-invariant velocity change (norm_of_difference, angle), of dimension (num_particles, 2)
        """
        rt = torch.zeros_like(vel_present)
        rt[:, 0] = torch.norm(vel_present - vel_past, dim=1) # Norm of velocity change (displacement in velocity space)
        rt[:, 1] = get_angle_2D(vel_present, vel_past) # Angle
        return rt