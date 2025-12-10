from .batch_geometry import to_frame_2D
import torch
import torch.nn as nn


def roll_state(x: torch.Tensor, num_past: int):
    """Push past positions and velocities to the n - 1 frame.

    state = [x[n], y[n], ..., x[n - num_past], y[n - num_past],
             u[n], v[n], ..., u[n - num_past], v[n - num_past],
             additional, attributes, here, ...]

    becomes

    state = [x[n], y[n], x[n], y[n], ..., x[n - num_past + 1], y[n - num_past + 1],
             u[n], v[n], u[n], v[n], ..., u[n - num_past + 1], v[n - num_past + 1],
             additional, attributes, here, ...]

    Arguments:
        x (torch.Tensor): A tensor of particle states, of dimension (num_particles, dim_particle_state).
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.

    Returns:
        torch.Tensor: An intermediate adjustment to the state, where the current positon is shifted to the n - 1 position, the n - 1 to the n - 2, and so on.
    """
    if num_past == 0:
        return x # Don't waste your breath
    x_new = x.clone() # Clone to preserve gradient calculations
    # Reference indices
    POS_START = 0
    POS_DIM = 2
    VEL_START = POS_START + (num_past + 1) * POS_DIM
    VEL_DIM = 2
    ATT_START = VEL_START + (num_past + 1) * VEL_DIM
    # Roll positions
    x_new[:, (POS_START + POS_DIM):(POS_START + (num_past + 1) * POS_DIM)] = x[:, POS_START:(POS_START + num_past * POS_DIM)]
    # Roll velocities
    x_new[:, (VEL_START + VEL_DIM):(VEL_START + (num_past + 1) * VEL_DIM)] = x[:, VEL_START:(VEL_START + num_past * VEL_DIM)]
    return x_new



class LocalVelocityUpdater(nn.Module):
    """
    State update function using a static prior inductive bias and a local coordinate system.
    
    Assuming that the state of the system most directly determines particles' velocities, the next position and velocity are computed for the next state using constant-velocity kinematic equations. To preserve rotational invariance, the updated velocity is given in a normal-tangential coordinate system.
    
    Attributes:
        dt (float): The time step for numerical integration.
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
    """
    
    def __init__(self, dt: float, num_past: int) -> None:
        """Constructor for a static prior velocity-based state updater.

        Arguments:
            dt (float): The time step for numerical integration.
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.

        Returns:
            None
        """
        super(LocalVelocityUpdater, self).__init__()
        self.dt = dt
        self.num_past = num_past

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Update particle states based on an velocity inductive bias.

    Assuming that the state of the system most directly determines particles' velocities, the next position and velocity are computed for the next state using constant-velocity kinematic equations.

    Arguments:
        x (torch.Tensor): A tensor of particle states, of dimension (num_particles, dim_particle_state).
        y (torch.Tensor): A tensor of new particle velocities, in a normal-tangential frame relative to the current velocity (v_tangential, v_normal), of dimension (num_particles, 2).

    Returns:
        torch.Tensor: The updated state, of dimension (num_particles, dim_particle_state)
    """
    # Reference indices
    POS_START = 0
    POS_DIM = 2
    VEL_START = POS_START + (self.num_past + 1) * POS_DIM
    VEL_DIM = 2
    ATT_START = VEL_START + (self.num_past + 1) * VEL_DIM
    
    # Process velocity
    x0 = x[:, POS_START:(POS_START + POS_DIM)] # For later
    v0 = x[:, VEL_START:(VEL_START + VEL_DIM)] # Tangent
    e1 = torch.tensor([1.0, 0.0], dtype=x.dtype, device=x.device).repeat(x.shape[0], 1) # Global frame reference vector
    v = to_frame_2D(y, v0, e1) # Rotate new velocities from normal/tangential to global frame

    # Update state
    x_new = roll_state(x, self.num_past) # Shift prior frames, cloning in the process
    x_new[:, POS_START:(POS_START + POS_DIM)] = x0 + v * self.dt # Update position
    x_new[:, VEL_START:(VEL_START + VEL_DIM)] = v # Update velocity
    # Update other parts of state as needed
    # Nothing for now, since the dynamics is what we're interested in!
    return x_new



class LocalAccelerationUpdater(nn.Module):
    """
    State update function using an inertial prior inductive bias and a local coordinate system.
    
    Assuming that the state of the system most directly determines particles' forces/accelerations, the next position and velocity are computed for the next state using constant-acceleration kinematic equations. To preserve rotational invariance, the updated acceleration is given in a normal-tangential coordinate system.
    
    Attributes:
        dt (float): Time step for numerical integration.
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
    """
    
    def __init__(self, dt: float, num_past: int) -> None:
        """Constructor for an acceleration-based state updater.

        Arguments:
            dt (float): Time step for numerical integration.
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.

        Returns:
            None
        """
        super(LocalAccelerationUpdater, self).__init__()
        self.dt = dt
        self.num_past = num_past

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Update particle states based on an acceleration inductive bias.

    Assuming that the state of the system most directly determines particles' forces/accelerations, the next position and velocity are computed for the next state using constant-acceleration kinematic equations.

    Arguments:
        x (torch.Tensor): A tensor of particle states, of dimension (num_particles, dim_particle_state).
        y (torch.Tensor): A tensor of particle accelerations, in a normal-tangential frame relative to the current velocity (a_tangential, a_normal), of dimension (num_particles, 2).

    Returns:
        torch.Tensor: The updated state, of dimension (num_particles, dim_particle_state)
    """
    # Reference indices
    POS_START = 0
    POS_DIM = 2
    VEL_START = POS_START + (self.num_past + 1) * POS_DIM
    VEL_DIM = 2
    ATT_START = VEL_START + (self.num_past + 1) * VEL_DIM

    # Process acceleration
    x0 = x[:, POS_START:(POS_START + POS_DIM)] # For later
    v0 = x[:, VEL_START:(VEL_START + VEL_DIM)] # Tangent
    e1 = torch.tensor([1.0, 0.0], dtype=x.dtype, device=x.device).repeat(x.shape[0], 1) # Global frame reference vector
    a = to_frame_2D(y, v0, e1) # Rotate new accelerations from normal/tangential to global frame

    # Update state
    x_new = roll_state(x, self.num_past) # Shift prior frames, cloning in the process
    x_new[:, POS_START:(POS_START + POS_DIM)] = x0 + v0 * self.dt + 1/2 * a * self.dt ** 2 # Update position
    x_new[:, VEL_START:(VEL_START + VEL_DIM)] = v0 +  a * self.dt # Update velocity
    # Update other parts of state as needed
    # Nothing for now, since the dynamics is what we're interested in!
    return x_new