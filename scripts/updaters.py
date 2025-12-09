import gns_tools as gt
import torch
import torch.nn as nn

class VelocityUpdater(nn.Module):
    """
    State update function using a velocity inductive bias.
    
    Assuming that the state of the system most directly determines particles' velocities, the next position and velocity are computed for the next state using constant-acceleration kinematic equations.
    
    Attributes:
        dt (float): Time step for numerical integration.
    """
    
    def __init__(self, dt: float) -> None:
        """Constructor for a velocity-based state updater.

        Arguments:
            dt (float): Time step for numerical integration.

        Returns:
            None
        """
        super(VelocityUpdater, self).__init__()
        self.dt = dt

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Update particle states based on an acceleration inductive bias.

    Assuming that the state of the system most directly determines particles' forces/accelerations, the next position and velocity are computed for the next state using constant-acceleration kinematic equations.

    Arguments:
        x (torch.Tensor): A tensor of particle states, of dimension (num_particles, dim_particle_state).
        y (torch.Tensor): A tensor of particle velocities, of dimension (num_particles, 2).

    Returns:
        torch.Tensor: The updated state, of dimension (num_particles, dim_particle_state)
    """
    x_new = x.clone() # Clone to preserve gradient calculations
    x_new[:, ???] += y * self.dt # Update position
    # Shift prior frames too
    x_new[:, ???] = y # Update velocity
    # Shift prior frames too
    return x_new

class AccelerationUpdater(nn.Module):
    """
    State update function using an acceleration inductive bias.
    
    Assuming that the state of the system most directly determines particles' forces/accelerations, the next position and velocity are computed for the next state using constant-acceleration kinematic equations.
    
    Attributes:
        dt (float): Time step for numerical integration.
    """
    
    def __init__(self, dt: float) -> None:
        """Constructor for an acceleration-based state updater.

        Arguments:
            dt (float): Time step for numerical integration.

        Returns:
            None
        """
        super(AccelerationUpdater, self).__init__()
        self.dt = dt

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Update particle states based on an acceleration inductive bias.

    Assuming that the state of the system most directly determines particles' forces/accelerations, the next position and velocity are computed for the next state using constant-acceleration kinematic equations.

    Arguments:
        x (torch.Tensor): A tensor of particle states, of dimension (num_particles, dim_particle_state).
        y (torch.Tensor): A tensor of particle accelerations, of dimension (num_particles, 2).

    Returns:
        torch.Tensor: The updated state, of dimension (num_particles, dim_particle_state)
    """
    x_new = x.clone() # Clone to preserve gradient calculations
    v0 = x[:, ???]
    x_new[:, ???] += v0 * self.dt + 1/2 * y * self.dt ** 2 # Update position
    # Shift prior frames too
    x_new[:, ???] += y * self.dt # Update velocity
    # Shift prior frames too
    return x_new