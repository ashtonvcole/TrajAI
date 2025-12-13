import torch
import torch.nn as nn



class StateComposer(nn.Module):
    """
    Convert simple state vectors to ones with memory from a certain number of past states.

    state_reduced = [x1[n], y1[n],
                     u1[n], v1[n],
                     additional, attributes, here, ...]

    becomes

    state = [x1[n], y1[n], ..., x1[n - num_past], y1[n - num_past],
             u1[n], v1[n], ..., u1[n - num_past], v1[n - num_past],
             additional, attributes, here, ...]
    
    Attributes:
        num_past (int): The number of past reduced states, besides the present one, which compose a full state.
    """

    def __init__(self, num_past: int) -> None:
        """Constructor for state composer.

        Arguments:
            num_past (int): The number of past reduced states, besides the present one, which compose a full state.
        
        Returns:
            None
        """
        super(StateComposer, self).__init__()
        self.num_past = num_past

        # Pre-calculate indices based on state convention
        self.POS_START = 0
        self.POS_DIM = 2
        self.VEL_START = self.POS_START + (self.num_past + 1) * self.POS_DIM
        self.VEL_DIM = 2
        self.ATT_START = self.VEL_START + (self.num_past + 1) * self.VEL_DIM
        self.POS_START_RED = 0
        self.VEL_START_RED = self.POS_START_RED + self.POS_DIM
        self.ATT_START_RED = self.VEL_START_RED + self.VEL_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compose full states from a tensor of reduced states.

        Arguments:
            x (torch.Tensor): A tensor of reduced particle states, of dimension (num_time_step, ..., dim_state_reduced).

        Returns:
            torch.Tensor: A tensor of full particle states, of dimension (num_time_step - num_past, ..., dim_state).
        """
        shape = x.shape # Extract x's shape
        num_time_step = shape[0]
        shape[0] -= self.num_past # Reduce time series along axis of composition
        shape[-1] += self.num_past * (self.POS_DIM + self.VEL_DIM) # Add slots for past positions and velocities
        x_new = torch.zeros(shape, device=x.device) # New states
        for i in range(self.num_past + 1):
            TIME_START = self.num_past - i
            TIME_STOP = num_time_step - i
            POS_START_FULL = self.POS_START + i * self.POS_DIM
            VEL_START_FULL = self.VEL_START + i * self.VEL_DIM
            x_new[:, :, POS_START_FULL:(POS_START_FULL + self.POS_DIM)] = x[TIME_START:TIME_STOP, :, self.POS_START_RED:(self.POS_START_RED + self.POS_DIM)] # Copy position
            x_new[:, :, VEL_START_FULL:(VEL_START_FULL + self.VEL_DIM)] = x[TIME_START:TIME_STOP, :, self.VEL_START_RED:(self.VEL_START_RED + self.VEL_DIM)] # Copy velocity
        x_new[:, :, self.ATT_START:] = x[self.num_past:, :, self.ATT_START_RED:] # Copy rest of state
        return x_new



class StateDecomposer(nn.Module):
    """
    Convert a full state vector to a reduced state vector.

    state = [x1[n], y1[n], ..., x1[n - (window - 1)], y1[n - (window - 1)],
             u1[n], v1[n], ..., u1[n - (window - 1)], v1[n - (window - 1)],
             additional, attributes, here, ...]

    becomes

    state_reduced = [x1[n], y1[n],
                     u1[n], v1[n],
                     additional, attributes, here, ...]
    
    Attributes:
        num_past (int): The number of past reduced states, besides the present one, which compose a full state.
    """

    def __init__(self, num_past: int) -> None:
        """Constructor for state decomposer.

        Arguments:
            num_past (int): The number of past reduced states, besides the present one, which compose a full state.
        
        Returns:
            None
        """
        super(StateDecomposer, self).__init__()
        self.num_past = num_past

        # Pre-calculate indices based on state convention
        self.POS_START = 0
        self.POS_DIM = 2
        self.VEL_START = self.POS_START + (self.num_past + 1) * self.POS_DIM
        self.VEL_DIM = 2
        self.ATT_START = self.VEL_START + (self.num_past + 1) * self.VEL_DIM
        self.POS_START_RED = 0
        self.VEL_START_RED = self.POS_START_RED + self.POS_DIM
        self.ATT_START_RED = self.VEL_START_RED + self.VEL_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compose reduced states from a tensor of full states.

        Arguments:
            x (torch.Tensor): A tensor of full particle states, of dimension (num_particles, dim_state).

        Returns:
            torch.Tensor: A tensor of full particle states, of dimension (num_particles, dim_state_reduced).
        """
        shape = x.shape # Extract x's shape
        shape[-1] -= self.num_past * (self.POS_DIM + self.VEL_DIM) # Remove slots for past positions and velocities
        x_new = torch.zeros(shape, device=x.device) # New states
        x_new[:, self.POS_START_RED:(self.POS_START_RED + self.POS_DIM)] = x[:, self.POS_START:(self.POS_START + self.POS_DIM)] # Copy current position
        x_new[:, self.VEL_START_RED:(self.VEL_START_RED + self.VEL_DIM)] = x[:, self.VEL_START:(self.VEL_START + self.VEL_DIM)] # Copy current velocity
        x_new[:, self.ATT_START_RED:] = x[:, self.ATT_START:] # Copy rest of state
        return x_new