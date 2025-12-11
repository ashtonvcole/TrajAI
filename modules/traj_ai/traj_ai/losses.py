import torch
import torch.nn as nn



class StatePositionMeanSquarePositionLoss(nn.Module):
    """
    Compute the MSE position loss between a state and a position.
    
    Attributes:
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
    """

    def __init__(self, num_past: int) -> None:
        """Constructor for position mean square displacement error loss function.

        Arguments:
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.

        Returns:
            None
        """
        super(StatePositionMeanSquarePositionLoss, self).__init__()
        self.num_past = num_past

        # Pre-calculate indices based on state convention
        self.POS_START = 0
        self.POS_DIM = 2
        self.VEL_START = self.POS_START + (self.num_past + 1) * self.POS_DIM
        self.VEL_DIM = 2

    def forward(self, x_pred: torch.Tensor, pos_true: torch.Tensor) -> torch.Tensor:
        """Get mean square error position loss between a batch of states and positions.

        Arguments:
            x_pred (torch.Tensor): A tensor of predicted particle states, of dimension (num_particles, dim_particle_state).
            pos_true (torch.Tensor): A tensor of ground truth positions, of dimension (num_particles, dim_pos).

        Returns:
            torch.Tensor: The scalar loss.
        """
        pos_pred = x_pred[:, self.POS_START:(self.POS_START + self.POS_DIM)]
        return nn.functional.mse_loss(pos_pred, pos_true)



class StateStateMeanSquarePositionLoss(nn.Module):
    """
    Compute the MSE position loss between a state and a state.
    
    Attributes:
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
    """

    def __init__(self, num_past: int) -> None:
        """Constructor for position mean square displacement error loss function.

        Arguments:
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.

        Returns:
            None
        """
        super(StateStateMeanSquarePositionLoss, self).__init__()
        self.num_past = num_past

        # Pre-calculate indices based on state convention
        self.POS_START = 0
        self.POS_DIM = 2
        self.VEL_START = self.POS_START + (self.num_past + 1) * self.POS_DIM
        self.VEL_DIM = 2

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """Get mean square error position loss between batches of states.

        Arguments:
            x_pred (torch.Tensor): A tensor of predicted particle states, of dimension (num_particles, dim_particle_state).
            x_true (torch.Tensor): A tensor of ground truth states, of dimension (num_particles, dim_particle_state).

        Returns:
            torch.Tensor: The scalar loss.
        """
        pos_pred = x_pred[:, self.POS_START:(self.POS_START + self.POS_DIM)]
        pos_true = x_true[:, self.POS_START:(self.POS_START + self.POS_DIM)]
        return nn.functional.mse_loss(pos_pred, pos_true)



class StateVelocityMeanSquareVelocityLoss(nn.Module):
    """
    Compute the MSE position loss between a state and a velocity.
    
    Attributes:
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
    """

    def __init__(self, num_past: int) -> None:
        """Constructor for velocity mean square displacement error loss function.

        Arguments:
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.

        Returns:
            None
        """
        super(StateVelocityMeanSquareVelocityLoss, self).__init__()
        self.num_past = num_past

        # Pre-calculate indices based on state convention
        self.POS_START = 0
        self.POS_DIM = 2
        self.VEL_START = self.POS_START + (self.num_past + 1) * self.POS_DIM
        self.VEL_DIM = 2

    def forward(self, x_pred: torch.Tensor, vel_true: torch.Tensor) -> torch.Tensor:
        """Get mean square error velocity loss between a batch of states and velocities.

        Arguments:
            x_pred (torch.Tensor): A tensor of predicted particle states, of dimension (num_particles, dim_particle_state).
            vel_true (torch.Tensor): A tensor of ground truth velocities, of dimension (num_particles, dim_vel).

        Returns:
            torch.Tensor: The scalar loss.
        """
        vel_pred = x_pred[:, self.VEL_START:(self.VEL_START + self.VEL_DIM)]
        return nn.functional.mse_loss(vel_pred, vel_true)



class StateStateMeanSquareVelocityLoss(nn.Module):
    """
    Compute the MSE velocity loss between a state and a state.
    
    Attributes:
        num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.
    """

    def __init__(self, num_past: int) -> None:
        """Constructor for velocity mean square displacement error loss function.

        Arguments:
            num_past (int): The number of additional positions and velocities encoded into the state, besides the current ones.

        Returns:
            None
        """
        super(StateStateMeanSquareVelocityLoss, self).__init__()
        self.num_past = num_past

        # Pre-calculate indices based on state convention
        self.POS_START = 0
        self.POS_DIM = 2
        self.VEL_START = self.POS_START + (self.num_past + 1) * self.POS_DIM
        self.VEL_DIM = 2

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """Get mean square error velocity loss between batches of states.

        Arguments:
            x_pred (torch.Tensor): A tensor of predicted particle states, of dimension (num_particles, dim_particle_state).
            x_true (torch.Tensor): A tensor of ground truth states, of dimension (num_particles, dim_particle_state).

        Returns:
            torch.Tensor: The scalar loss.
        """
        vel_pred = x_pred[:, self.VEL_START:(self.VEL_START + self.VEL_DIM)]
        vel_true = x_true[:, self.VEL_START:(self.VEL_START + self.VEL_DIM)]
        return nn.functional.mse_loss(vel_pred, vel_true)