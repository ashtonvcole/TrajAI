import torch
import torch.nn as nn

class RectNN(nn.Module):
    """A sequential MLP with a fixed width for all hidden layers."""

    def __init__(self, input_dim: int, output_dim: int, width: int, depth: int, activation_type: nn.Module) -> None:
        """Constructor for a RectNN.
    
        Arguments:
            input_dim (int): Dimension of input.
            output_dim (int): Dimension of output.
            width (int): Number of perceptrons per hidden layer.
            depth (int): Number of hidden layers.
            activation_type (nn.Module): Type of activation function to be instantiated for each layer.
    
        Returns:
            None
        """
        super(RectNN, self).__init__()
        layers = nn.ModuleList()
        if depth == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, width)) # input to hidden 1
            layers.append(activation_type())
            for i in range(depth - 1):
                layers.append(nn.Linear(width, width)) # hidden i to hidden i + 1
                layers.append(activation_type())
            layers.append(nn.Linear(width, output_dim)) # hidden depth to output
        self.network = nn.Sequential(*layers) # Construct sequential network from list of layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the neural network to a batch of data.

        Arguments:
            x (torch.Tensor): Inputs to the neural network. This may be either a single input of dimension (input_dim), or a batch of dimension (n, input_dim).

        Returns:
            torch.Tensor: The output of the neural netowrk, either of dimension (output_dim) or (n, output_dim).
        """
        return self.network(x)