import gnn
import torch
import torch.nn as nn

class GraphNeuralSimulator(nn.Module):
    """The state transition function of a system, composed of a Graph Neural Network and an updater."""

    def __init__(self, graph_neural_network: gnn.GraphNeuralNetwork, updater: nn.Module, connector = None) -> None:
        """Constructor for a Graph Neural simulator.

        As opposed to a Graph Neural Network, the simulator consists of both a network and an updater function. Note that the updater function should not be trainable, but rather something analytical representing an inductive bias. For example, maybe the Graph Neural Network learns an acceleration, and the updater translates this to a new position using numerical integration.

        Arguments:
            graph_neural_network (gnn.GraphNeuralNetwork): A Graph Neural Network with input dimension x_dim and output dimension y_dim.
            updater (nn.Module): An appropriate Pytorch module taking in node states of dimension x_dim, and node updates of dimension y_dim, producing a new state of dimension x_dim.
            connector (optional): An appropriate function or non-trainable Pytorch module taking in particle states and determining which ones influence others, e.g., by a distance threshold. This should return a tensor of dimension (2, num_edges). If not specified or None, it is assumed that all particles influence each other's states, and the latent graph will be fully connected. For large particle simulations, this will worsen performance.

        Returns:
            None
        """
        super(GraphNeuralSimulator, self).__init__()
        self.graph_neural_network = graph_neural_network
        self.updater = updater
        self.connector = connector

    def forward(self, x: torch.Tensor):
        """Apply the Graph Neural Simulator to a batch of data.

        This method acts as a state transition function. It is given a state and predicts the next state in a sequence.
        
        Arguments:
            x (torch.Tensor): A batch of particle states composing the global state of the system, of dimension (num_particles, dim_particle_state).

        Returns:
            torch.Tensor: The output of the Graph Neural Simulator.
        """
        connectivity = None
        if self.connector is not None:
            connectivity = self.connector(x)
        return self.updater(x, self.graph_neural_network(x, connectivity))