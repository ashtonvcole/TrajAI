import .gnn
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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



def rollout(simulator: GraphNeuralSimulator, x: torch.Tensor, num_pred: int) -> torch.Tensor:
    """Apply a Graph Neural Simulator in a rollout fashion, to predict future states of a set of particles.

    Arguments:
        simulator (GraphNeuralSimulator): A Graph Neural Simulator, which acts as a state tranition function.
        x (torch.Tensor): A list of particle states, of dimension (num_states, num_particles, dim_state).
        num_pred (int): The number of forward predictions to make.

    Returns:
        torch.Tensor: The full time series of system states, of dimenion (num_states + num_pred, num_particles, dim_state).
    """
    model.eval()
    x_series = x.clone() # Clone to preserve gradient calculations
    for i in range(num_pred):
        with torch.no_grad():
            x = x_series[-1, :, :] # Current global state, of dimension (num_particles, dim_state)
            x_new = simulator(x) # Get new global state
            x_series = torch.cat((x_series, x_new.unsqueeze(0)) dim=0)
    return x_series



def rollout_reduced(simulator: GraphNeuralSimulator, x_rollout: torch.Tensor, num_pred: int, state_composer: nn.Module, state_decomposer: nn.Module) -> torch.Tensor:
    """Apply a Graph Neural Simulator in a rollout fashion, to predict future states of a set of particles.

    Reduced rollout is useful when a Graph Neural Simulator incorporates information about past states, i.e. memory, into the forward prediction. The variable x_rollout does not hold the full states, but rather strictly the information associated with the present. Thus, to apply the simulator, full sates need to be constructed using the stae composer.

    Arguments:
        simulator (GraphNeuralSimulator): A Graph Neural Simulator, which acts as a state tranition function.
        x_rollout (torch.Tensor): A list of reduced particle states, of dimension (num_states, num_particles, dim_state_reduced). Note that num_state must be large enough for the state_composer.
        num_pred (int): The number of forward predictions to make.
        state_composer (nn.Module): A function which converts a window of reduced states to a full state of dimension dim_state. Must have an attribute window (int) which holds the total number of reduced states are used to compose the full state.
        state_decomposer (nn.Module): A function which converts a full state of dimension dim_state to a window of reduced states, of dimension (window, num_particles, dim_state_rediced).

    Returns:
        torch.Tensor: The full time series of system states, of dimenion (num_states + num_pred, num_particles, dim_state_reduced).
    """
    model.eval()
    window = state_composer.window # How many reduced states are embedded in a single particle state
    x_series = x.clone() # Clone to preserve gradient calculations
    for i in range(num_pred):
        with torch.no_grad():
            x = state_composer(x_series[-window:, :, :]) # Current global state, of dimension (num_particles, dim_state)
            x_new = simulator(x) # Get new global state
            x_new_reduced = state_decomposer(x_new)[-1, :, :] # Extract only most recent reduced state from the window
            x_series = torch.cat((x_series, x_new_reduced.unsqueeze(0)) dim=0) # Append to time series
    return x_series