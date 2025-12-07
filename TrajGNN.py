# Master GNN class
# Has encoder (sequential), processor (gn), decoder (sequential)
# Apply inductive bias appropriately? or in notebook?
# GN processor class too
# Update which does message passing M times
# Loss functions
# Approximators for different inductive biases




import numpy as np # Do I really need this?
import torch
import torch_geometric as pyg
import torch.nn as nn

class GNN(nn.Module):
    """Graph Neural Network with encoder, processor, decoder, and updater."""
    
    def __init__(self, encoder: nn.Module, processor: GN, decoder: nn.Module, updater: nn.Module):
        """Constructor for a GNN.

        Arguments:
            encoder (nn.Module): An appropriate Pytorch module encoding inputs into the latent graph space.
            processor (GN): A graph network applying message passing in the latent graph space.
            decoder (nn.Module): An appropriate Pytorch module decoding the processed graph into the output space.
            updater (nn.Module): A Pytorch module ???

        Returns:
            None
        """
        supet().__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder

    def forward():
        pass

class GN(nn.Module):
    """Graph Network, which performs message passing on a graph. Not to be confused with a Graph Neural Network (GNN)."""

class GraphLayer(pyg.nn.MessagePassing):
    """A message passing layer for a Graph Network."""

    def __init__(self, dim_node: int, dim_edge: int, layers: int) -> NoneType:
        """Constructor for a GraphLayer, which performs a single round of message passing.

        Inputs:
            dim_node (int): Dimension of nodes.
            dim_edge (int): Dimension edge weights.
            layers (int): Number of layers for the graph update MLPs.

        Outputs:
            None
        """
        self.edge_updater = MLP(2 * dim_node + dim_edge, hidden_size, layers #### FIX THISW
        self.node_updater = MLP(2 * dim_node * 2, hidden_size, layers) #### FIX THIS

    def message(self, node_a: int, node_b, edge_ab):
        """hmmm

        Arguments:

        Returns:
        """
        x = torch.cat((node_a, node_b, node_ab), dim=-1)
        return self.edge_updater(x)

    def aggregate(self, inputs, index):
        """Aggregate edge updates associated with a node.

        Arguments:
            inputs (???): ???
            index (???): ???

        Returns:
        """
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, reduce="sum")
        return (inputs, out)

    def forward():
        """hmmm

        Arguments:

        Returns:
        """
        return


class RectNN(nn.Module):
    """A sequential MLP with a fixed width for all hidden layers."""

    def __init__(self, input_dim: int, output_dim: int, width: int, depth: int, activation_type=nn.Tanh) -> NoneType:
        """Constructor for a RectNN.
    
        Arguments:
            input_dim (int): Dimension of input.
            output_dim (int): Dimension of output.
            width (int): Number of perceptrons per hidden layer.
            depth (int): Number of hidden layers.
            activation_type (class): Type of activation function to be instantiated for each layer.
    
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

    # DEFINE FORWARD, LOSS FUNCTIONS?