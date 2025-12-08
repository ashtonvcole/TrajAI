# Master GNN class
# Has encoder (sequential), processor (gn), decoder (sequential)
# Apply inductive bias appropriately? or in notebook?
# GN processor class too
# Update which does message passing M times
# Loss functions
# Approximators for different inductive biases



import mlp
import torch
import torch_geometric as pyg
import torch.nn as nn

class GNN(nn.Module):
    """Graph Neural Network with encoder, processor, and decoder."""
    
    def __init__(self, encoder: nn.Module, processor: GN, decoder: nn.Module):
        """Constructor for a GNN.

        Arguments:
            encoder (nn.Module): An appropriate Pytorch module encoding inputs into the latent graph space.
            processor (GN): A graph network applying message passing in the latent graph space.
            decoder (nn.Module): An appropriate Pytorch module decoding the processed graph into the output space.

        Returns:
            None
        """
        supet().__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder

    def forward():
        # Encode
        # Message passing
        # Decode
        pass

class GN(nn.Module):
    """Graph Network, which performs message passing on a graph. Not to be confused with a Graph Neural Network (GNN)."""

    def __init__(self):
        return 0

class GraphLayer(pyg.nn.MessagePassing):
    """A message passing layer for a Graph Network."""

    def __init__(self, dim_node: int, dim_edge: int, mlp_width: int, mlp_depth) -> NoneType:
        """Constructor for a GraphLayer, which performs a single round of message passing.

        Arguments:
            dim_node (int): Dimension of nodes.
            dim_edge (int): Dimension edge weights.
            mlp_width (int): Number of perceptrons per layer for the graph update MLPs.
            mlp_depth (int): Number of layers for the graph update MLPs.

        Returns:
            None
        """
        self.edge_updater = mlp.RectNN(
            2 * dim_node + dim_edge,
            dim_edge,
            mlp_width,
            mlp_depth
        )
        self.node_updater = mlp.RectNN(
            dim_node + dim_edge,
            dim_node,
            mlp_width, 
            mlp_depth
        )

    def message(self, node_a: int, node_b, edge_ab):
        """Get a message associated with an edge.

        Arguments:
            node_a (torch.Tensor): The first node associated with the directed edge. This may be either a single input of dimension (node_dim), or a batch of dimension (n, node_dim).
            node_b (torch.Tensor): The second node associated with the directed edge. This may be either a single input of dimension (node_dim), or a batch of dimension (n, node_dim).
            edge_ab (torch.Tensor): The directed edge. This may be either a single input of dimension (edge_dim), or a batch of dimension (n, edge_dim).

        Returns:
            torch.Tensor: The message, either of dimension (edge_dim) or (n, edge_dim).
        """
        x = torch.cat((node_a, node_b, node_ab), dim=-1)
        return self.edge_updater(x)

    def aggregate(self, inputs: torch.Tensor, index: ???) -> tuple:
        """Aggregate edge updates associated with a node.

        Arguments:
            inputs (???): ???
            index (???): ???

        Returns:
            (inputs, out) (tuple)
            inputs (???): ???
            out (???): ???
        """
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, reduce="sum")
        return (inputs, out)

    def forward(nodes: torch.Tensor, edges: torch.Tensor, connectivity: torch.Tensor) -> tuple:
        """Apply the graph layer to a graph.

        Given a graph, represented by nodes, edge weights, and a connectivity mapping, the layer produces new nodes and edge weights through a process called message passing. Messages are created by feeding an edge's weights and its associated nodes into a MLP. This message is added to the existing edge weights. Nodes are updated by aggregating messages directed at a node, and feeding them and the node through a second MLP.

        Arguments:
            nodes (torch.Tensor): The initial nodes, with dimension (num_nodes, dim_node).
            edges (torch.Tensor): The initial edges, with dimension (num_edges, dim_edge).
            connectivity (torch.Tensor): A mapping from directed edges to nodes, with dimension (2, num_edges). For example, connectivity[1, 5] holds the second node index associated with edge 5.

        Returns:
            (nodes_out, edges_out) (tuple)
            nodes_out (torch.Tensor): The updated nodes, with dimension (num_nodes, dim_node).
            edges_out (torch.Tensor): The updated edges, with dimension (num_edges, dim_edge).
        """
        delta_edges, net_messages = self.propagate(connectivity, x=(nodes, nodes), edge_feature=edges)
        delta_nodes = self.node_updater(torch.cat((nodes, net_messages), dim=-1))
        edges_out = edges + delta_edges
        nodes_out = nodes + delta_nodes
        return nodes_out, edges_out