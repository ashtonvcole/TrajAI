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
        """Apply the Graph Neural Network to a batch of data.
        Arguments:
            ???

        Returns:
            ???
        """
        # Encode
        # Message passing
        # Decode
        pass



class GraphNetwork(nn.Module):
    """Graph Network, which performs message passing on a graph. Not to be confused with a Graph Neural Network (GNN)."""

    def __init__(self, dim_node: int, dim_edge: int, num_layers: int, mlp_width: int, mlp_depth: int) -> None:
        """Constructor for a GraphNetwork.

        Arguments:
            dim_node (int): Dimension of nodes.
            dim_edge (int): Dimension edge weights.
            num_layers (int): Number of message passing GraphLayers in the network.
            mlp_width (int): Number of perceptrons per layer for the graph update MLPs.
            mlp_depth (int): Number of layers for the graph update MLPs.

        Returns:
            None
        """
        super(GraphNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers.append(GraphLayer(dim_node, dim_edge, mlp_width, mlp_depth))

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, connectivity: torch.Tensor) -> tuple:
        """Apply the Graph Network to a batch of data.

        Arguments:
            nodes (torch.Tensor): The initial nodes, with dimension (num_nodes, dim_node).
            edges (torch.Tensor): The initial edges, with dimension (num_edges, dim_edge).
            connectivity (torch.Tensor): A mapping from directed edges to nodes, with dimension (2, num_edges). For example, connectivity[1, 5] holds the second node index associated with edge 5.

        Returns:
            (nodes_out, edges_out) (tuple)
            nodes_out (torch.Tensor): The updated nodes, with dimension (num_nodes, dim_node).
            edges_out (torch.Tensor): The updated edges, with dimension (num_edges, dim_edge).
        """
        nodes_out = nodes
        edges_out = edges
        for layer in self.layers:
            nodes_out, edges_out = layer(nodes_out, edges_out, connectivity)
        return nodes_out, edges_out


        
class GraphLayer(pyg.nn.MessagePassing):
    """A message passing layer for a Graph Network."""

    def __init__(self, dim_node: int, dim_edge: int, mlp_width: int, mlp_depth: int) -> None:
        """Constructor for a GraphLayer, which performs a single round of message passing.

        Arguments:
            dim_node (int): Dimension of nodes.
            dim_edge (int): Dimension edge weights.
            mlp_width (int): Number of perceptrons per layer for the graph update MLPs.
            mlp_depth (int): Number of layers for the graph update MLPs.

        Returns:
            None
        """
        super(GraphLayer, self).__init__()
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

    def message(self, node_a: torch.Tensor, node_b: torch.Tensor, edge_ab: torch.Tensor) -> torch.Tensor:
        """Get a message associated with an edge.

        Arguments:
            node_a (torch.Tensor): The first node associated with the directed edge. This may be either a single input of dimension (dim_node), or a batch of dimension (n, dim_node).
            node_b (torch.Tensor): The second node associated with the directed edge. This may be either a single input of dimension (dim_node), or a batch of dimension (n, dim_node).
            edge_ab (torch.Tensor): The directed edge. This may be either a single input of dimension (dim_edge), or a batch of dimension (n, dim_edge).

        Returns:
            torch.Tensor: The message, either of dimension (edge_dim) or (n, edge_dim).
        """
        return self.edge_updater(torch.cat((node_a, node_b, edge_ab), dim=-1))

    def aggregate(self, messages: torch.Tensor, indices: torch.Tensor) -> tuple:
        """Aggregate edge updates associated with a node.

        Arguments:
            messages (torch.Tensor): The output tensor from message(). This is the collection of all messages, of dimension (num_edges, dim_edge).
            indices (torch.Tensor): The receiver nodes for each message, of dimension (num_edges). This can be derived from connectivity[1, :].

        Returns:
            (messages, net_messages) (tuple)
            messages (torch.Tensor): The same messages passed in to the function.
            net_messages (torch.Tensor): The aggregated messages per node, of dimension (num_nodes, dim_edge).
        """
        # Scatter aggregates all of the messages and directs them to the correct node
        # These sums are placed in the net_messages tensor of length num_nodes
        # The receiver node index in index corresponds to the index in net_messages where the message is summed
        net_messages = torch_scatter.scatter(messages, indices, dim=self.node_dim, reduce="sum")
        return (messages, net_messages)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, connectivity: torch.Tensor) -> tuple:
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
        # Propagate calls message, aggregate, and update
        # delta_edge is the corresponding message.
        delta_edges, net_messages = self.propagate(connectivity, x=(nodes, nodes), edge_feature=edges)
        delta_nodes = self.node_updater(torch.cat((nodes, net_messages), dim=-1))
        edges_out = edges + delta_edges
        nodes_out = nodes + delta_nodes
        return nodes_out, edges_out