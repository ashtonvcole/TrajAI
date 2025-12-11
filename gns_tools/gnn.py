import .mlp
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torch_scatter



class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network with encoder, processor, and decoder."""
    
    def __init__(self, relater: nn.Module, encoder_node: nn.Module, encoder_edge: nn.Module, processor: nn.Module, decoder: nn.Module):
        """Constructor for a Graph Neural Network.

        Arguments:
            relater (nn.Module): An appropriate Pytorch module defining a directed relationship between two particles as a function of their states. This may or may not have symmetry or anti-symmetry, e.g., distance, displacement, etc.
            encoder_node (nn.Module): An appropriate Pytorch module encoding particle states into the latent graph space as graph nodes. The output dimension should be dim_node. 
            encoder_edge (nn.Module): An appropriate Pytorch module encoding particle relationships into multidimensional edge weights. The output dimension should be dim_edge.
            processor (nn.Module): An appropriate Pytorch module applying message passing in the latent graph space, e.g. a GraphNetwork object.
            decoder (nn.Module): An appropriate Pytorch module decoding the processed graph into the output space. The input dimension should be dim_node.

        Returns:
            None
        """
        super(GraphNeuralNetwork, self).__init__()
        self.relater = relater
        self.encoder_node = encoder_node
        self.encoder_edge = encoder_edge
        self.processor = processor
        self.decoder = decoder

    def forward(self, x: torch.Tensor, connectivity: torch.Tensor = None):
        """Apply the Graph Neural Network to a batch of data.
        
        Arguments:
            x (torch.Tensor): A batch of particle states composing the global state of the system, of dimension (num_particles, dim_particle_state).
            connectivity (torch.Tensor, optional): Which particles are considered to influence each other. This influence is one-way. The tensor has dimension (2, num_edges), where the first particle influences the second particle.

        Returns:
            torch.Tensor: The output of the Graph Neural Network.
        """
        # Get numer of particles from dimension
        num_particles = x.shape[0]
        # If connectivity is None, assume that the graph is fully connected
        if connectivity is None:
            # Build connectivity where every node is connected to every node
            # repeat_interleave: [1, 2, 3] -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
            nodes_source = torch.arange(num_particles, device=x.device).repeat_interleave(num_particles)
            # repeat: [1, 2, 3] -> [1, 2, 3, 1, 2, 3, 1, 2, 3]
            nodes_target = torch.arange(num_particles, device=x.device).repeat(num_particles)
            # Remove edges pointing to self, i.e. source = target
            mask = nodes_source != nodes_target
            # Combine these into one tensor
            connectivity = torch.stack((nodes_source[mask], nodes_target[mask]), dim=0)
        # Get particle-particle relationships
        # Use connectivity to reference source and target nodes
        r = self.relater(x[connectivity[0]], x[connectivity[1]])
        # Encode
        V = self.encoder_node(x)
        E = self.encoder_edge(r)
        # Construct data object (graph) from nodes, edges, and connectivity
        graph = pyg_data.Data(
            x=V,
            edge_index=connectivity,
            edge_attr=E
        )
        # Message passing
        graph = self.processor(graph)
        # Decode
        return self.decoder(graph.x)



class GraphNetwork(nn.Module):
    """Graph Network, which performs message passing on a graph. Not to be confused with a Graph Neural Network (GNN)."""

    def __init__(self, dim_node: int, dim_edge: int, num_layers: int, mlp_width: int, mlp_depth: int, mlp_activation = nn.ReLU) -> None:
        """Constructor for a GraphNetwork.

        Arguments:
            dim_node (int): Dimension of nodes.
            dim_edge (int): Dimension edge weights.
            num_layers (int): Number of message passing GraphLayers in the network.
            mlp_width (int): Number of perceptrons per layer for the graph update MLPs.
            mlp_depth (int): Number of layers for the graph update MLPs.
            mlp_activation (nn.Module subtype, optional): Activation function type for the graph update MLPs. Default is nn.ReLU.

        Returns:
            None
        """
        super(GraphNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers.append(GraphLayer(dim_node, dim_edge, mlp_width, mlp_depth, mlp_activation))

    def forward(self, graph: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Apply the Graph Network to a graph.

        Arguments:
            graph (torch_geometric.data.Data): The initial graph, with nodes, edges, and connectivity.

        Returns:
            torch_geometric.data.Data: The updated graph object.
        """
        for layer in self.layers:
            graph = layer(graph)
        return graph


        
class GraphLayer(pyg.nn.MessagePassing):
    """A message passing layer with mlp.RectNN MLPs for a Graph Network."""

    def __init__(self, dim_node: int, dim_edge: int, mlp_width: int, mlp_depth: int, mlp_activation = nn.ReLU) -> None:
        """Constructor for a GraphLayer, which performs a single round of message passing.
        Arguments:
            dim_node (int): Dimension of nodes.
            dim_edge (int): Dimension edge weights.
            mlp_width (int): Number of perceptrons per layer for the mlp.RectNN graph update MLPs.
            mlp_depth (int): Number of layers for the mlp.RectNN graph update MLPs.
            mlp_activation (nn.Module subtype, optional): Activation function type for the mlp.RectNN graph update MLPs. Default is nn.ReLU.

        Returns:
            None
        """
        super(GraphLayer, self).__init__()
        self.node_dim = 0 # Expect 2D tensor, as list of node vectors
        self.edge_updater = mlp.RectNN(
            2 * dim_node + dim_edge,
            dim_edge,
            mlp_width,
            mlp_depth,
            mlp_activation
        )
        self.node_updater = mlp.RectNN(
            dim_node + dim_edge,
            dim_node,
            mlp_width, 
            mlp_depth,
            mlp_activation
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

    def forward(self, graph: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Apply the graph layer to a graph.

        Given a graph, represented by nodes, edge weights, and a connectivity mapping, the layer produces new nodes and edge weights through a process called message passing. Messages are created by feeding an edge's weights and its associated nodes into a MLP. This message is added to the existing edge weights. Nodes are updated by aggregating messages directed at a node, and feeding them and the node through a second MLP.

        Arguments:
            graph (torch_geometric.data.Data): The initial graph, with nodes, edges, and connectivity.

        Returns:
            torch_geometric.data.Data: The updated graph object.
        """
        # Extract nodes, edges, and connectivity for operations
        nodes = graph.x
        edges = graph.edge_attr
        connectivity = graph.edge_index
        # Propagate calls message, aggregate, and update
        # delta_edge is the corresponding message.
        delta_edges, net_messages = self.propagate(connectivity, x=(nodes, nodes), edge_feature=edges)
        delta_nodes = self.node_updater(torch.cat((nodes, net_messages), dim=-1))
        # Update the same graph object
        graph.x = nodes + delta_nodes
        graph.edge_attr = edges + delta_edges
        return graph