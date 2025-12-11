import gns_tools as gt
import traj_ai as ta

def main():
    ####################################
    # Part 0: Specify Model parameters #
    ####################################

    # Misc
    dt = 2 / 29.97 # Every other frame of 30 FPS video is provided
    num_past = 5 # Incorporate past n frames plus the present frame
    epsilon=1e-6 # For handling zero velocity singularity, at the cost of introducing small bias

    # State dimensions used to help compute space dimensions
    # See traj_ai module for state convention
    dim_pos = 2
    dim_vel = 2
    len_attr = 0 # Additional attributes passed in

    # Space dimensions
    dim_state = (1 + num_past) * (dim_pos + dim_vel) + len_attr # Dimension of state space
    dim_state_objective = num_past * dim_pos + 1 + num_past * dim_vel + len_attr # Dimension of objective state space, reduced by 3 DOF, since you lose absolute position (2 DOF) and orientation (1 DOF)
    dim_relation = (1 + num_past) * (dim_pos + dim_vel + 1) # Dimension of (objective) relation space, composed of relative positions, relative velocities, and distances
    dim_update = 2 # Either learning local 2D velocity or acceleration
    dim_node = 8 # Dimension of graph space
    dim_edge = 8 # Dimension of edge weights in graph space
    
    # Node encoder MLP parameters
    width_encoder_node = 16
    depth_encoder_node = 2
    activation_type_encoder_node = nn.ReLU
    
    # Edge encoder MLP parameters
    width_encoder_edge = 16
    depth_encoder_edge = 2
    activation_type_encoder_edge = nn.ReLU
    
    # Graph layer MLP parameters (within GraphNetwork processor)
    num_layers_gn = 3 # n rounds of message passing
    width_gn_layer_mlps = 16 # For both node and edge update MLPs
    depth_gn_layer_mlps = 2 # For both node and edge update MLPs
    activation_type_gn_layer_mlps = nn.ReLU # For both node and edge update MLPs
    
    # Decoder
    width_decoder = 16
    depth_decoder = 2
    activation_type_decoder = nn.ReLU

    #######################
    # Part 1: Build Model #
    #######################

    # Define relater: state space to relation space
    # Expresses how a influencer particle relates to an influenced particle
    relater = NormalTangentialDistanceObjectiveStateRelater(
        num_past=num_past,
        epsilon=epsilon
    )

    # Define node encoder: state space to graph space
    # Composed of objective state transcoder and MLP
    encoder_node = nn.Sequential(
        ta.NormalTangentialObjectiveStateTranscoder(
            num_past=num_past,
            epsilon=epsilon
        ),
        gt.RectNN(
            input_dim=dim_state_objective,
            output_dim=dim_node,
            width=width_encoder_node,
            depth=depth_encoder_node,
            activation_type=activation_type_encoder_node
        )
    )

    # Define edge encoder: relation space to edge space
    # Just an MLP, since the chosen relater is already objective
    encoder_edge = gt.RectNN(
        input_dim=dim_relation,
        output_dim=dim_edge,
        width=width_encoder_edge,
        depth=depth_encoder_edge,
        activation_type=activation_type_encoder_edge
    )

    # Define processor: graph space to graph space
    # The graph network composed of a sequence of graph layers transforming the graph
    processor = gt.GraphNetwork(
        dim_node=dim_node,
        dim_edge=dim_edge,
        num_layers=num_layers_gn,
        mlp_width=width_gn_layer_mlps,
        mlp_depth=depth_gn_layer_mlps,
        mlp_activation=activation_type_gn_layer_mlps
    )

    # Define decoder: graph space to update space
    # Maps to two dimensions: either a learned velocity or acceleration
    decoder = gt.RectNN(
        input_dim=dim_node,
        output_dim=dim_update,
        width=width_decoder,
        depth=depth_decoder,
        activation_type=activation_type_decoder
    )

    # Define graph neural network: state space to update space
    # Composed of relater, node encoder, edge encoder, processor, and decoder
    graph_neural_network = gt.GraphNeuralNetwork(
        relater=relater,
        encoder_node=encoder_node,
        encoder_edge=encoder_edge,
        processor=processor,
        decoder=decoder
    )

    # Define updater: state and update spaces to state space
    # Takes in state and local velocity or acceleration, and updates state accordingly
    updater = ta.LocalAccelerationUpdater(
        dt=dt, 
        num_past=num_past
    )

    # Define connector: determine which states influence each other
    # With none, the graph is assumed fully connected
    connector = None

    # Define graph neural simulator: state space to state space
    # Composed of graph neural network and updater
    simulator = gt.GraphNeuralSimulator(
        graph_neural_network=graph_neural_network,
        updater=updater,
        connector=connector
    )

    #######################
    # Part 2: Train Model #
    #######################

    # Need to define some things here

    losses_train, losses_one_step, losses_rollout = gt.train(
        simulator: gns.GraphNeuralSimulator,
        train_loader: torch.DataLoader,
        val_loader: torch.DataLoader,
        val_rollout: torch.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        num_epochs: int = 500,
        rollout_interval: int = 10,
        pr: int = 0,
        patience: int = 0,
        loss_threshold: float = 0
    )

    ######################
    # Part 3: Save Model #
    ######################