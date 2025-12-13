import gns_tools as gt
import traj_ai as ta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

def main():
    print('Starting train.py')
    
    ####################################
    # Part 0: Specify Model parameters #
    ####################################

    print('Specifying model parameters...')

    # File paths
    PATH_CHUNKS = '../data/processed/normalized_data.pt'
    PATH_STATS = '../data/processed/stats.pt'
    PATH_MODEL = '../models/simulator.pt'

    # Misc
    dt = 2 / 29.97 # Every other frame of 30 FPS video is provided
    num_past = 5 # Incorporate past n frames plus the present frame
    epsilon=1e-6 # For handling zero velocity singularity, at the cost of introducing small bias

    # State dimensions used to help compute space dimensions
    # See traj_ai module for state convention
    dim_pos = 2
    dim_vel = 2
    len_attr = 2 # Additional attributes passed in

    # Space dimensions
    # dim_state_reduced = dim_pos + dim_vel + len_attr # Not used
    dim_state = (1 + num_past) * (dim_pos + dim_vel) + len_attr # Dimension of state space
    dim_state_objective = num_past * dim_pos + 1 + num_past * dim_vel + len_attr # Dimension of objective state space, reduced by 3 DOF, since you lose absolute position (2 DOF) and orientation (1 DOF)
    dim_relation = (1 + num_past) * (dim_pos + dim_vel + 1) # Dimension of (objective) relation space, composed of relative positions, relative velocities, and distances
    dim_update = 2 # Either learning local 2D velocity or acceleration
    dim_node = 32 # Dimension of graph space
    dim_edge = 32 # Dimension of edge weights in graph space
    
    # Node encoder MLP parameters
    width_encoder_node = 32
    depth_encoder_node = 2
    activation_type_encoder_node = nn.LeakyReLU
    
    # Edge encoder MLP parameters
    width_encoder_edge = 32
    depth_encoder_edge = 2
    activation_type_encoder_edge = nn.LeakyReLU
    
    # Graph layer MLP parameters (within GraphNetwork processor)
    num_layers_gn = 5 # n rounds of message passing
    width_gn_layer_mlps = 32 # For both node and edge update MLPs
    depth_gn_layer_mlps = 2 # For both node and edge update MLPs
    activation_type_gn_layer_mlps = nn.LeakyReLU # For both node and edge update MLPs
    
    # Decoder
    width_decoder = 32
    depth_decoder = 2
    activation_type_decoder = nn.LeakyReLU

    #######################
    # Part 1: Build Model #
    #######################
    
    print('Building GNS model...')

    # Define relater: state space to relation space
    # Expresses how a influencer particle relates to an influenced particle
    relater = ta.NormalTangentialDistanceObjectiveStateRelater(
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

    ########################
    # Part 2: Load in data #
    ########################
    
    print('Loading data...')

    # Read files
    chunks = torch.load(PATH_CHUNKS)
    stats = torch.load(PATH_STATS)

    # Train/validation split
    train_val_split = 0.8
    split_index = int(len(chunks) * train_val_split)
    chunks_train = chunks[:split_index]
    chunks_val = chunks[split_index:]

    # Build Datasets
    train_dataset = TrajectoryDataset(chunks_train)
    val_dataset = TrajectoryDataset(chunks_val)

    # Build DataLoaders
    def identity_collate(batch):
        return batch[0]
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True,
        num_workers=0,
        collate_fn=identity_collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=identity_collate
    )
    val_rollout = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=identity_collate
    )

    #######################
    # Part 3: Train Model #
    #######################

    print('Training model...')

    # Define parameters for training function
    state_composer = ta.StateComposer(num_past)
    state_decomposer = ta.StateDecomposer(num_past)
    criterion = CombinedLoss(0) # No past steps since train_reduced computes loss on reduced state vector of dimension dim_state_reduced
    optimizer = optim.Adam(
        simulator.parameters(),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.98
    )
    num_epochs = 100
    rollout_interval = 10 # Compute rollout loss every n epochs
    pr = 1 # Print every n epochs
    patience = 50 # Stop after this many epochs of no improvement
    loss_threshold=1e-6 # Stop after loss drops below this

    losses_train, losses_one_step, losses_rollout = gt.train_reduced(
        simulator=simulator,
        train_loader=train_loader,
        val_loader=val_loader,
        val_rollout=val_rollout,
        state_composer=state_composer,
        state_decomposer=state_decomposer,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        rollout_interval=rollout_interval,
        pr=pr,
        patience=patience,
        loss_threshold=loss_threshold
    )

    #################################
    # Part 4: Save Model and Losses #
    #################################

    print('Saving TrajAI model...')

    torch.save(simulator.state_dict(), 'simulator.pth')

    print('Process complete!')



class TrajectoryDataset(Dataset):
    """
    Batch data structure for GNS training.

    Attributes:
        chunks (list): A list of data.
    """

    def __init__(self, chunks: list) -> None:
        """Constructor for trajectory dataset.

        Arguments:
            trajectories (torch.Tensor): A list of tensors, each of dimension (num_frames, num_particles, dim_state).

        Returns:
            None
        """
        self.chunks = chunks
        
    def __len__(self):
        """Get length of the data set.

        Arguments:
            None

        Returns:
            int: The length of chunks.
        """
        return len(self.chunks)

    def __getitem__(self, idx):
        """Index the data set.

        Arguments:
            idx: A key provided in brackets.

        Returns:
            When traj is requested, the chunks.
        """
        return SimpleNamespace(traj=self.chunks[idx])



class CombinedLoss(nn.Module):
    def __init__(self, num_step: int) -> None:
        super(CombinedLoss, self).__init__()
        self.loss_pos = ta.StateStateMeanSquarePositionLoss(num_step)
        self.loss_val = ta.StateStateMeanSquareVelocityLoss(num_step)

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        return self.loss_pos(x_pred, x_true) + self.loss_val(x_pred, x_true)



if __name__ == '__main__':
    main()