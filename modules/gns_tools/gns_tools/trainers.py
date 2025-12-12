import .gns
import torch
import torch.nn as nn
import torch.optim as optim



def train(simulator: gns.GraphNeuralSimulator, train_loader: torch.DataLoader, val_loader: torch.DataLoader, val_rollout: torch.DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, num_epochs: int = 500, rollout_interval: int = 10, pr: int = 0, patience: int = 0, loss_threshold: float = 0) -> tuple:
    """Generic training function for a GraphNeuralSimulator.

    The rollout is computed using the full state. If the simulator input state includes past physical states, e.g., the past 5 positions and velocities, this may be memory-inefficient.

    Arguments:
        simulator (gns.GraphNeuralSimulator): A simulator to train.
        train_loader (torch.DataLoader): An appropriate Pytorch DataLoader object. For any batch, which corresponds to a contiguous time series, batch.traj is a tensor of states of dimension (len_trajectory, num_particles, dim_state).
        val_loader (torch.DataLoader): An appropriate Pytorch DataLoader object, of the same structure as train_loader.
        val_rollout (torch.DataLoader): An appropriate Pytorch DataLoader object, of the same structure as train_loader.
        criterion (nn.Module): An appropriate loss function to minimize during training. This criterion should operate on states (state_pred, state_target), not state_reduced or the update (internal gnn.GraphNeuralNetwork output).
        optimizer (optim.Optimizer): An appropriate Pytorch optimizer, e.g., Adam. Must not require a closure function, like L-BFGS does.
        scheduler (optim.lr_scheduler._LRScheduler): An appropriate Pytorch learning rate scheduler.
        num_epochs (int, optional): How many rounds of optimization to conduct. Default is 500.
        rollout_interval (int, optional): How often to compute the rollout loss, which involves computing loss across an entire global state trajectory. Default is 10.
        pr (int, optional): How often to print training status, in epochs, with 0 meaning no printing. Default is 0.
        patience (int, optional): Whether to stop early, once loss has not improved for a certain interval, with 0 meaning no early stopping. Default is 0.
        loss_threshold (float, optional): Whether to stop early, once loss has dropped below a certain threshold, with 0 meaing no threshold. Default is 0.

    Returns:
        (losses_train, losses_one_step, losses_rollout) (tuple)
        losses_train (list): A list of the training losses
    """
    # Define losses
    losses_train = []
    losses_one_step = []
    losses_rollout = []

    # Used for early stopping
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Set to training mode
        simulator.train()

        # Cumulative loss
        loss_train = 0.0

        # Train data in batches
        for time_series in train_loader:
            # Make prediction of next states
            traj = time_series.traj # Extract series, of dimension (num_frames, num_particles, dim_state)
            x = traj[:-1, :, :] # Input set of global states, of dimension (num_frames - 1, num_particles, dim_state)
            y = traj[1:, :, :] # Resultant trajectory of global states, of dimension (num_frames - 1, num_particles, dim_state)
            pred = torch.zeros_like(y) # Hold all predictions for this time series

            # Iterate through frames
            for i in range(x.shape[0]):
                global_state = x[i, :, :] # Current global state, of dimension (num_particles, dim_state)
                pred[i, :, :] = simulator(global_state) # Prediction, of dimension (num_particles, dim_state)

            # Get loss based on criterion
            loss_batch = criterion(pred, y)
            loss_train += loss_batch.item()
            
            # Adjust weights to minimize loss
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            scheduler.step()
        loss_train /= len(train_loader) # Normalize sum
        losses_train.append(loss_train) # Record

        # Set to evaluation mode
        simulator.eval()
        
        # One Step MSE: predicting one step into the future
        loss_one_step = 0.0
        for time_series in val_loader:
            # Make prediction of next states
            traj = time_series.traj # Extract series, of dimension (num_frames, num_particles, dim_state)
            x = traj[:-1, :, :] # Input set of global states, of dimension (num_frames - 1, num_particles, dim_state)
            y = traj[1:, :, :] # Resultant trajectory of global states, of dimension (num_frames - 1, num_particles, dim_state)
            pred = torch.zeros_like(y) # Hold all predictions for this time series

            # Iterate through frames
            for i in range(x.shape[0]):
                global_state = x[i, :, :] # Current global state, of dimension (num_particles, dim_state)
                pred[i, :, :] = simulator(global_state) # Prediction, of dimension (num_particles, dim_state)

            # Get loss based on criterion
            loss_one_step += criterion(pred, y).item()
        loss_one_step /= len(val_loader) # Normalize sum
        losses_one_step.append(loss_one_step) # Record
        
        # Rollout MSE: predicting several steps into the future
        if epoch % rollout_interval == 0:
            loss_rollout = 0.0
            for time_series in val_rollout:
                # Make rollout prediction
                # Note that if the system requires several frames for predictions, these are assumed encoded in the first state
                # This operates on the full, not a reduced state
                # This may be inefficient
                traj = time_series.traj # Extract series, of dimension (num_frames, num_particles, dim_state)
                pred = gns.rollout(simulator, traj[0, :, :].unsqueeze(0), traj.shape[0] - 1) # Of dimension (num_frames, num_particles, dim_state)
    
                # Get loss based on criterion
                loss_rollout += criterion(pred, traj).item()
            loss_rollout /= len(val_rollout) # Normalize sum
            losses_rollout.append(loss_rollout) # Record
        
        # Early stopping
        if loss_train < best_loss:
            best_loss = loss_train # Update best loss
            patience_counter = 0 # Reset patience
        else:
            patience_counter += 1 # 
            
        if patience_counter > patience:
            print('Stopping due to patience counter being met.')
            break
        elif loss_train < loss_threshold:
            print('Stopping due to minimum loss threshold being met.')
            break
        
        # Print progress if desired
        if pr != 0 and (epoch + 1) % pr == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_train:.6f}, Patience: {patience_counter}')

    # Set to evaluation mode
    simulator.eval()
    
    return losses_train, losses_one_step, losses_rollout



def train_reduced(simulator: gns.GraphNeuralSimulator, train_loader: torch.DataLoader, val_loader: torch.DataLoader, val_rollout: torch.DataLoader, state_composer: nn.Module, state_decomposer: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, num_epochs: int = 500, rollout_interval: int = 10, pr: int = 0, patience: int = 0, loss_threshold: float = 0) -> tuple:
    """Generic training function for a GraphNeuralSimulator.

    Arguments:
        simulator (gns.GraphNeuralSimulator): A simulator to train.
        train_loader (torch.DataLoader): An appropriate Pytorch DataLoader object. For any batch, which corresponds to a contiguous time series, batch.traj is a tensor of states of dimension (len_trajectory, num_particles, dim_state_reduced).
        val_loader (torch.DataLoader): An appropriate Pytorch DataLoader object, of the same structure as train_loader.
        val_rollout (torch.DataLoader): An appropriate Pytorch DataLoader object, of the same structure as train_loader.
        state_composer (nn.Module): A function which converts a window of reduced states to a full state of dimension dim_state. Must have an attribute num_past (int) which holds the number of reduced states, besides the present, which are used to compose the full state.
        state_decomposer (nn.Module): A function which converts a batch of full states of dimension (num_particles, dim_state) to a single reduced state, of dimension (num_particles, dim_state_reduced).
        criterion (nn.Module): An appropriate loss function to minimize during training. This criterion should operate on states (state_reduced_pred, state_reduced_target), not the full state or the update (internal gnn.GraphNeuralNetwork output).
        optimizer (optim.Optimizer): An appropriate Pytorch optimizer, e.g., Adam. Must not require a closure function, like L-BFGS does.
        scheduler (optim.lr_scheduler._LRScheduler): An appropriate Pytorch learning rate scheduler.
        num_epochs (int, optional): How many rounds of optimization to conduct. Default is 500.
        rollout_interval (int, optional): How often to compute the rollout loss, which involves computing loss across an entire global state trajectory. Default is 10.
        pr (int, optional): How often to print training status, in epochs, with 0 meaning no printing. Default is 0.
        patience (int, optional): Whether to stop early, once loss has not improved for a certain interval, with 0 meaning no early stopping. Default is 0.
        loss_threshold (float, optional): Whether to stop early, once loss has dropped below a certain threshold, with 0 meaing no threshold. Default is 0.

    Returns:
        (losses_train, losses_one_step, losses_rollout) (tuple)
        losses_train (list): A list of the training losses
    """
    # Define losses
    losses_train = []
    losses_one_step = []
    losses_rollout = []

    # Used for early stopping
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Set to training mode
        simulator.train()

        # Cumulative loss
        loss_train = 0.0

        # Train data in batches
        for time_series in train_loader:
            # Make prediction of next states
            traj = time_series.traj # Extract series, of dimension (num_frames, num_particles, dim_state_reduced)
            x = state_composer(traj[:-1, :, :]) # Input set of global states, of dimension (num_frames - num_past - 1, num_particles, dim_state)
            y = traj[(state_composer.num_past + 1):, :, :] # Resultant trajectory of global states, of dimension (num_frames - num_past - 1, num_particles, dim_state_reduced)
            pred = torch.zeros_like(y) # Hold all predictions for this time series

            # Iterate through frames
            for i in range(x.shape[0]):
                global_state = x[i, :, :] # Current global state, of dimension (num_particles, dim_state)
                pred[i, :, :] = state_decomposer(simulator(global_state)) # Prediction, of dimension (num_particles, dim_state_reduced)

            # Get loss based on criterion
            loss_batch = criterion(pred, y) # Save for gradient descent
            loss_train += loss_batch.item()
            
            # Adjust weights to minimize loss
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            scheduler.step()
        loss_train /= len(train_loader) # Normalize sum
        losses_train.append(loss_train) # Record

        # Set to evaluation mode
        simulator.eval()
        
        # One Step MSE: predicting one step into the future
        loss_one_step = 0.0
        for time_series in val_loader:
            # Make prediction of next states
            traj = time_series.traj # Extract series, of dimension (num_frames, num_particles, dim_state_reduced)
            x = state_composer(traj[:-1, :, :]) # Input set of global states, of dimension (num_frames - num_past - 1, num_particles, dim_state)
            y = traj[(state_composer.num_past + 1):, :, :] # Resultant trajectory of global states, of dimension (num_frames - num_past - 1, num_particles, dim_state_reduced)
            pred = torch.zeros_like(y) # Hold all predictions for this time series

            # Iterate through frames
            for i in range(x.shape[0]):
                global_state = x[i, :, :] # Current global state, of dimension (num_particles, dim_state)
                pred[i, :, :] = state_decomposer(simulator(global_state)) # Prediction, of dimension (num_particles, dim_state_reduced)

            # Get loss based on criterion
            loss_one_step += criterion(pred, y).item()
        loss_one_step /= len(val_loader) # Normalize sum
        losses_one_step.append(loss_one_step) # Record
        
        # Rollout MSE: predicting several steps into the future
        if epoch % rollout_interval == 0:
            loss_rollout = 0.0
            for time_series in val_rollout:
                # Make rollout prediction
                # This operates on the reduced state
                window = state_composer.num_past + 1 # How many reduced states compose a full state
                traj = time_series.traj # Extract series, of dimension (num_frames, num_particles, dim_state_reduced)
                pred = gns.rollout_reduced(simulator, traj[0:window, :, :], traj.shape[0] - window, state_composer, state_decomposer) # Of dimension (num_frames, num_particles, dim_state_reduced)
    
                # Get loss based on criterion
                loss_rollout += criterion(pred, traj).item()
            loss_rollout /= len(val_rollout) # Normalize sum
            losses_rollout.append(loss_rollout) # Record
        
        # Early stopping
        if loss_train < best_loss:
            best_loss = loss_train # Update best loss
            patience_counter = 0 # Reset patience
        else:
            patience_counter += 1 # 
            
        if patience_counter > patience:
            print('Stopping due to patience counter being met.')
            break
        elif loss_train < loss_threshold:
            print('Stopping due to minimum loss threshold being met.')
            break
        
        # Print progress if desired
        if pr != 0 and (epoch + 1) % pr == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_train:.6f}, Patience: {patience_counter}')

    # Set to evaluation mode
    simulator.eval()
    
    return losses_train, losses_one_step, losses_rollout