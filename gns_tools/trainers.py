import .gns
import torch
import torch.nn as nn
import torch.optim as optim



def train(simulator: gns.GraphNeuralSimulator, train_loader: torch.DataLoader, val_loader: torch.DataLoader, val_rollout: torch.DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, num_epochs: int = 500, rollout_interval: int = 10, pr: int = 0, patience: int = 0, loss_threshold: float = 0) -> tuple:
    """Generic training function for a GraphNeuralSimulator.

    The rollout is computed using the full state. If the simulator input state includes past physical states, e.g., the past 5 positions and velocities, this may be memory-inefficient.

    Arguments:
        simulator (gns.GraphNeuralSimulator): A simulator to train.
        train_loader (torch.DataLoader): An appropriate Pytorch DataLoader object. For any batch, batch.x is a tensor of input states of dimension (len_batch, num_particles, dim_state). Meanwhile, batch.y is the corresponding target state, of the same dimension.
        val_loader (torch.DataLoader): An appropriate Pytorch DataLoader object, of the same structure as train_loader.
        val_rollout (torch.DataLoader): An appropriate Pytorch DataLoader object. For any batch, batch.x is a tensor of input states of dimension (len_batch, num_particles, dim_state).
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

    for i in range(num_epochs):
        # Set to training mode
        model.train()

        # Cumulative loss
        loss_train = 0.0

        # Train data in batches
        for batch in train_loader:
            # Make prediction of next states
            x = batch.x # Current state
            y = batch.y # Resultant state
            pred = simulator(x)

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
        model.eval()
        
        # One Step MSE: predicting one step into the future
        loss_one_step = 0.0
        for batch in val_loader:
            # Make prediction of next states
            x = batch.x # Current state
            y = batch.y # Resultant state
            pred = simulator(x)

            # Get loss based on criterion
            loss_one_step += criterion(pred, y).item()
        loss_one_step /= len(val_loader) # Normalize sum
        losses_one_step.append(loss_one_step) # Record
        
        # Rollout MSE: predicting several steps into the future
        if epoch % rollout_interval == 0:
            loss_rollout = 0.0
            for batch in val_rollout:
                # Make rollout prediction
                # Note that if the system requires several frames for predictions, these are assumed encoded in the first state
                # This operates on the full, not a reduced state
                # This may be inefficient
                traj = batch.x # Full system state trajectory, of dimension (num_frames, num_particles, dim_state)
                pred = gns.rollout(simulator, traj[0, :, :], traj.shape[0] - 1)
    
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
    model.eval()
    
    return loss_train, loss_one_step, loss_rollout