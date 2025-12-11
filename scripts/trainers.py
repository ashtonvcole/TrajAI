import gnstools as gt
import torch
import torch.nn as nn



def train(simulator: gt.GraphNeuralSimulator, train_loader: torch.DataLoader, val_loader: torch.DataLoader, val_rollout: torch.DataLoader, criterion: nn.Module, optimizer: ???, scheduler: ???, num_epochs: int = 1000, pr: int = 0, patience: int = 500, loss_threshold: bool = 0) -> tuple:
    """Generic training function for a GraphNeuralSimulator.

    Arguments:
        simulator (gt.GraphNeuralSimulator): A simulator to train.
        train_loader (torch.DataLoader): ???
        val_loader (torch.DataLoader): ???
        val_rollout (torch.DataLoader): ???
        criterion (nn.Module): An appropriate loss function to minimize during training.
        optimizer (???): ???
        scheduler (???): ???
        num_epochs (int, optional): How many rounds of optimization to conduct. Default is 1000.
        pr (int, optional): How often to print training status, in epochs, with 0 meaning no printing. Default is 0.
        patience (int, optional): Whether to stop early, once loss has not improved for a certain interval, with 0 meaning no early stopping. Default is 0.
        loss_threshold (float, optional): Whether to stop early, once loss has dropped below a certain threshold, with 0 meaing no threshold. Default is 0.

    Returns:
        losses_train, losses_val, losses_one_step, losses_rollout (tuple)
    """
    # Define losses
    losses_train = []
    losses_val = []
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

        # Set to evaluation mode
        model.eval()
        
        # One Step MSE: predicting one step into the future
        # hmm
        
        # Rollout MSE: predicting several steps into the future
        # Not sure how to implement

        # Record losses
        losses_train.append(loss_train)
        
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
    
    return loss_train, loss_val, loss_one_step, loss_rollout