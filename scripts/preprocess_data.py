import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace



def main():
    source_path = '../data/raw'
    num_past = 5
    folders = [p.name for p in Path(source_path).iterdir() if p.is_dir()]
    chunks = []

    # Iteratively read and open files
    for folder in folders:
        file_path = f'{source_path}/{folder}/combined_data.csv'
        print(f'Opening {file_path}')
        df = pd.read_csv(file_path)
        chunks += process_data(df, min_chunk_length=(num_past + 2))
        print()

    # Get normalization stats
    all_data = torch.cat([chunk.view(-1, chunk.shape[-1]) for chunk in chunks], dim=0)
    stats = {
        'mean': all_data.mean(dim=0),
        'std': all_data.std(dim=0)
    }
    stats['std'][stats['std'] < 1e-6] = 1.0
    
    print(f"Global Mean: {stats['mean']}")
    print(f"Global Std:  {stats['std']}")

    print("Normalizing chunks")
    normalized_chunks = []
    for chunk in chunks:
        # (chunk - mean) / std
        normalized_chunk = (chunk - stats['mean'].view(1, 1, -1)) / stats['std'].view(1, 1, -1)
        normalized_chunks.append(normalized_chunk)

    # Save list of tensors to file
    print('Saving to file')
    torch.save(chunks, '../data/processed/data.pt')
    torch.save(stats, '../data/processed/stats.pt')

    print('Complete')



def process_data(df: pd.DataFrame, min_chunk_length: int) -> torch.Tensor:
    # Process single time series, return it as a list of items to be added to list
    print('Processing time series')
    
    # Split states, pick out quantities of interest
    num_frames = len(df[df['track_id'] == 'track_1'])
    print(f'{num_frames} video frames identified')
    num_runners = int(len(df) / num_frames)
    print(f'{num_runners} runners identified')
    
    # Check data consistency
    if num_frames * num_runners != len(df):
        raise ValueError('Runners do not have the same number of frames')
    
    # Chunking: use masking to reduce trajectories
    # Create new data frame to hold which runners are visible at a given moment
    print('Splitting video into chunks')
    masks = df.pivot(index='frame', columns='track_id', values='mask')
    # runner_columns = [f'track_{i + 1}' for i in range(num_runners)] # Flaw: track_* skips an index
    runner_columns = [c for c in masks.columns if isinstance(c, str) and c.startswith('track_')]
    def sort_key(x): # Custom sorting for track_* strings... compare the integer
        try:
            return int(x.split('_')[1])
        except (IndexError, ValueError):
            return 0
    runner_columns.sort(key=sort_key)
    num_runners_found = len(runner_columns)
    if num_runners_found != num_runners: # Some sort of frame mismatch happened
        raise ValueError(f'Expected {num_runners} runners, found {num_runners_found} runners. Runners may not have the same number of frames.')
    masks['current_state'] = masks[runner_columns].apply(tuple, axis=1)
    # Compute chunks in masks
    masks['state_changed'] = masks['current_state'] != masks['current_state'].shift(1)
    masks['chunk_id'] = masks['state_changed'].cumsum()
    # Count number of runners
    masks['num_runners'] = masks[runner_columns].sum(axis=1)

    # Create list of trajectories
    print('Processing chunks')
    tensors = [] # List of Pytorch tensors
    # Iterate through each chunk
    min_runners = 2
    # Iterate through each chunk
    for chunk_id, chunk_data in masks.groupby('chunk_id'):
        print(f'chunk id: {chunk_id} -----------------------')
        len_chunk = len(chunk_data)
        print(f'length of chunk: {len_chunk}')
        # Exclude if too short
        if len_chunk < min_chunk_length:
            print('Too short, excluding')
            continue
        runners_state = chunk_data[runner_columns].iloc[0] # Which runners are in
        runners_present = [f'track_{j + 1}' for j in range(num_runners) if runners_state[f'track_{j + 1}'] == 1]
        print(f'present runners: {runners_present}')
        num_runners_chunk = chunk_data['num_runners'].iloc[0]
        print(f'number of runners: {num_runners_chunk}')
        # Exclude if no runners
        if num_runners_chunk < min_runners:
            print('Too short, excluding')
            continue
        frames_in_chunk = chunk_data.index
        print(f'associated video frames {frames_in_chunk}')
            
        # Create 3D tensor
        dim_state_reduced = 2 + 2 + 2 # (x, y, vx, vy, max(bw, bh), conf)
        states_chunk = torch.zeros((len_chunk, num_runners_chunk, dim_state_reduced), dtype=torch.float32)
        
        # Loop across runners to fill in data
        for i in range(num_runners_chunk):
            runner_data = df[(df['track_id'] == runners_present[i]) & df['frame'].isin(frames_in_chunk)]
            states_chunk[:, i, 0:2] = torch.from_numpy(runner_data[['wx', 'wy']].to_numpy()) # x, y
            states_chunk[:, i, 2:4] = torch.from_numpy(runner_data[['vx', 'vy']].to_numpy()) # vx, vy
            states_chunk[:, i, 4] = torch.max(torch.from_numpy(runner_data[['bw', 'bh']].to_numpy()), dim=-1)[0] # max(bw, bh)
            states_chunk[:, i, 5] = torch.from_numpy(runner_data['conf'].to_numpy()) # conf
        tensors.append(states_chunk)
    # Filter zero velocities or split video
    # Cut chunks which
    return tensors



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



if __name__ == '__main__':
    main()