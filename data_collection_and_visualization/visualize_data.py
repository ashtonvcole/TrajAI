# plots position and velocity from data
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
import glob

# --- Configuration ---
INPUT_ROOT = 'output_data/GNN_READY'
OUTPUT_ROOT = 'output_data/VISUALIZATIONS'

# Animation Settings
ANIMATION_FPS = 15
REAL_WORLD_FPS = 30
TRAIL_SECONDS = 5
TRAIL_LENGTH_FRAMES = REAL_WORLD_FPS * TRAIL_SECONDS

# Set plot aesthetics
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f0f0f0"})

def create_static_plots(df, video_name, save_dir):
    """
    Generates a 2x2 grid comparing World Space vs Metric Space.
    Top: Position (wx, wy) vs (wx_m, wy_m)
    Bottom: Velocity (World Speed vs Metric Speed)
    """
    
    # Calculate Speeds
    # World Speed (using vx, vy)
    df['speed_world'] = np.sqrt(df['vx']**2 + df['vy']**2)
    # Metric Speed (using vx_m, vy_m)
    df['speed_metric'] = np.sqrt(df['vx_m']**2 + df['vy_m']**2)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Get ID colors
    track_ids = df['track_id'].unique()
    palette = sns.color_palette("husl", len(track_ids))
    color_map = dict(zip(track_ids, palette))
    
    # Common plot settings
    scatter_kws = {'hue': 'track_id', 'style': 'athlete_bool', 
                   'markers': {1: 'o', 0: 'X'}, 's': 50, 'palette': color_map, 
                   'linewidth': 0}
    
    line_kws = {'hue': 'track_id', 'style': 'athlete_bool', 
                'dashes': False, 'palette': color_map, 'marker': 'o', 
                'markersize': 3, 'linewidth': 1.5}

    # --- 1. Top Left: World Position (wx, wy) ---
    sns.scatterplot(data=df, x='wx', y='wy', ax=axes[0, 0], **scatter_kws)
    axes[0, 0].set_title('Position: World Space (wx, wy)')
    axes[0, 0].axis('equal')
    axes[0, 0].legend_.remove() 

    # --- 2. Top Right: Metric Position (wx_m, wy_m) ---
    sns.scatterplot(data=df, x='wx_m', y='wy_m', ax=axes[0, 1], **scatter_kws)
    axes[0, 1].set_title('Position: Metric Space (wx_m, wy_m)')
    axes[0, 1].axis('equal')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- 3. Bottom Left: World Velocity ---
    sns.lineplot(data=df, x='frame', y='speed_world', ax=axes[1, 0], **line_kws)
    axes[1, 0].set_title('Velocity Magnitude: World Space')
    axes[1, 0].set_ylabel('Speed (units/frame)')
    axes[1, 0].legend_.remove()

    # --- 4. Bottom Right: Metric Velocity ---
    sns.lineplot(data=df, x='frame', y='speed_metric', ax=axes[1, 1], **line_kws)
    axes[1, 1].set_title('Velocity Magnitude: Metric Space')
    axes[1, 1].set_ylabel('Speed (m/s)')
    axes[1, 1].legend_.remove()

    plt.suptitle(f'Tracking Analysis (World vs Metric) - {video_name}', fontsize=20)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'analysis_plots.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    -> Saved 2x2 analysis plots (wx/wy vs wx_m/wy_m)")


def animate_world_space(df, video_name, save_dir):
    """
    Generates an animation with trails using World Coordinates (wx, wy).
    """
    x_col, y_col = 'wx', 'wy'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f'Tracking Animation (World Space: wx, wy) - {video_name}')
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')

    # Set limits based on data range + 5% padding
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_aspect('equal')

    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Setup IDs and Colors
    track_ids = df['track_id'].unique()
    palette = sns.color_palette("husl", len(track_ids))
    color_map = dict(zip(track_ids, palette))

    plot_elements = {}
    for track_id in track_ids:
        color = color_map[track_id]
        
        is_athlete = df[df['track_id'] == track_id]['athlete_bool'].iloc[0]
        marker_style = 'o' if is_athlete == 1 else 'X'

        # Trail
        trail_line, = ax.plot([], [], color=color, linewidth=2, alpha=0.6)
        # Head
        head_marker, = ax.plot([], [], marker=marker_style, color=color, markersize=10, markeredgecolor='white', linestyle='None')
        
        plot_elements[track_id] = {'trail': trail_line, 'head': head_marker}

    sorted_frames = sorted(df['frame'].unique())

    def update(frame_idx):
        current_raw_frame = sorted_frames[frame_idx]
        frame_text.set_text(f'Frame: {current_raw_frame}')
        
        trail_start_frame = current_raw_frame - TRAIL_LENGTH_FRAMES
        artists_to_update = [frame_text]

        for track_id in track_ids:
            # Filter for this track, valid data only
            track_df = df[(df['track_id'] == track_id) & (df['mask'] == 1)]

            # 1. Update Head
            current_pos = track_df[track_df['frame'] == current_raw_frame]
            if not current_pos.empty:
                plot_elements[track_id]['head'].set_data(current_pos[x_col].values, current_pos[y_col].values)
            else:
                plot_elements[track_id]['head'].set_data([], [])

            # 2. Update Trail
            trail_data = track_df[(track_df['frame'] > trail_start_frame) & (track_df['frame'] <= current_raw_frame)]
            if not trail_data.empty:
                plot_elements[track_id]['trail'].set_data(trail_data[x_col].values, trail_data[y_col].values)
            else:
                 plot_elements[track_id]['trail'].set_data([], [])
            
            artists_to_update.extend([plot_elements[track_id]['head'], plot_elements[track_id]['trail']])
            
        return artists_to_update

    # Create Animation
    ani = animation.FuncAnimation(fig, update, frames=len(sorted_frames), interval=50, blit=True)

    save_path = os.path.join(save_dir, 'animation_world_space.gif')
    print(f"    -> Generating world space animation...")
    
    writer = animation.PillowWriter(fps=ANIMATION_FPS)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"    -> Saved animation to {save_path}")


def process_visualizations():
    csv_files = glob.glob(os.path.join(INPUT_ROOT, '*', 'combined_data.csv'))
    
    if not csv_files:
        print(f"No combined_data.csv files found in {INPUT_ROOT}")
        return

    for csv_path in csv_files:
        video_dir = os.path.dirname(csv_path)
        video_name = os.path.basename(video_dir)
        
        print(f"Visualizing {video_name}...")
        save_dir = os.path.join(OUTPUT_ROOT, video_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Load Data
        df = pd.read_csv(csv_path)
        
        # Filter for STATIC plots
        valid_df = df[df['mask'] == 1].copy()

        if valid_df.empty:
            print(f"  [Warning] No valid data found in {video_name}. Skipping.")
            continue

        # 1. Generate Static Grid Plots (Wx/Wy and Metric)
        create_static_plots(valid_df, video_name, save_dir)

        # 2. Generate Animation (World Space: wx, wy)
        animate_world_space(df, video_name, save_dir)

if __name__ == "__main__":
    process_visualizations()