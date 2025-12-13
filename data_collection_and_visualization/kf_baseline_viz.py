# PLOT KF vs GT
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import seaborn as sns

# Config
INPUT_ROOT = 'output_data/GNN_READY'
OUTPUT_ROOT = 'output_data/BASELINES'

# Time Settings
FPS = 30.0            
DATA_STEP = 2.0       
DT = DATA_STEP / FPS  

# Plot padding
HORIZON_FRAMES = 10   
KF_STEPS_AHEAD = int(HORIZON_FRAMES / DATA_STEP) 

# Camera Padding
PAD_PERCENT = 0.05     

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f0f0f0"})

class SimpleKalmanFilter:
    def __init__(self, dt=DT):
        self.dt = dt
        self.x = np.zeros(4) 
        self.F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.P = np.eye(4) * 10.0   
        self.R = np.eye(2) * 1.0    
        self.Q = np.eye(4) * 0.1    

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def predict_multi_step(self, steps):
        F_multi = np.linalg.matrix_power(self.F, steps)
        return F_multi @ self.x

    def update(self, z):
        y = z - self.H @ self.x             
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) 
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

def run_kf_on_video(df, video_name):
    df['kf_wx'] = np.nan
    df['kf_wy'] = np.nan
    df['kf_error'] = np.nan
    rmse_list = []

    for track_id in df['track_id'].unique():
        track_mask = (df['track_id'] == track_id) & (df['mask'] == 1)
        track_df = df[track_mask].sort_values('frame')
        if len(track_df) < 20: continue 

        coords = track_df[['wx', 'wy']].values
        frames = track_df['frame'].values

        kf = SimpleKalmanFilter(dt=DT)
        kf.x[:2] = coords[0]
        if len(coords) > 1:
            dx = coords[1][0] - coords[0][0]
            dy = coords[1][1] - coords[0][1]
            steps = (frames[1] - frames[0]) / DATA_STEP
            if steps > 0:
                kf.x[2] = dx / steps 
                kf.x[3] = dy / steps

        predictions = {} 
        for i in range(1, len(coords)):
            current_frame = frames[i]
            meas = coords[i]
            future_state = kf.predict_multi_step(steps=KF_STEPS_AHEAD) 
            target_frame = current_frame + HORIZON_FRAMES
            predictions[target_frame] = future_state[:2]
            kf.predict()
            kf.update(meas)

        for f, (px, py) in predictions.items():
            mask = (df['track_id'] == track_id) & (df['frame'] == f) & (df['mask'] == 1)
            if mask.any():
                df.loc[mask, 'kf_wx'] = px
                df.loc[mask, 'kf_wy'] = py
                gt_x = df.loc[mask, 'wx'].values[0]
                gt_y = df.loc[mask, 'wy'].values[0]
                dist = np.sqrt((gt_x - px)**2 + (gt_y - py)**2)
                df.loc[mask, 'kf_error'] = dist
                rmse_list.append(dist)

    rmse = np.sqrt(np.mean(np.array(rmse_list)**2)) if rmse_list else 0
    return df, rmse

def visualize_comparison(df, video_name, rmse):
    save_dir = os.path.join(OUTPUT_ROOT, video_name)
    os.makedirs(save_dir, exist_ok=True)
    
    valid_df = df[(df['mask'] == 1) & df['kf_wx'].notna()].copy()
    if valid_df.empty: return

    track_ids = valid_df['track_id'].unique()
    colors = sns.color_palette("husl", len(track_ids))
    
    # --- 1. Static Plot ---
    plt.figure(figsize=(12, 10))
    for i, tid in enumerate(track_ids):
        t_data = valid_df[valid_df['track_id'] == tid].sort_values('frame')
        if t_data.empty: continue
        c = colors[i]
        plt.plot(t_data['wx'], t_data['wy'], color=c, label=f"{tid}", linewidth=2, alpha=0.8)
        plt.plot(t_data['kf_wx'], t_data['kf_wy'], color=c, linestyle='--', linewidth=2, alpha=0.6)
        plt.scatter(t_data.iloc[0]['wx'], t_data.iloc[0]['wy'], color=c, marker='o', s=100, zorder=5, edgecolor='black')
        plt.scatter(t_data.iloc[-1]['wx'], t_data.iloc[-1]['wy'], color=c, marker='x', s=100, zorder=5, linewidth=3)

    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Ground Truth'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='KF Prediction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='Start', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='x', color='black', lw=0, label='End', markersize=10, markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Track IDs', markersize=8)
    ]
    plt.title(f"Ground Truth vs. Kalman Filter (Horizon={HORIZON_FRAMES})\nVideo: {video_name} | RMSE: {rmse:.3f}m")
    plt.legend(handles=legend_elements, loc='upper left')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kf_vs_gt_static.png'), dpi=150)
    plt.close()

    # --- 2. Animation (Fixed Cam) ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"KF Prediction (Fixed Cam)\nVideo: {video_name}")
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_aspect('equal') 

    all_x = pd.concat([valid_df['wx'], valid_df['kf_wx']])
    all_y = pd.concat([valid_df['wy'], valid_df['kf_wy']])
    
    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    
    pad_x = span_x * PAD_PERCENT
    pad_y = span_y * PAD_PERCENT
    
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    # -----------------------------

    legend_elements_ani = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='Ground Truth', markersize=8),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='black', label='KF Prediction', markersize=8, markeredgewidth=2),
        Line2D([0], [0], color='black', lw=1, label='Error Vector')
    ]
    ax.legend(handles=legend_elements_ani, loc='upper right')

    elements = {} 
    for i, tid in enumerate(track_ids):
        c = colors[i]
        gt_dot, = ax.plot([], [], 'o', color=c, markersize=8)
        kf_ghost, = ax.plot([], [], 'x', color=c, markersize=8, markeredgewidth=2, alpha=0.7)
        conn_line, = ax.plot([], [], '-', color=c, linewidth=1, alpha=0.5)
        elements[tid] = (gt_dot, kf_ghost, conn_line)

    frames = sorted(valid_df['frame'].unique())

    def update(frame_idx):
        curr_f = frames[frame_idx]
        artists = []

        for tid in track_ids:
            row = valid_df[(valid_df['track_id'] == tid) & (valid_df['frame'] == curr_f)]
            gt_dot, kf_ghost, conn_line = elements[tid]
            
            if not row.empty:
                gt_x, gt_y = row.iloc[0]['wx'], row.iloc[0]['wy']
                kf_x, kf_y = row.iloc[0]['kf_wx'], row.iloc[0]['kf_wy']
                gt_dot.set_data([gt_x], [gt_y])
                kf_ghost.set_data([kf_x], [kf_y])
                conn_line.set_data([gt_x, kf_x], [gt_y, kf_y])
            else:
                gt_dot.set_data([], [])
                kf_ghost.set_data([], [])
                conn_line.set_data([], [])
            artists.extend([gt_dot, kf_ghost, conn_line])
        
        return artists

    print(f"  -> Generating animation for {video_name}...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    writer = animation.PillowWriter(fps=15)
    ani.save(os.path.join(save_dir, 'kf_prediction_animation.gif'), writer=writer)
    plt.close()

def main():
    csv_files = glob.glob(os.path.join(INPUT_ROOT, '*', 'combined_data.csv'))
    all_metrics = []

    if not csv_files:
        print("No data found.")
        return

    for csv_path in csv_files:
        video_name = os.path.basename(os.path.dirname(csv_path))
        print(f"Processing {video_name}...")
        df = pd.read_csv(csv_path)
        df_processed, rmse = run_kf_on_video(df, video_name)
        all_metrics.append({'video': video_name, 'horizon_frames': HORIZON_FRAMES, 'rmse': rmse})
        visualize_comparison(df_processed, video_name, rmse)

    if all_metrics:
        stats_df = pd.DataFrame(all_metrics)
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        stats_df.to_csv(os.path.join(OUTPUT_ROOT, 'kf_rmse_scores.csv'), index=False)
        print("\nDone.")

if __name__ == "__main__":
    main()