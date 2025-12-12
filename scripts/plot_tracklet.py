import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# ==========================================
# 1. LOAD DATA
# ==========================================
# Option A: Load from a file (Uncomment the next 2 lines if using a file)
with open("output_data/MAX_0084/12810003/data.json", "r") as f:
    data = json.load(f)



df = pd.DataFrame(data)

# ==========================================
# 2. PLOTTING
# ==========================================
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# --- CONFIG: X-AXIS INTERVAL ---
# Change this to 10, 50, or 100 depending on how long your track is
FRAME_INTERVAL = 50

# --- PLOT 1: VELOCITY ---
axes[0].plot(df['frame'], df['vx_m'], label='Velocity X', color='blue', alpha=0.7)
axes[0].plot(df['frame'], df['vy_m'], label='Velocity Y', color='red', alpha=0.7)
axes[0].set_ylabel("Velocity (px/s)")
axes[0].set_title("1. Velocity Spikes")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# --- PLOT 2: SCREEN Y POSITION ---
axes[1].plot(df['frame'], df['sy'], label='Screen Y', color='green', marker='o', markersize=3)
axes[1].set_ylabel("Screen Y (px)")
axes[1].set_title("2. Vertical Lane Position")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].invert_yaxis()

# --- PLOT 3: BOX HEIGHT ---
axes[2].plot(df['frame'], df['bh'], label='Box Height', color='purple', marker='x')
axes[2].set_ylabel("Height (px)")
axes[2].set_title("3. Geometric Stability")
axes[2].grid(True, linestyle='--', alpha=0.5)

# --- PLOT 4: CONFIDENCE ---
axes[3].plot(df['frame'], df['conf'], label='Confidence', color='orange', marker='.')
axes[3].set_ylabel("Conf (0-1)")
axes[3].set_xlabel("Frame Number")
axes[3].set_title("4. Tracker Confidence")
axes[3].set_ylim(0, 1.1)
axes[3].grid(True, linestyle='--', alpha=0.5)
axes[3].axhline(0.3, color='red', linestyle=':', label='Low Threshold (0.3)')
axes[3].legend()

# --- APPLY X-AXIS TICKS ---
# Force ticks every N frames
locator = ticker.MultipleLocator(FRAME_INTERVAL)
axes[3].xaxis.set_major_locator(locator)

# Rotate labels slightly if they overlap
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()