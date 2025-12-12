import os
import json
import pandas as pd
import numpy as np
import glob

# --- Configuration ---
INPUT_ROOT = 'output_data/clean_stage2'
OUTPUT_ROOT = 'output_data/GNN_READY'
FRAME_INTERVAL = 2  # Step size for the dense timeline

DATA_COLUMNS = [
    'wx', 'wy', 'vx', 'vy', 'sx', 'sy', 
    'bw', 'bh', 'conf', 
    'wx_m', 'wy_m', 'vx_m', 'vy_m', 'scale_factor'
]

def process_stage2_data():
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input directory '{INPUT_ROOT}' does not exist.")
        return

    video_folders = [f for f in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, f))]

    if not video_folders:
        print("No video folders found.")
        return

    for video_name in video_folders:
        video_input_dir = os.path.join(INPUT_ROOT, video_name)
        video_output_dir = os.path.join(OUTPUT_ROOT, video_name)
        
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"Processing {video_name}...")

        all_objects_data = []
        global_max_frame = 0

        # 1. Get ALL subfolders (objects) and sort them
        # Sorting ensures track_1 always corresponds to the same folder every time you run this
        subfolders = [f for f in os.listdir(video_input_dir) if os.path.isdir(os.path.join(video_input_dir, f))]
        subfolders.sort()

        # 2. Iterate through all objects (Athletes + Others mixed)
        for index, folder_name in enumerate(subfolders):
            object_path = os.path.join(video_input_dir, folder_name)
            
            # Determine type: 'other' starts with "other", everything else is athlete
            is_athlete = not folder_name.lower().startswith("other")
            
            # Assign a universal ID: track_1, track_2, etc.
            # (index + 1) ensures we start from track_1
            track_id = f"track_{index + 1}"

            # Load ALL matching JSON files
            json_files = glob.glob(os.path.join(object_path, "data*.json"))
            
            if not json_files:
                continue

            combined_json_data = []
            for jf in json_files:
                try:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            combined_json_data.extend(data)
                        elif isinstance(data, dict):
                            combined_json_data.append(data)
                except Exception as e:
                    print(f"  Error reading {jf}: {e}")

            if not combined_json_data:
                continue

            df = pd.DataFrame(combined_json_data)
            
            # Sort and deduplicate
            if 'frame' in df.columns:
                df = df.sort_values('frame').drop_duplicates(subset=['frame'])
                max_f = df['frame'].max()
                if max_f > global_max_frame:
                    global_max_frame = max_f
            
            all_objects_data.append({
                'df': df,
                'track_id': track_id,
                'is_athlete': is_athlete
            })

        if not all_objects_data:
            print(f"  No valid object data found in {video_name}.")
            continue

        # 3. Create Dense Timeline & Format Columns
        final_dfs = []
        all_frames = np.arange(start=0, stop=global_max_frame + 1, step=FRAME_INTERVAL)
        timeline_df = pd.DataFrame({'frame': all_frames})

        for obj in all_objects_data:
            df = obj['df']
            is_athlete = obj['is_athlete']
            track_id = obj['track_id']

            # Merge with dense timeline
            merged_df = pd.merge(timeline_df, df, on='frame', how='left')

            # --- Logic for columns ---
            
            # mask: 1 if data existed in original file, 0 if padding
            check_col = 'wx' if 'wx' in merged_df.columns else (merged_df.columns[-1] if len(merged_df.columns) > 1 else 'frame')
            merged_df['mask'] = merged_df[check_col].notna().astype(int)

            # athlete_bool: 1 if athlete, 0 if other
            merged_df['athlete_bool'] = 1 if is_athlete else 0

            # track_id: Universal ID for everyone
            merged_df['track_id'] = track_id

            # Fill missing data columns with 0
            for col in DATA_COLUMNS:
                if col not in merged_df.columns:
                    merged_df[col] = 0
            
            merged_df[DATA_COLUMNS] = merged_df[DATA_COLUMNS].fillna(0)

            # Reorder columns
            # Note: Changed 'athlete_id' to 'track_id' per the new logic
            final_cols = ['athlete_bool', 'track_id', 'mask', 'frame'] + DATA_COLUMNS
            merged_df = merged_df[final_cols]
            
            final_dfs.append(merged_df)

        # 4. Save Combined CSV
        if final_dfs:
            combined_df = pd.concat(final_dfs, ignore_index=True)
            output_path = os.path.join(video_output_dir, 'combined_data.csv')
            combined_df.to_csv(output_path, index=False)
            print(f"  Saved {len(final_dfs)} total objects to: {output_path}")

if __name__ == "__main__":
    process_stage2_data()