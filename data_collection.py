import cv2
import numpy as np
import os
import json
import shutil
import threading
import queue
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = 'baha2'          # <--- FOLDER containing videos
TARGET_CLASS_NAME = "Athlete"
YOLO_MODEL_PATH = 'models/best_color.pt'
OUTPUT_DIR = "output_data"

# Performance Settings
SHOW_PREVIEW = False
FRAME_STRIDE = 2              # Process every Nth frame (2 = Double Speed)
STABILIZATION_SCALE = 0.5     # Downscale ORB
WORKER_THREADS = 2

# Filtering & Collection
CONF_THRESHOLD = 0.1
MIN_TRACK_DURATION_SEC = 2.0  
SAVE_IMAGE_INTERVAL_SEC = 1.0
PADDING_PERCENT = 0.15
VELOCITY_WINDOW = 5           # Smooth velocity over last 5 data points

# ==========================================
# 0. CREATE FAST TRACKER CONFIG
# ==========================================
# We create this once to ensure it exists
TRACKER_CONFIG_PATH = "models/custom_botsort.yaml"

# ==========================================
# THREADED IMAGE SAVER
# ==========================================
class ImageSaver:
    def __init__(self):
        self.q = queue.Queue()
        self.active = True
        self.threads = []
        for _ in range(WORKER_THREADS):
            t = threading.Thread(target=self._worker)
            t.daemon = True
            t.start()
            self.threads.append(t)

    def save(self, path, img):
        if self.active:
            self.q.put((path, img))

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None: break
            path, img = item
            try:
                cv2.imwrite(path, img)
            except Exception as e:
                print(f"Error saving {path}: {e}")
            self.q.task_done()

    def stop(self):
        self.active = False
        self.q.join()
        for _ in range(WORKER_THREADS):
            self.q.put(None)

# ==========================================
# STABILIZER
# ==========================================
class Stabilizer:
    def __init__(self, width, height):
        self.scale = STABILIZATION_SCALE
        self.w_small = int(width * self.scale)
        self.h_small = int(height * self.scale)
        
        self.orb = cv2.ORB_create(nfeatures=1000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.prev_kp = None
        self.prev_des = None
        self.H_cumulative = np.eye(3)

    def update(self, frame_gray, mask):
        # Resize
        gray_small = cv2.resize(frame_gray, (self.w_small, self.h_small))
        mask_small = None
        if mask is not None:
            mask_small = cv2.resize(mask, (self.w_small, self.h_small))
        
        # Detect
        kp_curr, des_curr = self.orb.detectAndCompute(gray_small, mask=mask_small)
        
        if des_curr is not None and self.prev_des is not None and len(des_curr) > 10 and len(self.prev_des) > 10:
            matches = self.flann.knnMatch(self.prev_des, des_curr, k=2)
            good_matches = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance]

            if len(good_matches) > 10:
                pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / self.scale
                pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / self.scale
                
                # Estimate Motion
                M_frame, _ = cv2.estimateAffinePartial2D(pts_curr, pts_prev, method=cv2.RANSAC)
                
                if M_frame is not None:
                    row = np.array([[0.0, 0.0, 1.0]])
                    H_frame = np.vstack([M_frame, row])
                    self.H_cumulative = np.dot(self.H_cumulative, H_frame)

        self.prev_kp = kp_curr
        self.prev_des = des_curr
        return self.H_cumulative

# ==========================================
# HELPERS
# ==========================================
def get_padded_crop(frame, box, padding_percent=0.15):
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img = frame.shape[:2]
    w, h = x2 - x1, y2 - y1
    pad_w = int(w * padding_percent)
    pad_h = int(h * padding_percent)
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - int(pad_h * 0.5))
    nx2 = min(w_img, x2 + pad_w)
    ny2 = min(h_img, y2 + pad_h)
    return frame[ny1:ny2, nx1:nx2]

def filter_trajectories(raw_tracks, effective_fps):
    clean_tracks = {}
    min_frames = int(MIN_TRACK_DURATION_SEC * effective_fps)
    
    valid_candidates = {}
    avg_speeds = []

    for tid, data in raw_tracks.items():
        if len(data) < min_frames: continue

        w_points = np.array([(d['wx'], d['wy']) for d in data])
        total_dist = np.sum(np.sqrt(np.sum(np.diff(w_points, axis=0)**2, axis=1)))
        
        if total_dist < 50: continue

        duration = (data[-1]['frame'] - data[0]['frame']) / effective_fps
        avg_speed = total_dist / duration if duration > 0 else 0
        
        # Velocity Consistency
        vels = []
        for i in range(1, len(data)):
             dt = (data[i]['frame'] - data[i-1]['frame']) / effective_fps
             if dt > 0:
                 dx = data[i]['wx'] - data[i-1]['wx']
                 dy = data[i]['wy'] - data[i-1]['wy']
                 vels.append(np.sqrt(dx*dx + dy*dy) / dt)
        
        if not vels: continue
        vel_std = np.std(vels)

        valid_candidates[tid] = { 'data': data, 'avg_speed': avg_speed, 'vel_std': vel_std }
        avg_speeds.append(avg_speed)

    if not avg_speeds: return {}
    median_speed = np.median(avg_speeds)
    
    for tid, info in valid_candidates.items():
        avg, std = info['avg_speed'], info['vel_std']
        
        if avg > (median_speed * 5.0): continue
        if avg < (median_speed / 5.0): continue
        if std > (avg * 3.0): continue
        
        clean_tracks[tid] = info['data']

    return clean_tracks

def process_single_video(video_path, model, target_id):
    """
    Processes one video file. Returns True if successful.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n=== Processing: {video_name} ===")
    
    # Setup Output Directory
    video_out_dir = os.path.join(OUTPUT_DIR, video_name)
    
    # SKIP CHECK: If folder exists and has data, skip it
    if os.path.exists(video_out_dir) and len(os.listdir(video_out_dir)) > 0:
        print(f"Skipping {video_name} (Already processed). Delete folder to re-run.")
        return True

    if os.path.exists(video_out_dir): shutil.rmtree(video_out_dir)
    os.makedirs(video_out_dir)

    cap = cv2.VideoCapture(video_path)
    base_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    effective_fps = base_fps / FRAME_STRIDE
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stab = Stabilizer(width, height)
    saver = ImageSaver() # Start new saver threads for this video

    raw_tracks = defaultdict(list)
    last_image_save_time = defaultdict(float)
    
    # Read first frame for initialization
    ret, prev_frame = cap.read()
    if not ret: 
        print(f"Could not read {video_path}")
        return False
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    stab.update(prev_gray, mask=None)

    raw_frame_count = 0
    pbar = tqdm(total=total_raw_frames, unit="frames", desc=f"Track: {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        raw_frame_count += 1
        pbar.update(1)

        # Frame Skipping
        if raw_frame_count % FRAME_STRIDE != 0:
            continue
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stab_mask = np.ones_like(frame_gray, dtype=np.uint8) * 255
        
        # YOLO Tracking
        results = model.track(frame, persist=True, tracker=TRACKER_CONFIG_PATH, 
                              classes=[target_id], conf=CONF_THRESHOLD, 
                              imgsz=640, verbose=False)
        
        current_data = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                current_data.append((tid, x1, y1, x2, y2, conf))
                
                pad = 20
                mx1, my1 = max(0, int(x1-pad)), max(0, int(y1-pad))
                mx2, my2 = min(width, int(x2+pad)), min(height, int(y2+pad))
                cv2.rectangle(stab_mask, (mx1, my1), (mx2, my2), 0, -1)

        # Stabilize
        H_cumulative = stab.update(frame_gray, stab_mask)

        # Store Data
        for (tid, x1, y1, x2, y2, conf) in current_data:
            cx, cy = (x1 + x2) / 2, y2
            screen_pt = np.array([[[cx, cy]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(screen_pt, H_cumulative)
            wx, wy = world_pt[0][0]

            # Velocity (Sliding Window)
            vx, vy = 0.0, 0.0
            history = raw_tracks[tid]
            
            if len(history) >= VELOCITY_WINDOW:
                prev = history[-VELOCITY_WINDOW]
                dt = (raw_frame_count - prev['frame']) / base_fps
                if dt > 0:
                    vx = (wx - prev['wx']) / dt
                    vy = (wy - prev['wy']) / dt
            elif len(history) > 0:
                prev = history[-1]
                dt = (raw_frame_count - prev['frame']) / base_fps
                if dt > 0:
                    vx = (wx - prev['wx']) / dt
                    vy = (wy - prev['wy']) / dt

            raw_tracks[tid].append({
                'frame': raw_frame_count,
                'wx': wx, 'wy': wy,
                'vx': vx, 'vy': vy,
                'sx': cx, 'sy': cy,
                'bw': (x2-x1), 'bh': (y2-y1),
                'conf': conf
            })

            # Save Image
            duration = len(raw_tracks[tid]) / effective_fps
            if duration >= MIN_TRACK_DURATION_SEC:
                if (raw_frame_count - last_image_save_time[tid]) >= (base_fps * SAVE_IMAGE_INTERVAL_SEC):
                    save_dir = os.path.join(video_out_dir, str(tid))
                    os.makedirs(save_dir, exist_ok=True)
                    crop = get_padded_crop(frame, (x1, y1, x2, y2), PADDING_PERCENT)
                    if crop.size > 0:
                        saver.save(os.path.join(save_dir, f"{raw_frame_count:06d}.jpg"), crop.copy())
                        last_image_save_time[tid] = raw_frame_count

    pbar.close()
    print(f"Waiting for image saver for {video_name}...")
    saver.stop()
    cap.release()
    
    # Save JSON
    print("Filtering tracks...")
    clean_tracks = filter_trajectories(raw_tracks, effective_fps)
    
    print(f"Saving {len(clean_tracks)} tracks for {video_name}...")
    for tid, data in clean_tracks.items():
        track_dir = os.path.join(video_out_dir, str(tid))
        if not os.path.exists(track_dir): os.makedirs(track_dir)
        
        serializable = []
        for d in data:
            serializable.append({k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) for k,v in d.items()})
            
        with open(os.path.join(track_dir, "data.json"), 'w') as f:
            json.dump(serializable, f, indent=4)
            
    return True

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Folder '{INPUT_FOLDER}' not found.")
        return

    # Load Model ONCE
    print(f"Loading YOLO Model from {YOLO_MODEL_PATH}...")
    model = YOLO(YOLO_MODEL_PATH)
    
    target_id = None
    for cid, cname in model.names.items():
        if cname == TARGET_CLASS_NAME:
            target_id = cid; break
    if target_id is None: print(f"Error: Class '{TARGET_CLASS_NAME}' not found."); return

    # Find all videos
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp4', '.mov'))]
    print(f"Found {len(video_files)} videos in {INPUT_FOLDER}")

    for vid_file in video_files:
        full_path = os.path.join(INPUT_FOLDER, vid_file)
        try:
            process_single_video(full_path, model, target_id)
        except Exception as e:
            print(f"CRITICAL ERROR processing {vid_file}: {e}")
            continue

    print("\n\nAll videos processed.")

if __name__ == "__main__":
    main()