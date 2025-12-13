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

# CONFIG
INPUT_FOLDER = 'baha2'
TARGET_CLASS_NAME = "Athlete"
YOLO_MODEL_PATH = 'models/best_color2.pt'
OUTPUT_DIR = "output_data"
SHOW_PREVIEW = False        
FRAME_STRIDE = 2              
STABILIZATION_SCALE = 0.5     
WORKER_THREADS = 4

# Filtering Parameters
CONF_THRESHOLD = 0.1          
MIN_TRACK_DURATION_SEC = 1  
SAVE_IMAGE_INTERVAL_SEC = 0.5
PADDING_PERCENT = 0.15

# Velocity Smoothing
VELOCITY_WINDOW = 15          
MIN_SPEED_THRESHOLD = 50.0    

# TRACKER CONFIG
TRACKER_CONFIG_PATH = "models/custom_botsort.yaml"

# UTILITY CLASSES
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
        if self.active: self.q.put((path, img))

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None: break
            path, img = item
            try: cv2.imwrite(path, img)
            except Exception as e: print(f"Error saving {path}: {e}")
            self.q.task_done()

    def stop(self):
        self.active = False
        self.q.join()
        for _ in range(WORKER_THREADS): self.q.put(None)

class Stabilizer:
    def __init__(self, width, height):
        self.scale = STABILIZATION_SCALE
        self.w_small = int(width * self.scale)
        self.h_small = int(height * self.scale)
        self.orb = cv2.ORB_create(nfeatures=1000)
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.prev_kp = None
        self.prev_des = None
        self.H_cumulative = np.eye(3)

    def update(self, frame_gray, mask):
        gray_small = cv2.resize(frame_gray, (self.w_small, self.h_small))
        mask_small = None
        if mask is not None: mask_small = cv2.resize(mask, (self.w_small, self.h_small))
        
        kp_curr, des_curr = self.orb.detectAndCompute(gray_small, mask=mask_small)
        
        if des_curr is not None and self.prev_des is not None and len(des_curr) > 10 and len(self.prev_des) > 10:
            matches = self.flann.knnMatch(self.prev_des, des_curr, k=2)
            good_matches = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance]

            if len(good_matches) > 10:
                pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / self.scale
                pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / self.scale
                M_frame, _ = cv2.estimateAffinePartial2D(pts_curr, pts_prev, method=cv2.RANSAC)
                if M_frame is not None:
                    row = np.array([[0.0, 0.0, 1.0]])
                    H_frame = np.vstack([M_frame, row])
                    self.H_cumulative = np.dot(self.H_cumulative, H_frame)

        self.prev_kp = kp_curr
        self.prev_des = des_curr
        return self.H_cumulative

# PROCESSING FUNCTIONS
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

def split_tracks_on_glitches(raw_tracks):
    """
    Detects sudden shape changes. Splits track to prevent Identity Swaps.
    """
    split_tracks = {}
    parent_map = {} 
    
    # SAFETY SHIFT: Ensure this is larger than the max expected IDs in a video
    ID_OFFSET_MULTIPLIER = 10000 
    
    for tid, data in raw_tracks.items():
        if not data: continue
        
        heights = [p['bh'] for p in data]
        median_h = np.median(heights)
        
        current_segment = []
        segment_idx = 0
        
        for i, p in enumerate(data):
            is_glitch = False
            
            # Glitch Checks
            if p['bh'] < (median_h * 0.70): is_glitch = True
            elif p['bh'] > (median_h * 1.3): is_glitch = True
            elif p['c   onf'] < 0.15: is_glitch = True
            
            if is_glitch:
                if len(current_segment) > 5:
                    if segment_idx == 0:
                        new_tid = tid
                    else:
                        new_tid = (tid * ID_OFFSET_MULTIPLIER) + segment_idx
                        parent_map[new_tid] = tid 
                    
                    split_tracks[new_tid] = current_segment
                    segment_idx += 1
                
                current_segment = []
            else:
                current_segment.append(p)
        
        # Save tail
        if len(current_segment) > 5:
            if segment_idx == 0:
                new_tid = tid
            else:
                new_tid = (tid * ID_OFFSET_MULTIPLIER) + segment_idx
                parent_map[new_tid] = tid
                
            split_tracks[new_tid] = current_segment

    return split_tracks, parent_map

def reorganize_split_images(video_out_dir, final_tracks, parent_map):
    for tid, data in final_tracks.items():
        if tid not in parent_map: continue
        
        original_id = parent_map[tid]
        src_folder = os.path.join(video_out_dir, str(original_id))
        dst_folder = os.path.join(video_out_dir, str(tid))
        
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
            
        for p in data:
            frame_num = p['frame']
            filename = f"{frame_num:06d}.jpg"
            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(dst_folder, filename)
            
            if os.path.exists(src_path):
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"Warning: Failed to move {filename} from {original_id} to {tid}: {e}")

def filter_trajectories(tracks, effective_fps):
    clean = {}
    min_frames = int(MIN_TRACK_DURATION_SEC * effective_fps)
    
    for tid, data in tracks.items():
        if len(data) < min_frames: continue
        w_points = np.array([(d['wx'], d['wy']) for d in data])
        total_dist = np.sum(np.sqrt(np.sum(np.diff(w_points, axis=0)**2, axis=1)))
        if total_dist < 50: continue
        clean[tid] = data
    return clean

def normalize_tracks_dynamic(clean_tracks, total_frames, reference_height_m=1.75):
    frame_heights = defaultdict(list)
    for tid, data in clean_tracks.items():
        for p in data:
            if p['bh'] > p['bw']: 
                frame_heights[p['frame']].append(p['bh'])

    raw_scales = np.zeros(total_frames + 500)
    valid_frames = sorted(frame_heights.keys())
    
    if not valid_frames: return clean_tracks

    frame_medians = {f: np.median(frame_heights[f]) for f in valid_frames}
    current_h = frame_medians[valid_frames[0]]
    
    for f in range(1, total_frames + 1):
        if f in frame_medians: current_h = frame_medians[f]
        if current_h > 0: raw_scales[f] = reference_height_m / current_h
        else: raw_scales[f] = 1.0

    window = 60
    smooth_scales = np.convolve(raw_scales, np.ones(window)/window, mode='same')

    normalized_tracks = {}
    for tid, data in clean_tracks.items():
        new_data = []
        for p in data:
            f = p['frame']
            s = smooth_scales[f] if f < len(smooth_scales) else smooth_scales[-1]
            new_p = p.copy()
            new_p['wx_m'] = p['wx'] * s
            new_p['wy_m'] = p['wy'] * s
            new_p['vx_m'] = p['vx'] * s
            new_p['vy_m'] = p['vy'] * s
            new_p['scale_factor'] = s
            new_data.append(new_p)
        normalized_tracks[tid] = new_data
        
    return normalized_tracks

def process_single_video(video_path, model, target_id):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(OUTPUT_DIR, video_name)
    
    if os.path.exists(video_out_dir) and len(os.listdir(video_out_dir)) > 0:
        print(f"Skipping {video_name} (Already processed).")
        return

    print(f"\n=== Processing: {video_name} ===")
    if os.path.exists(video_out_dir): shutil.rmtree(video_out_dir)
    os.makedirs(video_out_dir)

    cap = cv2.VideoCapture(video_path)
    base_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    effective_fps = base_fps / FRAME_STRIDE
    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stab = Stabilizer(width, height)
    saver = ImageSaver()
    raw_tracks = defaultdict(list)
    last_img_time = defaultdict(float)
    
    ret, prev_frame = cap.read()
    if not ret: return
    stab.update(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), None)

    raw_frame_count = 0
    pbar = tqdm(total=total_raw_frames, unit="frames")

    while True:
        ret, frame = cap.read()
        if not ret: break
        raw_frame_count += 1
        pbar.update(1)

        if raw_frame_count % FRAME_STRIDE != 0: continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stab_mask = np.ones_like(frame_gray, dtype=np.uint8) * 255
        
        results = model.track(frame, persist=True, tracker=TRACKER_CONFIG_PATH, 
                              classes=[target_id], conf=CONF_THRESHOLD, imgsz=640, verbose=False)
        
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

        if SHOW_PREVIEW:
            annotated_frame = frame.copy()
            for (tid, x1, y1, x2, y2, conf) in current_data:
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"ID:{tid} {conf:.2f}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ph, pw = annotated_frame.shape[:2]
            if pw > 1920:
                scale_view = 1920 / pw
                annotated_frame = cv2.resize(annotated_frame, (1920, int(ph * scale_view)))

            cv2.imshow(f"Preview - {video_name}", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[STOP] User pressed 'q'. Exiting...")
                break

        H_cumulative = stab.update(frame_gray, stab_mask)

        for (tid, x1, y1, x2, y2, conf) in current_data:
            cx, cy = (x1 + x2) / 2, y2
            screen_pt = np.array([[[cx, cy]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(screen_pt, H_cumulative)
            wx, wy = world_pt[0][0]

            vx, vy = 0.0, 0.0
            history = raw_tracks[tid]
            
            if len(history) >= VELOCITY_WINDOW:
                prev = history[-VELOCITY_WINDOW]
                dt = (raw_frame_count - prev['frame']) / base_fps
                if dt > 0:
                    raw_vx, raw_vy = (wx - prev['wx']) / dt, (wy - prev['wy']) / dt
                    if np.sqrt(raw_vx**2 + raw_vy**2) > MIN_SPEED_THRESHOLD:
                        vx, vy = raw_vx, raw_vy
            elif len(history) > 0:
                prev = history[-1]
                dt = (raw_frame_count - prev['frame']) / base_fps
                if dt > 0:
                    raw_vx, raw_vy = (wx - prev['wx']) / dt, (wy - prev['wy']) / dt
                    if np.sqrt(raw_vx**2 + raw_vy**2) > MIN_SPEED_THRESHOLD:
                        vx, vy = raw_vx, raw_vy

            raw_tracks[tid].append({
                'frame': raw_frame_count, 'wx': wx, 'wy': wy, 'vx': vx, 'vy': vy,
                'sx': cx, 'sy': cy, 'bw': (x2-x1), 'bh': (y2-y1), 'conf': conf
            })

            duration = len(raw_tracks[tid]) / effective_fps
            if duration >= MIN_TRACK_DURATION_SEC:
                if (raw_frame_count - last_img_time[tid]) >= (base_fps * SAVE_IMAGE_INTERVAL_SEC):
                    s_dir = os.path.join(video_out_dir, str(tid))
                    os.makedirs(s_dir, exist_ok=True)
                    crop = get_padded_crop(frame, (x1, y1, x2, y2), PADDING_PERCENT)
                    if crop.size > 0:
                        saver.save(os.path.join(s_dir, f"{raw_frame_count:06d}.jpg"), crop.copy())
                        last_img_time[tid] = raw_frame_count

    pbar.close()
    
    # CLEANUP 
    cv2.destroyAllWindows()

    print("Waiting for image saver...")
    saver.stop()
    cap.release()
    
    # POST PROCESSING 
    print("1. Splitting tracks on glitches (Anti-Mixup)...")
    split_tracks, parent_map = split_tracks_on_glitches(raw_tracks)
    print(f"   - Tracks: {len(raw_tracks)} -> {len(split_tracks)}")
    
    print("2. Filtering tracks...")
    clean_tracks = filter_trajectories(split_tracks, effective_fps)
    
    print("3. Applying Dynamic Normalization...")
    final_tracks = normalize_tracks_dynamic(clean_tracks, total_raw_frames)
    
    print("4. Reorganizing Image Folders for Splits...")
    reorganize_split_images(video_out_dir, final_tracks, parent_map)
    
    print(f"5. Saving {len(final_tracks)} final tracks...")
    for tid, data in final_tracks.items():
        t_dir = os.path.join(video_out_dir, str(tid))
        if not os.path.exists(t_dir): os.makedirs(t_dir)
        
        serializable = []
        for d in data:
            serializable.append({k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) for k,v in d.items()})
        with open(os.path.join(t_dir, "data.json"), 'w') as f:
            json.dump(serializable, f, indent=4)

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: {INPUT_FOLDER} not found.")
        return
    print(f"Loading Model: {YOLO_MODEL_PATH}...")
    model = YOLO(YOLO_MODEL_PATH)
    
    target_id = None
    for cid, cname in model.names.items():
        if cname == TARGET_CLASS_NAME: target_id = cid; break
    if target_id is None: print("Class not found."); return

    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp4', '.mov'))]
    print(f"Found {len(video_files)} videos.")

    for vid in video_files:
        try: process_single_video(os.path.join(INPUT_FOLDER, vid), model, target_id)
        except Exception as e: print(f"Error processing {vid}: {e}")

    print("\nAll videos processed.")

if __name__ == "__main__":
    main()