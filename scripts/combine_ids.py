import os
import cv2
import json
import torch
import numpy as np
import shutil
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "output_data/MAX_0092"
NUM_ATHLETES = 8

# CLUSTERING THRESHOLD
# 0.6 was too loose (merged everyone). 
# 0.35 is tighter. If you still get 1 giant group, lower this to 0.25.
SIMILARITY_THRESHOLD = 0.35  

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# QUALITY CONTROL THRESHOLDS
MIN_CLEAN_FRAMES = 5          # Need at least 5 "perfect" frames to trust a tracklet
PURITY_THRESHOLD = 0.75       # Start and End of tracklet must look 75% similar
ASPECT_RATIO_MIN = 1.2        # H/W must be > 1.2 (Tall rectangle)
MEDIAN_WIDTH_TOLERANCE = 1.4  # Reject frames where box width > 1.4x the median

# ==========================================
# 1. FEATURE EXTRACTOR
# ==========================================
class IdentityEncoder:
    def __init__(self):
        print(f"Loading ResNet18 on {DEVICE}...")
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(DEVICE)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding_batch(self, images):
        if not images: return None
        tensors = [self.preprocess(img) for img in images]
        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            embeddings = self.model(batch).flatten(start_dim=1)
        return embeddings.cpu().numpy()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_track_metadata(track_dir):
    """Loads the JSON data to get bounding box sizes and time."""
    json_path = os.path.join(track_dir, "data.json")
    if not os.path.exists(json_path): return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def filter_clean_frames(data_points):
    """
    Returns indices of frames that are geometrically 'clean'.
    Filters out merged blobs (too wide) and squashy boxes.
    """
    if not data_points: return []

    widths = [d['bw'] for d in data_points]
    heights = [d['bh'] for d in data_points]
    
    median_w = np.median(widths)
    valid_indices = []

    for i in range(len(data_points)):
        w = widths[i]
        h = heights[i]
        
        # Aspect Ratio Check (Must be tall)
        if (h / w) < ASPECT_RATIO_MIN:
            continue

        # Relative Width Check (Blob Detector)
        if w > (median_w * MEDIAN_WIDTH_TOLERANCE):
            continue

        valid_indices.append(i)

    return valid_indices

def is_tracklet_pure(encoder, images):
    """Checks if the identity is consistent from Start to End."""
    n = len(images)
    if n < 5: return True 

    start_batch = images[:3]
    end_batch = images[-3:]

    start_vecs = encoder.get_embedding_batch(start_batch)
    end_vecs = encoder.get_embedding_batch(end_batch)
    
    v_start = np.mean(start_vecs, axis=0)
    v_end = np.mean(end_vecs, axis=0)
    
    v_start /= np.linalg.norm(v_start)
    v_end /= np.linalg.norm(v_end)
    
    sim = np.dot(v_start, v_end)
    return sim > PURITY_THRESHOLD

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    encoder = IdentityEncoder()
    
    tracklet_ids = []
    tracklet_embeddings = []
    tracklet_intervals = [] # Store (start_frame, end_frame) for constraints
    
    stats = {'total': 0, 'bad_geom': 0, 'bad_pure': 0, 'good': 0}

    print("--- Step 1: Scanning & Filtering Data ---")
    
    for item in os.listdir(DATA_DIR):
        track_dir = os.path.join(DATA_DIR, item)
        if not (os.path.isdir(track_dir) and item.isdigit()): continue
        
        stats['total'] += 1
        
        # A. Load Metadata
        meta_data = load_track_metadata(track_dir)
        if not meta_data: continue
        
        # B. Filter Clean Frames
        clean_indices = filter_clean_frames(meta_data)
        
        if len(clean_indices) < MIN_CLEAN_FRAMES:
            stats['bad_geom'] += 1
            continue

        # C. Load Images
        clean_images = []
        for idx in clean_indices:
            frame_num = meta_data[idx]['frame']
            img_name = f"{frame_num:06d}.jpg"
            img_path = os.path.join(track_dir, img_name)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    clean_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not clean_images: continue

        # D. Purity Check
        if not is_tracklet_pure(encoder, clean_images):
            stats['bad_pure'] += 1
            print(f"  ID {item}: Rejected (Impure/ID Swap detected)")
            continue

        # E. Compute Embedding & Store Interval
        step = max(1, len(clean_images) // 10)
        sample_imgs = clean_images[::step][:10]
        
        vecs = encoder.get_embedding_batch(sample_imgs)
        mean_embedding = np.mean(vecs, axis=0)
        
        norm = np.linalg.norm(mean_embedding)
        if norm > 0: mean_embedding = mean_embedding / norm
        
        # Store Data
        tracklet_ids.append(int(item))
        tracklet_embeddings.append(mean_embedding)
        
        # Get Frame Interval from metadata
        frames = [d['frame'] for d in meta_data]
        tracklet_intervals.append((min(frames), max(frames)))
        
        stats['good'] += 1

    print(f"\nStats: Scanned {stats['total']} IDs.")
    print(f"  - Rejected {stats['bad_geom']} for bad geometry")
    print(f"  - Rejected {stats['bad_pure']} for impurity")
    print(f"  - Kept {stats['good']} Clean IDs")

    if stats['good'] == 0:
        print("No valid data left!")
        return

    # --- Step 2: Clustering ---
    print("\n--- Step 2: Clustering Identities ---")
    
    X = np.array(tracklet_embeddings)
    
    # A. Compute Distance Matrix
    sim_matrix = np.dot(X, X.T)
    dist_matrix = 1.0 - sim_matrix
    dist_matrix[dist_matrix < 0] = 0
    
    # DEBUG: Check if ResNet thinks everyone is identical
    avg_dist = np.mean(dist_matrix)
    print(f"DEBUG: Avg Visual Distance: {avg_dist:.4f} (If < 0.2, threshold must be very low)")

    # B. Apply Time Constraints (Crucial for "Jersey Problem")
    print("Applying Time Overlap Constraints...")
    constraints_count = 0
    n = len(tracklet_ids)
    
    for i in range(n):
        for j in range(i + 1, n):
            start_i, end_i = tracklet_intervals[i]
            start_j, end_j = tracklet_intervals[j]
            
            # Check overlap
            if start_i <= end_j and start_j <= end_i:
                # Force separate
                dist_matrix[i, j] = 100.0
                dist_matrix[j, i] = 100.0
                constraints_count += 1
                
    print(f"Applied {constraints_count} 'Cannot-Link' constraints.")

    # C. Run Clustering
    clusterer = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=SIMILARITY_THRESHOLD,
        metric='precomputed', 
        linkage='average' 
    )
    labels = clusterer.fit_predict(dist_matrix)
    print(f"Algorithm found {len(set(labels))} potential unique identities.")

    # --- Step 3: Build Gallery ---
    print("\n--- Step 3: Building Gallery ---")
    
    raw_groups = defaultdict(list)
    for original_id, group_id in zip(tracklet_ids, labels):
        raw_groups[int(group_id)].append(original_id)

    sorted_groups = sorted(raw_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    final_mapping = {}
    trash_mapping = {}
    final_groups = {} # Define properly
    
    for i, (group_id, ids) in enumerate(sorted_groups):
        if i < NUM_ATHLETES:
            print(f"  Athlete {i}: Formed from {len(ids)} tracklets.")
            final_groups[i] = ids 
            for oid in ids: final_mapping[oid] = i
        else:
            for oid in ids: trash_mapping[oid] = -1

    # Save Mapping
    with open(os.path.join(DATA_DIR, "gallery_mapping.json"), 'w') as f:
        json.dump(final_mapping, f, indent=4)

    # --- Step 4: Visualize ---
    vis_dir = os.path.join(DATA_DIR, "gallery_clean")
    if os.path.exists(vis_dir): shutil.rmtree(vis_dir)
    os.makedirs(vis_dir)
    
    for new_id, original_ids in final_groups.items():
        group_folder = os.path.join(vis_dir, f"Athlete_{new_id}")
        os.makedirs(group_folder, exist_ok=True)
        
        # Save one representative image
        for oid in original_ids:
            oid_path = os.path.join(DATA_DIR, str(oid))
            images = [f for f in os.listdir(oid_path) if f.endswith('.jpg')]
            if images:
                shutil.copy(os.path.join(oid_path, images[0]), 
                            os.path.join(group_folder, f"id_{oid}.jpg"))

    print(f"\nGallery saved to '{vis_dir}'.")

if __name__ == "__main__":
    main()