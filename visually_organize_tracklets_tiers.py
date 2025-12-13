# Takes and organizes them on  how pure they are in terms of consistency from the vector resulting from passing image through ResNet
import os
import cv2
import json
import torch
import numpy as np
import shutil
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# CONFIG
TARGET_VIDEO_DIR = "output_data/MAX_0082" 

# Quality Thresholds (Lower variance = Better)
THRES_GOLD = 0.33   # Extremely tight/pure tracks
THRES_SILVER = 0.66 # Decent tracks
# Anything above 0.30 goes to "Rejected"

MIN_TRACK_LEN = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def calculate_track_stats(encoder, image_paths):
    """
    Returns: (variance_score, num_images)
    variance_score: Mean distance of all frames from the track center.
    """
    images = []
    # Sample to speed up
    sample_paths = image_paths[::3] if len(image_paths) > 30 else image_paths
    
    for p in sample_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
    if not images: return 1.0, 0

    embeddings = encoder.get_embedding_batch(images)
    
    # Normalize vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-6)
    
    # Calculate Centroid
    centroid = np.mean(embeddings, axis=0)
    centroid /= np.linalg.norm(centroid)
    
    # Calculate Variance (Mean Cosine Distance from Centroid)
    dists = 1.0 - np.dot(embeddings, centroid)
    mean_dist = np.mean(dists)
    
    return mean_dist, len(images)

def process_video_folder(video_dir, encoder):
    print(f"\nProcessing Video Folder: {video_dir}")
    
    # Create Output Tiers
    dirs = {
        "Gold": os.path.join(video_dir, "Sorted_Tier1_Gold"),
        "Silver": os.path.join(video_dir, "Sorted_Tier2_Silver"),
        "Rejected": os.path.join(video_dir, "Sorted_Rejected")
    }
    
    for d in dirs.values():
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)
    
    # Get tracklet folders
    track_ids = [d for d in os.listdir(video_dir) 
                 if os.path.isdir(os.path.join(video_dir, d)) 
                 and d.isdigit()] # Only process numeric IDs (ignore previous output folders)

    print(f"Scanning {len(track_ids)} tracklets...")
    
    stats = {"Gold": 0, "Silver": 0, "Rejected": 0}

    for tid in tqdm(track_ids):
        track_path = os.path.join(video_dir, tid)
        json_path = os.path.join(track_path, "data.json")
        images = sorted([os.path.join(track_path, f) for f in os.listdir(track_path) if f.endswith('.jpg')])
        
        # 1. Sanity Checks
        if not os.path.exists(json_path) or len(images) < MIN_TRACK_LEN:
            shutil.move(track_path, os.path.join(dirs["Rejected"], tid))
            stats["Rejected"] += 1
            continue

        # 2. Consistency Check
        variance, count = calculate_track_stats(encoder, images)
        
        # 3. Sorting Logic
        if variance < THRES_GOLD:
            dest = dirs["Gold"]
            category = "Gold"
        elif variance < THRES_SILVER:
            dest = dirs["Silver"]
            category = "Silver"
        else:
            dest = dirs["Rejected"]
            category = "Rejected"
            
        shutil.move(track_path, os.path.join(dest, tid))
        
        stats[category] += 1

    print("\nProcessing Complete.")
    print(f"  - Tier 1 (Gold):   {stats['Gold']} tracks (Pure)")
    print(f"  - Tier 2 (Silver): {stats['Silver']} tracks (Okay)")
    print(f"  - Rejected:        {stats['Rejected']} tracks (Noise/Short)")
    print(f"Check the 'Sorted_*' folders in {video_dir}")


def main():
    if not os.path.exists(TARGET_VIDEO_DIR):
        print(f"Error: {TARGET_VIDEO_DIR} not found.")
        return

    encoder = IdentityEncoder()
    process_video_folder(TARGET_VIDEO_DIR, encoder)

if __name__ == "__main__":
    main()