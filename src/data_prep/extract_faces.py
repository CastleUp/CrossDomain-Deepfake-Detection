import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image
import random

# --- CONFIGURATION ---
FF_CSV_PATH = "FaceForensics++_C23/csv/FF++_Metadata.csv"
CELEB_TXT_PATH = "Celed_df/List_of_testing_videos.txt"
BASE_DIR = Path(__file__).parent.parent.parent
FF_DIR = BASE_DIR / "FaceForensics++_C23"
CELEB_DIR = BASE_DIR / "Celed_df"
OUTPUT_DIR = BASE_DIR / "data/processed"
PLOTS_DIR = BASE_DIR / "results/plots"

FRAMES_PER_VIDEO = 30 # Limit frames per video to 30 (user requested 25-50)
TARGET_SIZE = (224, 224)
MARGIN = 0.25 # 25% margin

# УСТАНОВИТЕ В FALSE ДЛЯ ПОЛНОГО ЦИКЛА (займет 15+ часов на CPU)
TEST_MODE = True
MAX_VIDEOS_FF = 1000 if TEST_MODE else None  # По 1000 видео на тест
MAX_VIDEOS_CELEB = 1000 if TEST_MODE else None

# Create output dirs
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize MTCNN
mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)

def get_frames_evenly_spaced(video_path, num_frames=FRAMES_PER_VIDEO):
    """Extract frames from video evenly spaced."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []
        
    step = max(1, total_frames // num_frames)
    frame_indices = list(range(0, total_frames, step))[:num_frames]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, frame))
            
    cap.release()
    return frames

def crop_face(frame, box, margin=MARGIN):
    """Crop face with margin."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    # Add margin
    x1 = max(0, int(x1 - w * margin))
    y1 = max(0, int(y1 - h * margin))
    x2 = min(frame.shape[1], int(x2 + w * margin))
    y2 = min(frame.shape[0], int(y2 + h * margin))
    
    crop = frame[y1:y2, x1:x2]
    # Resize to TARGET_SIZE
    if crop.size > 0:
        crop = cv2.resize(crop, TARGET_SIZE)
        return crop
    return None

def process_video(video_path, save_dir, video_name, is_fake):
    """Process a single video: extract frames, detect face, crop, resize and save."""
    os.makedirs(save_dir, exist_ok=True)
    frames = get_frames_evenly_spaced(video_path, FRAMES_PER_VIDEO)
    
    saved_paths = []
    examples = [] # Store some examples for visualization
    
    for i, (frame_idx, frame) in enumerate(frames):
        try:
            # Detect face
            boxes, _ = mtcnn.detect(Image.fromarray(frame))
            if boxes is not None and len(boxes) > 0:
                box = boxes[0] # Take the largest face
                face_crop = crop_face(frame, box)
                if face_crop is not None:
                    # Save image
                    out_name = f"{video_name}_frame{frame_idx}.jpg"
                    out_path = os.path.join(save_dir, out_name)
                    # Convert back to BGR for saving
                    cv2.imwrite(out_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                    saved_paths.append(out_path)
                    
                    if len(examples) < 2: # Keep max 2 examples per video for plot
                        examples.append((frame, face_crop))
        except Exception as e:
            continue
            
    return saved_paths, examples

def visualize_examples(examples, title, save_path):
    """Save a plot with original frames and cropped faces."""
    if not examples:
        return
    
    n = min(4, len(examples))
    fig, axes = plt.subplots(n, 2, figsize=(10, 5*n))
    fig.suptitle(title, fontsize=16)
    
    if n == 1:
        axes = [axes]
        
    for i in range(n):
        orig, crop = examples[i]
        axes[i][0].imshow(orig)
        axes[i][0].set_title("Original Frame")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(crop)
        axes[i][1].set_title("Cropped Face (224x224)")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("Starting face extraction...")
    all_examples = []
    
    # 1. Process FaceForensics++
    print("Processing FaceForensics++...")
    ff_csv = BASE_DIR / FF_CSV_PATH
    if ff_csv.exists():
        df = pd.read_csv(ff_csv)
        
        # Перемешиваем
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if MAX_VIDEOS_FF is not None:
            df = df.head(MAX_VIDEOS_FF)
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="FF++"):
            # Skip errors
            try:
                rel_path = str(row['File Path'])
                label_str = str(row['Label']).upper()
                
                vid_path = FF_DIR / rel_path
                if not vid_path.exists():
                    continue
                    
                is_fake = (label_str == 'FAKE')
                label_dir = "fake" if is_fake else "real"
                # Train/Val split: 80% train, 20% val
                split = "train" if index < 0.8 * len(df) else "val"
                
                save_dir = OUTPUT_DIR / "FF++" / split / label_dir
                vid_name = Path(rel_path).stem
                
                # Пропускаем видео, если оно уже было обработано ранее
                if len(list(save_dir.glob(f"{vid_name}_*.jpg"))) > 0:
                    continue
                
                _, ex = process_video(vid_path, save_dir, vid_name, is_fake)
                all_examples.extend(ex)
            except Exception as e:
                continue
    else:
        print(f"Could not find {ff_csv}")

    # 2. Process Celeb-DF
    print("Processing Celeb-DF...")
    celeb_txt = BASE_DIR / CELEB_TXT_PATH
    if celeb_txt.exists():
        with open(celeb_txt, 'r') as f:
            lines = f.readlines()
            
        import random
        random.seed(42)
        random.shuffle(lines)
        if MAX_VIDEOS_CELEB is not None:
            lines = lines[:MAX_VIDEOS_CELEB]
            
        for line in tqdm(lines, desc="Celeb-DF"):
            parts = line.strip().split()
            if len(parts) == 2:
                label_str, rel_path = parts
                vid_path = CELEB_DIR / rel_path
                if not vid_path.exists():
                    continue
                    
                is_fake = (label_str == '0') # 1 is real, 0 is fake
                label_dir = "fake" if is_fake else "real"
                save_dir = OUTPUT_DIR / "Celeb-DF" / "test" / label_dir
                vid_name = Path(rel_path).stem
                
                # Пропускаем видео, если оно уже было обработано ранее
                if len(list(save_dir.glob(f"{vid_name}_*.jpg"))) > 0:
                    continue
                
                _, ex = process_video(vid_path, save_dir, vid_name, is_fake)
                all_examples.extend(ex)
    else:
        print(f"Could not find {celeb_txt}")
        
    # Save some examples
    if all_examples:
        random.shuffle(all_examples)
        visualize_examples(all_examples[:4], "Face Extraction Examples", PLOTS_DIR / "extraction_examples.png")
        print(f"Saved visualization to {PLOTS_DIR / 'extraction_examples.png'}")

if __name__ == "__main__":
    main()
