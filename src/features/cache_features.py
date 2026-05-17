import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import sys
# Add src to path so we can import from models
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.feature_extractors import get_extractors

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"
FEATURES_DIR = BASE_DIR / "data/features"
PLOTS_DIR = BASE_DIR / "results/plots"

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

class DeepfakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0 for real, 1 for fake
        
        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"
        
        if real_dir.exists():
            for img_path in real_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(0)
                
        if fake_dir.exists():
            for img_path in fake_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, str(img_path)

def visualize_tsne(features_np, labels_np, paradigm_name, dataset_name, split):
    """Generate and save t-SNE plot for features."""
    print(f"Generating t-SNE plot for {paradigm_name} ({dataset_name} {split})...")
    
    # If too many samples, subsample for t-SNE to save time
    max_samples = 2000
    if len(features_np) > max_samples:
        indices = np.random.choice(len(features_np), max_samples, replace=False)
        features_np = features_np[indices]
        labels_np = labels_np[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features_np)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels_np,
        palette=sns.color_palette("hls", 2),
        alpha=0.6,
        s=40
    )
    plt.title(f't-SNE of {paradigm_name} Features ({dataset_name} {split})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=['Real', 'Fake'])
    
    save_path = PLOTS_DIR / f'tsne_{paradigm_name}_{dataset_name}_{split}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")

def extract_and_save(dataset_name, split, extractors, device, batch_size=32):
    dataset_path = PROCESSED_DIR / dataset_name / split
    if not dataset_path.exists():
        print(f"Path does not exist: {dataset_path}, skipping...")
        return
        
    # Standard transform for ImageNet pretrained models
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DeepfakeImageDataset(dataset_path, transform=transform)
    if len(dataset) == 0:
        print(f"No images found in {dataset_path}, skipping...")
        return
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    features_dict = {
        "cnn": [], "vit": [], "spectral": []
    }
    all_labels = []
    
    # Timing variables
    inference_times = {"cnn": [], "vit": [], "spectral": []}

    print(f"Extracting features for {dataset_name} ({split}) - {len(dataset)} images")
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=f"{dataset_name} {split}"):
            images = images.to(device)
            labels = labels.to(device)
            
            # CNN
            start = time.time()
            cnn_feat = extractors["cnn"](images)
            inference_times["cnn"].append((time.time() - start) / images.size(0))
            features_dict["cnn"].append(cnn_feat.cpu())
            
            # ViT
            start = time.time()
            vit_feat = extractors["vit"](images)
            inference_times["vit"].append((time.time() - start) / images.size(0))
            features_dict["vit"].append(vit_feat.cpu())
            
            # Spectral
            start = time.time()
            spec_feat = extractors["spectral"](images)
            inference_times["spectral"].append((time.time() - start) / images.size(0))
            features_dict["spectral"].append(spec_feat.cpu())
            
            all_labels.append(labels.cpu())

    # Concatenate and save
    all_labels = torch.cat(all_labels, dim=0)
    labels_np = all_labels.numpy()
    
    print("\nAverage Inference Time per Frame (ms):")
    for name in extractors.keys():
        avg_time = np.mean(inference_times[name]) * 1000
        print(f"- {name.upper()}: {avg_time:.2f} ms")
        
        # Save features
        feats = torch.cat(features_dict[name], dim=0)
        save_path = FEATURES_DIR / f"{dataset_name}_{split}_{name}.pt"
        torch.save({'features': feats, 'labels': all_labels}, save_path)
        print(f"Saved {name} features to {save_path} (shape: {feats.shape})")
        
        # Visualize TSNE
        if feats.shape[0] > 10:
            visualize_tsne(feats.numpy(), labels_np, name.upper(), dataset_name, split)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("Loading feature extractors...")
    extractors = get_extractors(device)
    
    # We have three combinations to process based on our extract_faces script:
    # FF++ train, FF++ val, Celeb-DF test
    
    extract_and_save("FF++", "train", extractors, device)
    extract_and_save("FF++", "val", extractors, device)
    extract_and_save("Celeb-DF", "test", extractors, device)
    
    print("Feature extraction and caching completed successfully!")

if __name__ == "__main__":
    main()
