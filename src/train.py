import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path so we can import from models
sys.path.append(str(Path(__file__).parent.parent))
from src.models.mlp_classifier import MLPClassifier

BASE_DIR = Path(__file__).parent.parent
FEATURES_DIR = BASE_DIR / "data/features"
MODELS_DIR = BASE_DIR / "results/models"
PLOTS_DIR = BASE_DIR / "results/plots"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(paradigm_name):
    """Load train and val features for a given paradigm (cnn, vit, spectral)"""
    train_path = FEATURES_DIR / f"FF++_train_{paradigm_name}.pt"
    val_path = FEATURES_DIR / f"FF++_val_{paradigm_name}.pt"
    
    if not train_path.exists() or not val_path.exists():
        print(f"Features for {paradigm_name} not found. Please run feature extraction first.")
        return None, None
        
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    
    # Flatten spatial dimensions if needed (e.g., if CNN returns [B, C, 1, 1])
    train_features = train_data['features'].view(train_data['features'].size(0), -1)
    train_labels = train_data['labels'].float()
    
    val_features = val_data['features'].view(val_data['features'].size(0), -1)
    val_labels = val_data['labels'].float()
    
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    return train_dataset, val_dataset

def plot_learning_curves(history, paradigm_name):
    """Plot and save learning curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title(f'{paradigm_name.upper()} - Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title(f'{paradigm_name.upper()} - Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = PLOTS_DIR / f'learning_curves_{paradigm_name}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved learning curves to {save_path}")

def train_model(paradigm_name, num_epochs=50, batch_size=64, lr=1e-4):
    print(f"\n{'='*50}\nTraining MLP for {paradigm_name.upper()}\n{'='*50}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset, val_dataset = load_data(paradigm_name)
    if train_dataset is None:
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Determine input dimension from data
    input_dim = train_dataset.tensors[0].shape[1]
    print(f"Input feature dimension: {input_dim}")
    
    model = MLPClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_path = MODELS_DIR / f"mlp_{paradigm_name}_best.pth"
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_features.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_labels.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * batch_features.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
                  
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            
    print(f"Training completed. Best model saved to {best_model_path}")
    plot_learning_curves(history, paradigm_name)
    
def main():
    paradigms = ['cnn', 'vit', 'spectral']
    for p in paradigms:
        # 50 epochs should be enough as MLP is simple and features are high-level
        train_model(p, num_epochs=50, batch_size=64, lr=1e-4)

if __name__ == "__main__":
    main()
