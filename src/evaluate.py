import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.mlp_classifier import MLPClassifier

BASE_DIR = Path(__file__).parent.parent
FEATURES_DIR = BASE_DIR / "data/features"
MODELS_DIR = BASE_DIR / "results/models"
PLOTS_DIR = BASE_DIR / "results/plots"
RESULTS_DIR = BASE_DIR / "results"

os.makedirs(PLOTS_DIR, exist_ok=True)

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the point where FPR and FNR are equal
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_threshold_idx]
    return eer

def evaluate_model(paradigm_name, dataset_name, split):
    """Evaluate a trained model on a specific dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load features
    features_path = FEATURES_DIR / f"{dataset_name}_{split}_{paradigm_name}.pt"
    if not features_path.exists():
        print(f"Features file {features_path} not found.")
        return None
        
    data = torch.load(features_path)
    features = data['features'].view(data['features'].size(0), -1).to(device)
    labels = data['labels'].float().to(device)
    
    # Load Model
    model_path = MODELS_DIR / f"mlp_{paradigm_name}_best.pth"
    if not model_path.exists():
        print(f"Model file {model_path} not found.")
        return None
        
    input_dim = features.shape[1]
    model = MLPClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        scores = model(features)
        preds = (scores > 0.5).float()
        
    y_true = labels.cpu().numpy()
    y_scores = scores.cpu().numpy()
    y_preds = preds.cpu().numpy()
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_true, y_preds)
    eer = calculate_eer(y_true, y_scores)
    
    # Save False Positives/Negatives info
    fps = np.where((y_true == 0) & (y_preds == 1))[0]
    fns = np.where((y_true == 1) & (y_preds == 0))[0]
    
    return {
        'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 
        'f1': f1, 'eer': eer,
        'fps_indices': fps, 'fns_indices': fns
    }

def plot_roc_curves(results_dict, dataset_name):
    """Plot ROC curves for all paradigms on the same plot"""
    plt.figure(figsize=(10, 8))
    
    colors = {'cnn': 'blue', 'vit': 'red', 'spectral': 'green'}
    
    for paradigm, res in results_dict.items():
        if res is not None:
            plt.plot(res['fpr'], res['tpr'], color=colors[paradigm], lw=2,
                     label=f"{paradigm.upper()} (AUC = {res['auc']:.4f})")
                     
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves on {dataset_name}', fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    
    save_path = PLOTS_DIR / f'roc_curves_{dataset_name}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curves to {save_path}")

def main():
    paradigms = ['cnn', 'vit', 'spectral']
    
    results_ff = {}
    results_celeb = {}
    
    for p in paradigms:
        print(f"Evaluating {p.upper()} on FaceForensics++ (Intra-domain)...")
        results_ff[p] = evaluate_model(p, "FF++", "val")
        
        print(f"Evaluating {p.upper()} on Celeb-DF (Cross-domain)...")
        results_celeb[p] = evaluate_model(p, "Celeb-DF", "test")
        
    # Plot ROC curves
    plot_roc_curves(results_ff, "FaceForensics++")
    plot_roc_curves(results_celeb, "Celeb-DF")
    
    # Generate Summary Table
    print("\nGenerating Summary Table...")
    table_data = []
    
    for p in paradigms:
        row = {'Architecture': p.upper()}
        
        if results_ff[p]:
            row['AUC (FF++)'] = round(results_ff[p]['auc'], 4)
            row['F1 (FF++)'] = round(results_ff[p]['f1'], 4)
            row['EER (FF++)'] = round(results_ff[p]['eer'], 4)
        else:
            row['AUC (FF++)'] = row['F1 (FF++)'] = row['EER (FF++)'] = None
            
        if results_celeb[p]:
            row['AUC (Celeb-DF)'] = round(results_celeb[p]['auc'], 4)
            row['F1 (Celeb-DF)'] = round(results_celeb[p]['f1'], 4)
            row['EER (Celeb-DF)'] = round(results_celeb[p]['eer'], 4)
        else:
            row['AUC (Celeb-DF)'] = row['F1 (Celeb-DF)'] = row['EER (Celeb-DF)'] = None
            
        if results_ff[p] and results_celeb[p]:
            row['AUC Drop (Δ)'] = round(results_ff[p]['auc'] - results_celeb[p]['auc'], 4)
        else:
            row['AUC Drop (Δ)'] = None
            
        table_data.append(row)
        
    df = pd.DataFrame(table_data)
    csv_path = RESULTS_DIR / 'evaluation_summary.csv'
    df.to_csv(csv_path, index=False)
    
    print("\nFinal Results:")
    print(df.to_string(index=False))
    print(f"\nSaved summary table to {csv_path}")

if __name__ == "__main__":
    main()
