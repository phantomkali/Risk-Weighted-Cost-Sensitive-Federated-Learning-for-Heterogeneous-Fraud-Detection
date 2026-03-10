import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from src.preprocessing import prepare_data_loaders
from src.model import train_centralized, FraudMLP
from src.utils import threshold_sweep, plot_threshold_curves
from src.config import config
from pathlib import Path
from src.config import config
is_real = config["data"].get("use_real_dataset", False)
suffix = "_real" if is_real else "_synthetic"

def print_metrics(sweep_results, t_name, t_val):
    idx = (sweep_results["threshold"] - t_val).abs().argmin()
    row = sweep_results.iloc[idx]
    print(f"{t_name}:")
    print(f"  Precision: {row['precision']*100:.2f}%")
    print(f"  Recall: {row['recall']*100:.2f}%")
    print(f"  F1: {row['f1']*100:.2f}%")
    print(f"  Financial Loss: {row['loss']:.2f}")
    print(f"  % Fraud Amount Recovered: {row['pct_recovered']:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    train_loader, val_loader, test_loader = prepare_data_loaders()
    
    model_path = "results/models/centralized_best.pth"
    if Path(model_path).exists():
        print(f"Loading existing model from {model_path}")
        model = FraudMLP().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = train_centralized(train_loader, val_loader, device=device)
    
    print("\n=== Threshold Sweep on Test Set ===")
    sweep_results = threshold_sweep(model, test_loader, device)
    
    # Fixed 0.5
    print_metrics(sweep_results, "Fixed threshold ~0.5", 0.5)
    
    # Cost-optimal
    best_idx = sweep_results["loss"].argmin()
    best = sweep_results.iloc[best_idx]
    print_metrics(sweep_results, f"Cost-optimal threshold {best['threshold']:.3f}", best['threshold'])
    
    #plot_threshold_curves(sweep_results, save_path="results/figures/threshold_analysis_synthetic.png" if not config["data"].get("use_real_dataset", False) else "results/figures/threshold_analysis_real.png")
    plot_threshold_curves(sweep_results, save_path=f"results/figures/threshold_analysis{suffix}.png")
    print("\nStages 4–6 complete — economic + classification metrics generated")