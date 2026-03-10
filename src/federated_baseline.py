import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocessing import prepare_data_loaders, manual_stratified_split, normalize_features, create_loaders
from src.model import FraudMLP
from src.utils import threshold_sweep, plot_threshold_curves, find_optimal_threshold_client, sigmoid
from src.config import config
from src.config import config
is_real = config["data"].get("use_real_dataset", False)
suffix = "_real" if is_real else "_synthetic"

def create_non_iid_clients(train_df, n_clients=8, skew='strong'):
    """
    Create non-IID client splits with economic heterogeneity.
    
    Key design: Every client has non-zero fraud (min 0.1% or at least 1 case),
    but high-value clients have much higher fraud AMOUNTS (economic exposure).
    This creates meaningful threshold differentiation across clients.
    """
    np.random.seed(42)
    
    # Separate fraud and normal transactions
    fraud_df = train_df[train_df['Class'] == 1].copy()
    normal_df = train_df[train_df['Class'] == 0].copy()
    
    # Sort fraud by amount (high-value fraud goes to premium clients)
    fraud_df = fraud_df.sort_values('Amount', ascending=False).reset_index(drop=True)
    normal_df = normal_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle normal
    
    n_fraud = len(fraud_df)
    n_normal = len(normal_df)
    
    # Define client profiles (economic heterogeneity)
    # Premium clients (0, 1): High C_fp, get high-value fraud
    # Standard clients (2, 3, 4): Medium C_fp, get medium fraud  
    # Small clients (5, 6, 7): Low C_fp, get lower fraud
    
    # STEP 1: Guarantee minimum fraud per client (at least 2 cases each)
    min_fraud_per_client = 2
    total_minimum = min_fraud_per_client * n_clients
    
    if n_fraud < total_minimum:
        raise ValueError(f"Not enough fraud cases ({n_fraud}) to give {min_fraud_per_client} per client")
    
    # Start with minimum allocation for everyone
    fraud_allocations = [min_fraud_per_client] * n_clients
    remaining_fraud = n_fraud - total_minimum
    
    # STEP 2: Distribute remaining fraud by economic weight (premium clients get more)
    if skew == 'strong':
        # Relative weights for remaining fraud (premium clients get more high-value fraud)
        extra_weights = [0.35, 0.25, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02]
    else:
        # Mild skew: more uniform
        extra_weights = [0.15, 0.14, 0.13, 0.13, 0.12, 0.12, 0.11, 0.10]
    
    # Normalize weights
    weight_sum = sum(extra_weights)
    extra_weights = [w / weight_sum for w in extra_weights]
    
    # Allocate remaining fraud proportionally
    for i, weight in enumerate(extra_weights):
        extra = int(remaining_fraud * weight)
        fraud_allocations[i] += extra
    
    # Distribute any leftover due to rounding (give to premium clients)
    allocated_so_far = sum(fraud_allocations)
    leftover = n_fraud - allocated_so_far
    for i in range(leftover):
        fraud_allocations[i % n_clients] += 1
    
    # Normal transaction allocation (roughly equal with slight variation)
    normal_per_client = n_normal // n_clients
    
    # Build client dataframes
    client_dfs = []
    fraud_idx = 0
    normal_idx = 0
    
    for i in range(n_clients):
        # Get fraud allocation for this client (high-value fraud goes to low-index clients)
        n_client_fraud = fraud_allocations[i]
        client_fraud = fraud_df.iloc[fraud_idx:fraud_idx + n_client_fraud]
        fraud_idx += n_client_fraud
        
        # Get normal transactions
        n_client_normal = normal_per_client if i < n_clients - 1 else (n_normal - normal_idx)
        client_normal = normal_df.iloc[normal_idx:normal_idx + n_client_normal]
        normal_idx += n_client_normal
        
        # Combine and shuffle
        client_df = pd.concat([client_fraud, client_normal]).sample(frac=1, random_state=42+i)
        client_dfs.append(client_df.reset_index(drop=True))
    
    # Print summary
    print(f"\nCreated {n_clients} non-IID clients (skew: {skew})")
    print("-" * 80)
    total_fraud_amt = 0
    for i, cdf in enumerate(client_dfs):
        fraud_rate = cdf['Class'].mean()
        mean_amt = cdf['Amount'].mean()
        fraud_amt = cdf[cdf['Class'] == 1]['Amount'].sum()
        n_fraud_client = cdf['Class'].sum()
        total_fraud_amt += fraud_amt
        print(f"Client {i}: {len(cdf):5d} samples | {n_fraud_client:3.0f} fraud | "
              f"Rate {fraud_rate:.4%} | Mean Amt {mean_amt:8.2f} | Exposed ${fraud_amt:12,.2f}")
    print("-" * 80)
    print(f"Total exposed fraud amount: ${total_fraud_amt:,.2f}")
    
    return client_dfs

def compute_pos_weight(loader):
    """Compute class imbalance weight from data loader."""
    total = 0
    positives = 0
    for _, labels, _ in loader:
        total += labels.size(0)
        positives += labels.sum().item()
    if positives > 0:
        return total / positives
    return 1.0


def local_train(model, loader, epochs, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["model"]["lr"])
    
    # Compute pos_weight dynamically from this client's data
    pw = compute_pos_weight(loader)
    pos_weight = torch.tensor(min(pw, 100.0)).to(device)  # Cap at 100 for stability
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for _ in range(epochs):
        for feats, labels, _ in loader:
            feats, labels = feats.to(device), labels.float().to(device)
            optimizer.zero_grad()
            logits = model(feats).squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    return model

def fedavg_aggregate(models):
    global_dict = models[0].state_dict()
    for key in global_dict:
        global_dict[key] = torch.stack([m.state_dict()[key].float() for m in models]).mean(0)
    return global_dict

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Federated baseline on {device}")
    
    # Global test loader
    _, _, test_loader = prepare_data_loaders()
    
    # Load full data for partitioning
    df = pd.read_parquet(config["paths"]["synthetic_output"])
    train_df, _, _ = manual_stratified_split(df)
    
    client_dfs = create_non_iid_clients(train_df, n_clients=config["fl"]["n_clients"], skew=config["fl"]["non_iid_skew"])
    
    # Normalize consistently (fit on full train)
    train_norm, _, _, feature_cols = normalize_features(train_df.copy(), train_df.copy(), train_df.copy())
    
    # Create local train loaders
    client_loaders = []
    for cdf in client_dfs:
        cdf_norm = cdf.copy()
        cdf_norm[feature_cols] = (cdf[feature_cols] - train_norm[feature_cols].mean()) / train_norm[feature_cols].std()
        train_loader_local, _, _ = create_loaders(cdf_norm, cdf_norm, cdf_norm, feature_cols, batch_size=256)
        client_loaders.append(train_loader_local)  # Fixed: direct append
    
    # Global model - warm start from centralized if available
    global_model = FraudMLP().to(device)
    centralized_path = Path("results/models/centralized_best.pth")
    if centralized_path.exists():
        global_model.load_state_dict(torch.load(centralized_path, map_location=device))
        print("✅ Warm-start: Initialized global model from centralized pre-training")
    else:
        print("⚠️ No centralized model found - starting from scratch")
    
    rounds = config["fl"]["rounds"]
    local_epochs = config["fl"]["local_epochs"]
    
    for r in range(1, rounds + 1):
        local_models = []
        for loader in client_loaders:
            local_model = FraudMLP().to(device)
            local_model.load_state_dict(global_model.state_dict())
            local_model = local_train(local_model, loader, local_epochs, device)
            local_models.append(local_model)
        
        global_dict = fedavg_aggregate(local_models)
        global_model.load_state_dict(global_dict)
        print(f"Round {r}/{rounds} completed")
    
    torch.save(global_model.state_dict(), "results/models/fedavg_global.pth")
    print("FedAvg global model saved")
    
    # Final evaluation
    print("\n=== FedAvg Global Model on Test Set ===")
    sweep_results = threshold_sweep(global_model, test_loader, device)
    
    idx_05 = (sweep_results["threshold"] - 0.5).abs().argmin()
    fixed_05 = sweep_results.iloc[idx_05]
    print(f"Fixed threshold ~0.5: Loss {fixed_05['loss']:.2f}, % Recovered {fixed_05['pct_recovered']:.2f}%")
    
    best = sweep_results.iloc[sweep_results["loss"].argmin()]
    print(f"Cost-optimal threshold {best['threshold']:.3f}: Loss {best['loss']:.2f}, % Recovered {best['pct_recovered']:.2f}%")
    
    #plot_threshold_curves(sweep_results, save_path="results/figures/fedavg_threshold_analysis.png")
    plot_threshold_curves(sweep_results, save_path=f"results/figures/fedavg_threshold_analysis{suffix}.png")
    print("Stage 7 complete — FedAvg baseline ready for comparison")