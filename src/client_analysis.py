"""
Client-wise threshold and performance analysis.
Run this after training to generate detailed client comparison tables.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocessing import manual_stratified_split, normalize_features, create_loaders
from src.model import FraudMLP
from src.utils import sigmoid, find_optimal_threshold_client
from src.config import config


def analyze_client_thresholds(model, client_loaders, client_ids, device, model_name="Model"):
    """
    Analyze optimal thresholds for each client using their specific FP costs.
    
    Args:
        model: trained global model
        client_loaders: list of data loaders for each client
        client_ids: list of client identifiers
        device: computation device
        model_name: name for logging
    
    Returns:
        DataFrame with client-wise analysis
    """
    model.eval()
    results = []
    
    c_fp_dict = config["costs"].get("c_fp_per_client", {})
    
    for client_id, loader in zip(client_ids, client_loaders):
        # Collect predictions
        probs = []
        labels = []
        amounts = []
        
        with torch.no_grad():
            for feats, lbs, amts in loader:
                feats = feats.to(device)
                logits = model(feats).cpu().numpy()
                probs.extend(sigmoid(logits))
                labels.extend(lbs.numpy())
                amounts.extend(amts.numpy())
        
        probs = np.array(probs)
        labels = np.array(labels)
        amounts = np.array(amounts)
        
        # Find optimal threshold using client-specific cost
        best_threshold, min_loss, fp_rate, fn_amount = find_optimal_threshold_client(
            probs, labels, amounts, client_id
        )
        
        # Compute metrics at optimal threshold
        preds = (probs >= best_threshold).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        total_legit = (labels == 0).sum()
        total_fraud = (labels == 1).sum()
        total_fraud_amount = amounts[labels == 1].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Get client-specific cost
        c_fp = c_fp_dict.get(client_id, config["costs"]["c_fp"])
        
        results.append({
            "client_id": client_id,
            "c_fp": c_fp,
            "n_samples": len(labels),
            "fraud_rate": labels.mean(),
            "optimal_threshold": best_threshold,
            "min_loss": min_loss,
            "fp_rate": fp_rate * 100,  # as percentage
            "fn_amount": fn_amount,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })
    
    df = pd.DataFrame(results)
    return df


def create_comparison_table(fedavg_model, riskweighted_model, client_loaders, client_ids, device):
    """
    Create side-by-side comparison of FedAvg vs Risk-Weighted performance.
    
    Args:
        fedavg_model: FedAvg trained model
        riskweighted_model: Risk-weighted trained model
        client_loaders: list of data loaders
        client_ids: list of client IDs
        device: computation device
    
    Returns:
        DataFrame with comparison
    """
    fedavg_results = analyze_client_thresholds(fedavg_model, client_loaders, client_ids, device, "FedAvg")
    risk_results = analyze_client_thresholds(riskweighted_model, client_loaders, client_ids, device, "Risk-Weighted")
    
    # Merge for comparison
    comparison = pd.DataFrame({
        "Client": fedavg_results["client_id"],
        "C_FP": fedavg_results["c_fp"],
        "Samples": fedavg_results["n_samples"],
        "Fraud%": fedavg_results["fraud_rate"] * 100,
        
        # FedAvg metrics
        "FedAvg_Threshold": fedavg_results["optimal_threshold"],
        "FedAvg_Loss": fedavg_results["min_loss"],
        "FedAvg_FP_Rate": fedavg_results["fp_rate"],
        
        # Risk-Weighted metrics
        "RiskW_Threshold": risk_results["optimal_threshold"],
        "RiskW_Loss": risk_results["min_loss"],
        "RiskW_FP_Rate": risk_results["fp_rate"],
        
        # Improvements
        "Loss_Reduction": fedavg_results["min_loss"] - risk_results["min_loss"],
        "Loss_Reduction_%": ((fedavg_results["min_loss"] - risk_results["min_loss"]) / fedavg_results["min_loss"] * 100)
    })
    
    return comparison


if __name__ == "__main__":
    from src.risk_weighted_fl import create_non_iid_clients
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Client Analysis on {device}\n")
    
    # Load data
    is_real = config["data"].get("use_real_dataset", False)
    suffix = "_real" if is_real else "_synthetic"
    
    df = pd.read_parquet(config["paths"]["synthetic_output"])
    train_df, _, _ = manual_stratified_split(df)
    
    # Create client partitions (same as training)
    client_dfs = create_non_iid_clients(
        train_df, 
        n_clients=config["fl"]["n_clients"], 
        skew=config["fl"]["non_iid_skew"]
    )
    
    # Normalize and create loaders
    train_norm, _, _, feature_cols = normalize_features(train_df.copy(), train_df.copy(), train_df.copy())
    
    client_loaders = []
    for cdf in client_dfs:
        cdf_norm = cdf.copy()
        cdf_norm[feature_cols] = (cdf[feature_cols] - train_norm[feature_cols].mean()) / train_norm[feature_cols].std()
        loader, _, _ = create_loaders(cdf_norm, cdf_norm, cdf_norm, feature_cols, batch_size=256)
        client_loaders.append(loader)
    
    client_ids = list(range(len(client_dfs)))
    
    # Load models
    fedavg_path = "results/models/fedavg_global.pth"
    riskw_path = "results/models/risk_weighted_global.pth"
    
    if not Path(fedavg_path).exists() or not Path(riskw_path).exists():
        print("❌ Models not found. Run training first:")
        print(f"   - {fedavg_path}")
        print(f"   - {riskw_path}")
        exit(1)
    
    fedavg_model = FraudMLP().to(device)
    fedavg_model.load_state_dict(torch.load(fedavg_path, map_location=device))
    
    riskw_model = FraudMLP().to(device)
    riskw_model.load_state_dict(torch.load(riskw_path, map_location=device))
    
    print("=" * 80)
    print("CLIENT-WISE THRESHOLD ANALYSIS")
    print("=" * 80)
    
    # Individual analyses
    print("\n📊 FedAvg - Client-Specific Thresholds:")
    fedavg_analysis = analyze_client_thresholds(fedavg_model, client_loaders, client_ids, device, "FedAvg")
    print(fedavg_analysis[["client_id", "c_fp", "optimal_threshold", "min_loss", "fp_rate", "recall"]].to_string(index=False))
    
    print("\n📊 Risk-Weighted - Client-Specific Thresholds:")
    riskw_analysis = analyze_client_thresholds(riskw_model, client_loaders, client_ids, device, "Risk-Weighted")
    print(riskw_analysis[["client_id", "c_fp", "optimal_threshold", "min_loss", "fp_rate", "recall"]].to_string(index=False))
    
    # Comparison table
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    comparison = create_comparison_table(fedavg_model, riskw_model, client_loaders, client_ids, device)
    print(comparison.to_string(index=False))
    
    # Save results
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fedavg_analysis.to_csv(output_dir / f"fedavg_client_analysis{suffix}.csv", index=False)
    riskw_analysis.to_csv(output_dir / f"riskweighted_client_analysis{suffix}.csv", index=False)
    comparison.to_csv(output_dir / f"client_comparison{suffix}.csv", index=False)
    
    print(f"\n✅ Results saved to {output_dir}/")
    print("   - fedavg_client_analysis.csv")
    print("   - riskweighted_client_analysis.csv")
    print("   - client_comparison.csv")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Average Loss Reduction: {comparison['Loss_Reduction'].mean():.2f}")
    print(f"Total Loss Reduction: {comparison['Loss_Reduction'].sum():.2f}")
    print(f"Average % Improvement: {comparison['Loss_Reduction_%'].mean():.2f}%")
    print(f"\nClients with improvement: {(comparison['Loss_Reduction'] > 0).sum()} / {len(comparison)}")
