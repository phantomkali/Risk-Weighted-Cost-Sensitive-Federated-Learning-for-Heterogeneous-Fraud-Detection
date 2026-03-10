import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from src.config import config

def sigmoid(x):
    """Convert logits to probabilities"""
    return 1 / (1 + np.exp(-x))

def compute_financial_loss(probs, labels, amounts, threshold, c_fp=config["costs"]["c_fp"]):
    """
    Compute exact financial loss and % fraud amount recovered.
    """
    preds = (probs >= threshold).astype(int)
    fn_mask = (preds == 0) & (labels == 1)
    fp_mask = (preds == 1) & (labels == 0)
    
    loss_fn = amounts[fn_mask].sum() if fn_mask.any() else 0.0
    loss_fp = c_fp * fp_mask.sum()
    
    total_loss = loss_fn + loss_fp
    
    fraud_detected_amount = amounts[(preds == 1) & (labels == 1)].sum()
    total_fraud_amount = amounts[labels == 1].sum()
    percent_recovered = 100 * fraud_detected_amount / total_fraud_amount if total_fraud_amount > 0 else 0.0
    
    return total_loss, percent_recovered

def find_optimal_threshold_client(probs, labels, amounts, client_id, thresholds=np.linspace(0, 1, 101)):
    """
    Find optimal threshold for a specific client using their FP cost.
    
    Args:
        probs: prediction probabilities
        labels: ground truth labels
        amounts: transaction amounts
        client_id: client identifier (int)
        thresholds: threshold candidates to sweep
    
    Returns:
        best_threshold: optimal threshold for this client
        min_loss: minimum loss achieved
        fp_rate: false positive rate at optimal threshold
        fn_amount: false negative amount at optimal threshold
    """
    c_fp_dict = config["costs"].get("c_fp_per_client", {})
    c_fp = c_fp_dict.get(client_id, config["costs"]["c_fp"])  # fallback to global
    
    best_loss = np.inf
    best_threshold = 0.5
    best_fp_rate = 0.0
    best_fn_amount = 0.0
    
    total_legit = (labels == 0).sum()
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        # False negatives (missed fraud)
        fn_mask = (preds == 0) & (labels == 1)
        fn_loss = amounts[fn_mask].sum() if fn_mask.any() else 0.0
        
        # False positives (friction)
        fp_mask = (preds == 1) & (labels == 0)
        fp_count = fp_mask.sum()
        fp_loss = c_fp * fp_count
        
        total_loss = fn_loss + fp_loss
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_threshold = t
            best_fp_rate = fp_count / total_legit if total_legit > 0 else 0.0
            best_fn_amount = fn_loss
    
    return best_threshold, best_loss, best_fp_rate, best_fn_amount

def compute_classification_metrics(probs, labels, threshold):
    """
    Compute precision, recall, F1 at given threshold.
    Handles zero-division safely.
    """
    preds = (probs >= threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)
    return precision, recall, f1

def threshold_sweep(model, loader, device, thresholds=np.linspace(0, 1, 201)):
    """
    Full sweep: computes financial loss, % recovered, recall, precision, F1 for every threshold.
    Returns pandas DataFrame with all metrics.
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_amounts = []
    
    with torch.no_grad():
        for feats, labels, amounts in loader:
            feats = feats.to(device)
            logits = model(feats).cpu().numpy()
            probs = sigmoid(logits)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_amounts.extend(amounts.numpy())
    
    all_probs   = np.array(all_probs)
    all_labels  = np.array(all_labels)
    all_amounts = np.array(all_amounts)
    
    results = []
    for t in thresholds:
        loss, pct_rec = compute_financial_loss(all_probs, all_labels, all_amounts, t)
        recall    = ((all_probs >= t) & (all_labels == 1)).sum() / all_labels.sum() if all_labels.sum() > 0 else 0.0
        prec, rec, f1 = compute_classification_metrics(all_probs, all_labels, t)
        
        results.append({
            "threshold": t,
            "loss": loss,
            "pct_recovered": pct_rec,
            "recall": recall,
            "precision": prec,
            "f1": f1
        })
    
    df = pd.DataFrame(results)
    return df

def plot_threshold_curves(sweep_results, save_path="results/figures/threshold_analysis.png"):
    """
    Plot financial loss (primary), % recovered, recall, precision, F1.
    Non-blocking save — view manually.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Financial Loss (left y-axis)
    ax1.plot(sweep_results["threshold"], sweep_results["loss"], 'b-', linewidth=2, label='Financial Loss')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Financial Loss', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Right y-axis: multiple metrics
    ax2 = ax1.twinx()
    ax2.plot(sweep_results["threshold"], sweep_results["pct_recovered"], 'g--', linewidth=2, label='% Fraud Amount Recovered')
    ax2.plot(sweep_results["threshold"], sweep_results["recall"] * 100, 'r:', linewidth=2, label='Recall (%)')
    ax2.plot(sweep_results["threshold"], sweep_results["precision"] * 100, 'm-.', linewidth=1.5, label='Precision (%)')
    ax2.plot(sweep_results["threshold"], sweep_results["f1"] * 100, 'c-', linewidth=1.5, label='F1 (%)')
    
    ax2.set_ylabel('% Metrics', color='k', fontsize=12)
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.title('Threshold vs Financial Loss & Classification Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()  # Commented — non-blocking
    print(f"Plot saved to: {save_path} — open manually to view")