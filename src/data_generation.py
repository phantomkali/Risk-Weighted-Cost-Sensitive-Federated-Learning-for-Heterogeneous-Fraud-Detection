import numpy as np
import pandas as pd
from pathlib import Path
from src.config import config


def generate_synthetic_data():
    cfg = config["data"]
    amt = config["amount"]
    
    np.random.seed(cfg["seed"])
    
    n_samples = cfg["n_samples"]
    fraud_ratio = cfg["fraud_ratio"]
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # V1–V28: PCA-like Gaussian features
        # V1–V28: Normal centered at 0
    features_normal = np.random.normal(0, 1, (n_normal, 28))
    
    # Fraud: Strong shift (mean 1.5) + higher variance — clear anomaly
    features_fraud = np.random.normal(1.5, 2.0, (n_fraud, 28))
    
    features = np.vstack([features_normal, features_fraud])
    
    # Time: uniform over 2 days (172800 seconds)
    time = np.random.uniform(0, 172800, n_samples)
    
    # Amount: exponential with fraud skew
    amount_normal = np.random.exponential(scale=amt["normal_scale"], size=n_normal)
    amount_fraud = np.random.exponential(scale=amt["fraud_scale"], size=n_fraud) + amt["fraud_base"]
    amount = np.concatenate([amount_normal, amount_fraud])
    
    # Labels
    labels = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])
    
    # Shuffle everything together
    indices = np.random.permutation(n_samples)
    features = features[indices]
    time = time[indices]
    amount = amount[indices]
    labels = labels[indices]
    
    # Build DataFrame (exact schema for real dataset compatibility)
    df = pd.DataFrame(features, columns=[f'V{i}' for i in range(1, 29)])
    df['Time'] = time
    df['Amount'] = amount
    df['Class'] = labels
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    
    # Exploration & validation
    print("=== Synthetic Dataset Generated ===")
    print(f"Total samples: {len(df)}")
    print(f"Fraud ratio: {df['Class'].mean():.4%}")
    print("\nAmount statistics by class:")
    print(df.groupby('Class')['Amount'].describe())
    
    total_fraud_amount = df[df['Class'] == 1]['Amount'].sum()
    print(f"\nTotal exposed fraud amount: {total_fraud_amount:,.2f}")
    
    # Save
    output_path = Path(config["paths"]["synthetic_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Quick plots (requires matplotlib/seaborn)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x='Amount', hue='Class', bins=50, log_scale=True)
        plt.title('Amount Distribution (Log Scale)')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='Class', y='Amount')
        plt.title('Amount by Class')
        
        plt.tight_layout()
        plt.savefig("results/figures/stage1_amount_distribution.png")
        #plt.show()
        print("Plots saved to results/figures/")
    except Exception as e:
        print(f"Plotting skipped (optional): {e}")