import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from src.config import config

def load_data():
    if config["data"].get("use_real_dataset", False):
        path = Path(config["paths"]["real_dataset"])
        if not path.exists():
            raise FileNotFoundError(f"Real dataset not found: {path}. Place creditcard.csv in data/raw/")
        df = pd.read_csv(path)
        print(f"Loaded real European CC dataset: {len(df)} samples, fraud ratio {df['Class'].mean():.4%}")
        # Optional subsampling for speed (comment out for full)
        # df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        # print(f"Subsampled to 50,000 for testing")
    else:
        path = Path(config["paths"]["synthetic_output"])
        if not path.exists():
            raise FileNotFoundError(f"Synthetic not found: {path}")
        df = pd.read_parquet(path)
        print(f"Loaded synthetic dataset: {len(df)} samples")
    return df

def manual_stratified_split(df, val_size=0.15, test_size=0.15, seed=42):
    np.random.seed(seed)
    fraud = df[df['Class'] == 1].sample(frac=1, random_state=seed)
    normal = df[df['Class'] == 0].sample(frac=1, random_state=seed)
    
    n_fraud = len(fraud)
    n_normal = len(normal)
    
    # Fraud splits
    n_val_f = int(n_fraud * val_size)
    n_test_f = int(n_fraud * test_size)
    n_train_f = n_fraud - n_val_f - n_test_f
    
    fraud_train = fraud.iloc[:n_train_f]
    fraud_val = fraud.iloc[n_train_f:n_train_f + n_val_f]
    fraud_test = fraud.iloc[n_train_f + n_val_f:]
    
    # Normal splits
    n_val_n = int(n_normal * val_size)
    n_test_n = int(n_normal * test_size)
    n_train_n = n_normal - n_val_n - n_test_n
    
    normal_train = normal.iloc[:n_train_n]
    normal_val = normal.iloc[n_train_n:n_train_n + n_val_n]
    normal_test = normal.iloc[n_train_n + n_val_n:]
    
    train_df = pd.concat([normal_train, fraud_train]).sample(frac=1, random_state=seed)
    val_df = pd.concat([normal_val, fraud_val]).sample(frac=1, random_state=seed)
    test_df = pd.concat([normal_test, fraud_test]).sample(frac=1, random_state=seed)
    
    return train_df, val_df, test_df

def normalize_features(train_df, val_df, test_df):
    feature_cols = [c for c in train_df.columns if c not in ['Amount', 'Class']]
    
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()
    
    for df in [train_df, val_df, test_df]:
        df[feature_cols] = (df[feature_cols] - mean) / std
    
    return train_df, val_df, test_df, feature_cols

def create_loaders(train_df, val_df, test_df, feature_cols, batch_size=256):
    def df_to_tensors(df):
        feats = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        labels = torch.tensor(df['Class'].values, dtype=torch.long)
        amounts = torch.tensor(df['Amount'].values, dtype=torch.float32)
        return feats, labels, amounts
    
    train_feats, train_labels, train_amounts = df_to_tensors(train_df)
    val_feats, val_labels, val_amounts = df_to_tensors(val_df)
    test_feats, test_labels, test_amounts = df_to_tensors(test_df)
    
    train_dataset = TensorDataset(train_feats, train_labels, train_amounts)
    val_dataset = TensorDataset(val_feats, val_labels, val_amounts)
    test_dataset = TensorDataset(test_feats, test_labels, test_amounts)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def prepare_data_loaders():
    """Main function to prepare and return all DataLoaders"""
    df = load_data()
    
    train_df, val_df, test_df = manual_stratified_split(df)
    print(f"Splits: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
    print("Fraud ratios preserved:")
    print(f"Train: {train_df['Class'].mean():.4%}, Val: {val_df['Class'].mean():.4%}, Test: {test_df['Class'].mean():.4%}")
    
    train_df, val_df, test_df, feature_cols = normalize_features(train_df, val_df, test_df)
    print(f"Normalized features: {len(feature_cols)} cols (Amount excluded)")
    
    train_loader, val_loader, test_loader = create_loaders(train_df, val_df, test_df, feature_cols)
    print("DataLoaders ready:")
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # For direct testing only
    prepare_data_loaders()