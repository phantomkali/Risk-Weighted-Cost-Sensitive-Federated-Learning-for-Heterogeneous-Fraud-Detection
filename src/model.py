import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import config

class FraudMLP(nn.Module):
    def __init__(self, input_dim=29):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in config["model"]["hidden_layers"]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(config["model"]["dropout"]))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)  # Return logits

def get_pos_weight(train_loader: DataLoader):
    # Compute class imbalance for BCEWithLogitsLoss
    total = 0
    positives = 0
    for _, labels, _ in train_loader:
        total += labels.size(0)
        positives += labels.sum().item()
    return torch.tensor(total / positives) if positives > 0 else torch.tensor(1.0)

def train_centralized(train_loader, val_loader, device="cpu"):
    model = FraudMLP().to(device)
    cfg = config["model"]
    
    pos_weight = get_pos_weight(train_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
    
    best_val_loss = float('inf')
    best_state = None
    patience = cfg["early_stop_patience"]
    counter = 0
    
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for feats, labels, _ in train_loader:
            feats, labels = feats.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, labels, _ in val_loader:
                feats, labels = feats.to(device), labels.float().to(device)
                logits = model(feats)
                val_loss += criterion(logits, labels).item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    model.load_state_dict(best_state)
    torch.save(best_state, "results/models/centralized_best.pth")
    print("Best model saved to results/models/centralized_best.pth")
    return model