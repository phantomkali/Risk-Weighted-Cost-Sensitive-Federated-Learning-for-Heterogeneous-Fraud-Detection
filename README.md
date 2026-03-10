```md
# Risk-Weighted Cost-Sensitive Federated Learning for Fraud Detection

A federated learning framework that incorporates **client-specific economic costs**
into both **model aggregation** and **decision threshold optimization** for fraud detection.

This approach addresses the **heterogeneous financial constraints** of participating
institutions in collaborative fraud detection.

---

## 📋 Overview

Traditional federated learning methods such as **FedAvg** treat all clients equally during
model aggregation. However, in real-world fraud detection:

- **Different institutions incur different false positive costs**
  (customer friction, operational overhead)
- **Fraud exposure varies** across clients (transaction volume, fraud amount)
- **One-size-fits-all thresholds** are economically suboptimal

This project implements a **Risk-Weighted Cost-Sensitive Federated Learning (RW-CSFL)** framework that:

1. Weights client contributions by **economic risk exposure**
2. Optimizes **client-specific decision thresholds**
3. Demonstrates **substantial financial loss reduction** over standard FedAvg

---

## 🎯 Key Results

| Metric | Improvement |
|------|-------------|
| **Average Loss Reduction** | **64.39%** |
| **Total Loss Reduction** | **$11,406** |
| **Clients Improved** | **8 / 8 (100%)** |

---

## 📊 Per-Client Threshold Differentiation

| Client Type | C<sub>FP</sub> | FedAvg Threshold | Risk-Weighted Threshold |
|------------|---------------|------------------|-------------------------|
| Premium (high friction) | 200 | 0.26–0.27 | 0.17–0.20 |
| Standard | 50 | 0.27 | 0.14–0.17 |
| Small (low friction) | 10 | 0.27 | 0.13–0.20 |

**Interpretation**
- High-friction clients behave more conservatively
- Low-friction clients recover more fraud with controlled FP rates
- FedAvg fails to differentiate economically

---
---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt
````

---

### 2. Generate Synthetic Data

```bash
python -m src.data_generation
```

Generates:

* 50,000 transactions
* 0.2% fraud ratio
* PCA-like features (V1–V28)
* Realistic amount distributions

---

### 3. Train Centralized Baseline

```bash
python -m src.centralized
```

Used for:

* Performance comparison
* Warm-starting federated models

---

### 4. Run Federated Learning

```bash
python -m src.federated_baseline
python -m src.risk_weighted_fl
```

---

### 5. Analyze Results

```bash
python -m src.client_analysis
```

Outputs:

* `fedavg_client_analysis.csv`
* `riskweighted_client_analysis.csv`
* `client_comparison.csv`

---

## ⚙️ Configuration (`config.yaml`)

### Economic Costs

```yaml
costs:
  c_fp: 50
  c_fp_per_client:
    0: 200
    1: 200
    2: 50
    3: 50
    4: 50
    5: 10
    6: 10
    7: 10
```

### Federated Learning

```yaml
fl:
  n_clients: 8
  rounds: 20
  local_epochs: 10
  non_iid_skew: strong
```

---

## 📊 Methodology

### 1. Client-Specific Threshold Optimization

For client *i* with FP cost ( C_{fp}^{(i)} ):

```
τ_i* = argmin_τ [ Σ FN_amount + C_fp(i) × FP_count ]
```

---

### 2. Risk-Weighted Aggregation

Instead of FedAvg:

```
w_i = R_i / Σ R_j
R_i = L_i* / n_i
```

Clients with higher **economic exposure** exert more influence.

---

## 📈 Experimental Results (Synthetic)

| Client | C<sub>FP</sub> | FedAvg Loss | RiskW Loss | Reduction |
| ------ | -------------- | ----------- | ---------- | --------- |
| 0      | 200            | 9,594       | 4,904      | 48.9%     |
| 1      | 200            | 4,485       | 2,101      | 53.2%     |
| 2      | 50             | 2,225       | 558        | 74.9%     |
| 7      | 10             | 461         | 0          | 100%      |

**Average Improvement:** **64.39%**

---

## 🧪 Running Tests

```bash
python test_client_thresholds.py
```

Validates:

* Threshold monotonicity
* Loss decomposition correctness
* Edge cases

---

## 🔬 Using Real Data

1. Download dataset from Kaggle
2. Place `creditcard.csv` in `data/raw/`
3. Set in `config.yaml`:

```yaml
data:
  use_real_dataset: true
```

---

## 📚 Citation

```bibtex
@article{riskweighted_fl_fraud,
  title={Risk-Weighted Cost-Sensitive Federated Learning for Fraud Detection},
  author={Your Name},
  year={2026}
}
```

---

## 📄 License

MIT License

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests
4. Submit a pull request

---



## 🔁 Apply it

1. Replace contents of `README.md`
2. Run:
```bash
git add README.md
git commit -m "Clean and format README"
git push
````




