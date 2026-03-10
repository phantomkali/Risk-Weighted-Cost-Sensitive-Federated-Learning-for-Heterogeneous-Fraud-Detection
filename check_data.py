import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/synthetic.parquet')
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

print('=== Synthetic Data Feature Analysis ===')
print(f'Total samples: {len(df)}, Fraud: {len(fraud)}, Normal: {len(normal)}')
print(f'Fraud ratio: {len(fraud)/len(df):.4%}')

# Check V1-V3 separation
for col in ['V1', 'V2', 'V3']:
    n_mean, n_std = normal[col].mean(), normal[col].std()
    f_mean, f_std = fraud[col].mean(), fraud[col].std()
    separation = abs(f_mean - n_mean) / (n_std + 0.001)
    print(f'{col}: Normal={n_mean:.2f}+/-{n_std:.2f}, Fraud={f_mean:.2f}+/-{f_std:.2f}, Sep={separation:.2f}')

# Overall separation
normal_means = normal[[f'V{i}' for i in range(1, 29)]].mean()
fraud_means = fraud[[f'V{i}' for i in range(1, 29)]].mean()
normal_stds = normal[[f'V{i}' for i in range(1, 29)]].std()

diff = ((fraud_means - normal_means).abs() / (normal_stds + 0.001)).mean()
print(f'\nAverage normalized separation (z-score): {diff:.2f}')
