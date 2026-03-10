"""
Quick validation test for client-specific threshold optimization.
Run this to verify the implementation works before full experiments.
"""

import numpy as np
from src.utils import find_optimal_threshold_client
from src.config import config

print("=" * 60)
print("VALIDATION TEST: Client-Specific Threshold Optimization")
print("=" * 60)

# Create synthetic test data
np.random.seed(42)
n_samples = 1000

# Simulate predictions (higher scores for fraud)
probs = np.concatenate([
    np.random.beta(2, 8, 900),  # Legitimate transactions (low scores)
    np.random.beta(8, 2, 100)   # Fraud transactions (high scores)
])

labels = np.concatenate([
    np.zeros(900),  # Legitimate
    np.ones(100)    # Fraud
])

# Transaction amounts (fraud tends to be higher)
amounts = np.concatenate([
    np.random.lognormal(4, 1, 900),    # Legit: ~$88 mean
    np.random.lognormal(5.5, 1, 100)   # Fraud: ~$300 mean
])

# Shuffle
shuffle_idx = np.random.permutation(n_samples)
probs = probs[shuffle_idx]
labels = labels[shuffle_idx]
amounts = amounts[shuffle_idx]

print(f"\n📊 Test Dataset:")
print(f"  - Total samples: {n_samples}")
print(f"  - Fraud rate: {labels.mean():.2%}")
print(f"  - Mean transaction: ${amounts.mean():.2f}")
print(f"  - Total fraud amount: ${amounts[labels == 1].sum():.2f}")

# Test different client types
print("\n" + "=" * 60)
print("TESTING CLIENT-SPECIFIC THRESHOLDS")
print("=" * 60)

c_fp_dict = config["costs"].get("c_fp_per_client", {})

for client_id in [0, 2, 5]:
    print(f"\n🏦 Client {client_id}:")
    c_fp = c_fp_dict.get(client_id, config["costs"]["c_fp"])
    print(f"   C_fp: {c_fp}")
    
    best_t, min_loss, fp_rate, fn_amount = find_optimal_threshold_client(
        probs, labels, amounts, client_id
    )
    
    print(f"   ✓ Optimal threshold: {best_t:.3f}")
    print(f"   ✓ Minimum loss: ${min_loss:,.2f}")
    print(f"   ✓ FP rate: {fp_rate*100:.2f}%")
    print(f"   ✓ FN amount: ${fn_amount:,.2f}")
    
    # Verify threshold behavior
    if client_id == 0:  # Premium
        expected = "HIGH (> 0.5)"
        check = "✅" if best_t > 0.5 else "⚠️"
    elif client_id == 5:  # Small
        expected = "LOW (< 0.5)"
        check = "✅" if best_t < 0.5 else "⚠️"
    else:  # Mid-tier
        expected = "MEDIUM (~0.4-0.6)"
        check = "✅" if 0.35 < best_t < 0.65 else "⚠️"
    
    print(f"   Expected: {expected} {check}")

# Summary
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

results = []
for cid in range(8):
    t, loss, fpr, fna = find_optimal_threshold_client(probs, labels, amounts, cid)
    results.append((cid, c_fp_dict.get(cid, 50), t, loss))

print("\n| Client | C_FP | Threshold | Min Loss   |")
print("|--------|------|-----------|------------|")
for cid, cfp, t, loss in results:
    print(f"| {cid:6d} | {cfp:4d} | {t:9.3f} | ${loss:9,.2f} |")

# Check threshold ordering
premium_thresholds = [results[0][2], results[1][2]]
small_thresholds = [results[5][2], results[6][2], results[7][2]]

avg_premium = np.mean(premium_thresholds)
avg_small = np.mean(small_thresholds)

print("\n" + "=" * 60)
print("CORRECTNESS CHECKS")
print("=" * 60)

check1 = "✅" if avg_premium > avg_small else "❌"
print(f"{check1} Premium thresholds > Small thresholds")
print(f"   Premium avg: {avg_premium:.3f}")
print(f"   Small avg: {avg_small:.3f}")

check2 = "✅" if all(0 <= t <= 1 for _, _, t, _ in results) else "❌"
print(f"{check2} All thresholds in valid range [0, 1]")

check3 = "✅" if all(loss >= 0 for _, _, _, loss in results) else "❌"
print(f"{check3} All losses non-negative")

# Check different C_fp values produce different thresholds
unique_thresholds = len(set(round(t, 2) for _, _, t, _ in results))
check4 = "✅" if unique_thresholds > 1 else "❌"
print(f"{check4} Multiple unique thresholds ({unique_thresholds} distinct values)")

print("\n" + "=" * 60)
if all([check1 == "✅", check2 == "✅", check3 == "✅", check4 == "✅"]):
    print("✅ ALL CHECKS PASSED - Implementation is correct!")
    print("\nYou can now run:")
    print("  python -m src.risk_weighted_fl")
    print("  python -m src.client_analysis")
else:
    print("⚠️ SOME CHECKS FAILED - Review implementation")
print("=" * 60)
