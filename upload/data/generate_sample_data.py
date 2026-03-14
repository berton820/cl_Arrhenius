#!/usr/bin/env python3
"""
Generate a small sample dataset (~100 rows) for TC4 hot deformation.
Output: demo_tc4_100.csv in the same directory.
Columns: Temperature, StrainRate, TrueStrain, FlowStress (T_K computed by loader).
"""

import pandas as pd
import numpy as np
import os

SEED = 42
rng = np.random.default_rng(SEED)

# Grid: 5 temps × 4 strain rates × 5 strain points = 100 rows
temps = [800, 850, 900, 950, 1010]   # 1010 for temperature extrapolation test
rates = [0.001, 0.01, 0.1, 1.0]      # add 10 in real data for rate extrapolation; here we keep 4 for balance
strains = np.linspace(0.08, 0.55, 5).round(4).tolist()

# Zener-Hollomon-type: stress increases with strain rate and strain, decreases with T
def flow_stress(temp_C, strain_rate, true_strain):
    T_K = temp_C + 273.15
    Q = 3.5e5
    R = 8.314
    Z = strain_rate * np.exp(Q / (R * T_K))
    # Power-law: sigma ~ Z^m with strain hardening; scale so 50–350 MPa
    m = 0.18
    base = 45 * (Z ** m) * (1 + 0.6 * true_strain)
    return float(np.clip(base, 50.0, 350.0))

rows = []
for temp in temps:
    for sr in rates:
        for eps in strains:
            sigma = flow_stress(temp, sr, eps)
            sigma = sigma + rng.uniform(-2.5, 2.5)
            sigma = max(10.0, min(400.0, sigma))
            rows.append({
                "Temperature": temp,
                "StrainRate": sr,
                "TrueStrain": round(eps, 4),
                "FlowStress": round(sigma, 2),
            })

df = pd.DataFrame(rows)
out_path = os.path.join(os.path.dirname(__file__), "demo_tc4_100.csv")
df.to_csv(out_path, index=False)
print(f"Written {len(df)} rows to {out_path}")
