"""
Data loading and synthetic demo data generation.
All DataFrames use columns: Temperature, T_K, StrainRate, TrueStrain, FlowStress.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional

# Standard column names for external use
COL_TEMP = "Temperature"
COL_TK = "T_K"
COL_SR = "StrainRate"
COL_STRAIN = "TrueStrain"
COL_STRESS = "FlowStress"


def load_from_excel(path: str) -> pd.DataFrame:
    """
    Load hot deformation data from Excel.
    Each sheet = one temperature (sheet name = temp in °C).
    Row 0: strain rates; then pairs of columns (strain, stress).
    Returns DataFrame with columns Temperature, T_K, StrainRate, TrueStrain, FlowStress.
    """
    xlsx = pd.ExcelFile(path)
    all_data = []
    for sheet_name in xlsx.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        temp = int(sheet_name)
        num_pairs = df.shape[1] // 2
        for i in range(num_pairs):
            strain_rate = float(df.iloc[0, i * 2])
            strain = pd.to_numeric(df.iloc[1:, i * 2], errors="coerce")
            stress = pd.to_numeric(df.iloc[1:, i * 2 + 1], errors="coerce")
            for s_val, r_val in zip(strain, stress):
                if pd.notna(s_val) and pd.notna(r_val):
                    all_data.append({
                        COL_TEMP: temp,
                        COL_TK: temp + 273.15,
                        COL_SR: float(strain_rate),
                        COL_STRAIN: float(s_val),
                        COL_STRESS: float(r_val),
                    })
    out = pd.DataFrame(all_data)
    out = out[(out[COL_STRESS] > 0) & (out[COL_STRAIN] > 0)].copy()
    return out


def load_from_csv(path: str) -> pd.DataFrame:
    """
    Load from CSV with columns Temperature, StrainRate, TrueStrain, FlowStress.
    T_K is computed.
    """
    df = pd.read_csv(path)
    if COL_TK not in df.columns:
        df[COL_TK] = df[COL_TEMP] + 273.15
    df = df[(df[COL_STRESS] > 0) & (df[COL_STRAIN] > 0)].copy()
    return df


def generate_demo_data(
    seed: int = 42,
    n_temps: int = 3,
    n_rates: int = 3,
    n_strains: int = 5,
) -> pd.DataFrame:
    """
    Generate a minimal synthetic dataset so that all scripts can run without real data.
    Uses a simple Zener-Hollomon-type relation to produce physically plausible stress values.
    Total points ≈ n_temps * n_rates * n_strains (about 10–20 for default-like settings).
    """
    rng = np.random.default_rng(seed)
    # Temperatures: e.g. 800, 900, 950 so we have train (800, 900) and test (950)
    temps = np.linspace(800, 950, n_temps).astype(int).tolist()
    # Strain rates: e.g. 0.01, 0.1, 1.0 so we have train (0.01, 0.1) and test (1.0)
    rates = [0.01, 0.1, 1.0][:n_rates]
    strains = np.linspace(0.1, 0.5, n_strains)

    R_GAS = 8.314
    # Simple constitutive so that stress clearly varies with T, sr, strain (for regression)
    rows = []
    for temp in temps:
        T_K = temp + 273.15
        for sr in rates:
            for eps in strains:
                # Ensure clear variation: sigma ~ f(T, sr, eps) + noise
                sigma = 80 + 50 * np.log10(sr + 0.01) + 20 * (1 - (temp - 800) / 200) + 30 * eps
                sigma = float(np.clip(sigma + rng.uniform(-3, 3), 8.0, 350.0))
                rows.append({
                    COL_TEMP: temp,
                    COL_TK: T_K,
                    COL_SR: sr,
                    COL_STRAIN: round(eps, 4),
                    COL_STRESS: sigma,
                })
    return pd.DataFrame(rows)
