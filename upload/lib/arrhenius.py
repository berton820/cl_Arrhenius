"""
Arrhenius (Sellars-Tegart) constitutive model: parameter solving, polynomial compensation, prediction.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.interpolate import PchipInterpolator
from typing import List, Dict, Optional

R_GAS = 8.314  # J/(mol·K)

# Column names (must match data_loader)
COL_TEMP = "Temperature"
COL_TK = "T_K"
COL_SR = "StrainRate"
COL_STRAIN = "TrueStrain"
COL_STRESS = "FlowStress"


def interpolate_stress_pchip(
    df_zone: pd.DataFrame,
    temps_list: List[int],
    strain_targets: np.ndarray,
    rates_list: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Interpolate flow stress at discrete strain points using PCHIP per (T, strain_rate) curve.
    Returns DataFrame with columns Temperature, T_K, StrainRate, TrueStrain, FlowStress.
    """
    if rates_list is None:
        rates_list = sorted(df_zone[COL_SR].unique().tolist())
    rows = []
    for temp in temps_list:
        for sr in rates_list:
            sub = df_zone[(df_zone[COL_TEMP] == temp) & (df_zone[COL_SR] == sr)].sort_values(COL_STRAIN)
            grp = sub.groupby(COL_STRAIN)[COL_STRESS].mean().reset_index()
            if len(grp) < 4:
                continue
            x = grp[COL_STRAIN].values
            y = grp[COL_STRESS].values
            try:
                interp = PchipInterpolator(x, y, extrapolate=False)
                for eps in strain_targets:
                    if x.min() <= eps <= x.max():
                        sigma = float(interp(eps))
                        if np.isfinite(sigma) and sigma > 0:
                            rows.append({
                                COL_TEMP: temp,
                                COL_TK: temp + 273.15,
                                COL_SR: sr,
                                COL_STRAIN: round(float(eps), 4),
                                COL_STRESS: sigma,
                            })
            except Exception:
                pass
    return pd.DataFrame(rows)


def solve_arrhenius_params(
    df_disc: pd.DataFrame,
    temps_list: List[int],
    rates_list: List[float],
    strain_targets: np.ndarray,
) -> Dict[float, Dict[str, float]]:
    """
    Solve strain-compensated Arrhenius parameters at each discrete strain.
    Returns dict[eps -> {alpha, n, Q, lnA}].
    """
    results = {}
    for eps in strain_targets:
        eps_r = round(float(eps), 4)
        sub = df_disc[np.isclose(df_disc[COL_STRAIN], eps_r)]
        if len(sub) < 4:
            continue

        n1_slopes = []
        for t in temps_list:
            g = sub[sub[COL_TEMP] == t]
            if len(g) >= 2:
                s, _, _, _, _ = linregress(np.log(g[COL_STRESS]), np.log(g[COL_SR]))
                n1_slopes.append(s)

        beta_slopes = []
        for t in temps_list:
            g = sub[sub[COL_TEMP] == t]
            if len(g) >= 2:
                s, _, _, _, _ = linregress(g[COL_STRESS].values, np.log(g[COL_SR].values))
                beta_slopes.append(s)

        if not n1_slopes or not beta_slopes:
            continue

        n1 = float(np.mean(n1_slopes))
        beta = float(np.mean(beta_slopes))
        alpha = beta / n1

        n_slopes = []
        for t in temps_list:
            g = sub[sub[COL_TEMP] == t]
            if len(g) >= 2:
                x = np.log(np.sinh(alpha * g[COL_STRESS].values))
                y = np.log(g[COL_SR].values)
                if np.all(np.isfinite(x)):
                    s, _, _, _, _ = linregress(x, y)
                    n_slopes.append(s)

        if not n_slopes:
            continue

        n_val = float(np.mean(n_slopes))

        S_slopes = []
        for sr in rates_list:
            g = sub[sub[COL_SR] == sr]
            if len(g) >= 2:
                x = 1.0 / g[COL_TK].values
                y = np.log(np.sinh(alpha * g[COL_STRESS].values))
                if np.all(np.isfinite(y)):
                    s, _, _, _, _ = linregress(x, y)
                    S_slopes.append(s)

        if not S_slopes:
            continue

        Q = float(R_GAS * n_val * np.mean(S_slopes))

        sigma_all = sub[COL_STRESS].values
        T_K_all = sub[COL_TK].values
        sr_all = sub[COL_SR].values
        ln_Z = np.log(sr_all) + Q / (R_GAS * T_K_all)
        x_all = np.log(np.sinh(alpha * sigma_all))
        valid = np.isfinite(ln_Z) & np.isfinite(x_all)

        if int(valid.sum()) >= 3:
            _, intercept, _, _, _ = linregress(x_all[valid], ln_Z[valid])
            results[eps_r] = {"alpha": alpha, "n": n_val, "Q": Q, "lnA": float(intercept)}

    return results


def build_poly_dict(arr_params: Dict[float, Dict[str, float]], poly_degree: int = 6) -> tuple:
    """
    Build polynomial coefficients for alpha, n, Q, lnA vs strain.
    Returns (eps_arr, poly_dict).
    """
    eps_arr = np.array(sorted(arr_params.keys()), dtype=float)
    poly_dict = {}
    for key in ["alpha", "n", "Q", "lnA"]:
        vals = np.array([arr_params[e][key] for e in eps_arr], dtype=float)
        poly_dict[key] = np.polyfit(eps_arr, vals, min(poly_degree, len(eps_arr) - 1))
    return eps_arr, poly_dict


def arrhenius_predict(epsilon: float, strain_rate: float, T_K: float, poly_dict: Dict[str, np.ndarray]) -> float:
    """Predict flow stress (MPa) from strain, strain rate, and temperature (K)."""
    alpha = np.polyval(poly_dict["alpha"], epsilon)
    n_val = np.polyval(poly_dict["n"], epsilon)
    Q_val = np.polyval(poly_dict["Q"], epsilon)
    lnA = np.polyval(poly_dict["lnA"], epsilon)

    ln_Z = np.log(strain_rate) + Q_val / (R_GAS * T_K)
    x = np.exp((ln_Z - lnA) / n_val)
    sigma = (1.0 / alpha) * np.log(x + np.sqrt(x ** 2 + 1))
    return float(sigma)
