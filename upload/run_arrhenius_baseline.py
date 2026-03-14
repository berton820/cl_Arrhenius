#!/usr/bin/env python3
"""
Run Arrhenius baseline: strain-compensated constitutive model fit and evaluation.
Corresponds to notebook 2 (Arrhenius calculation). Uses demo data by default.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent so that upload/ can be run from repo root or from upload/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only modules needed for Arrhenius (no torch)
from lib.data_loader import load_from_excel, load_from_csv, generate_demo_data
from lib.arrhenius import (
    interpolate_stress_pchip,
    solve_arrhenius_params,
    build_poly_dict,
    arrhenius_predict,
    R_GAS,
)
from lib.metrics import compute_metrics
from lib.plot_utils import setup_style, save_fig

COL_TEMP = "Temperature"
COL_TK = "T_K"
COL_SR = "StrainRate"
COL_STRAIN = "TrueStrain"
COL_STRESS = "FlowStress"


def main():
    parser = argparse.ArgumentParser(description="Arrhenius baseline fit and evaluation")
    parser.add_argument("--data", type=str, default=None, help="Path to data (CSV or Excel). If not set, use demo data.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save figures. If not set, no figures saved.")
    args = parser.parse_args()

    # Load data
    if args.data is None:
        df_all = generate_demo_data(seed=42, n_temps=3, n_rates=3, n_strains=5)
        print("Using synthetic demo data (no --data provided).")
    else:
        path = args.data
        if path.lower().endswith(".csv"):
            df_all = load_from_csv(path)
        else:
            df_all = load_from_excel(path)
    print(f"Total points: {len(df_all)}")

    temps_list = sorted(df_all[COL_TEMP].unique().tolist())
    rates_list = sorted(df_all[COL_SR].unique().tolist())
    strain_targets = np.arange(0.05, 0.65, 0.05)

    # Interpolate to discrete strain grid
    df_disc = interpolate_stress_pchip(df_all, temps_list, strain_targets, rates_list=rates_list)
    if len(df_disc) < 10:
        print("Warning: too few discrete points after interpolation. Using coarser strain grid.")
        strain_targets = np.linspace(0.15, 0.5, 5)
        df_disc = interpolate_stress_pchip(df_all, temps_list, strain_targets, rates_list=rates_list)
    print(f"Discrete points: {len(df_disc)}")

    # Solve Arrhenius and build polynomial
    arr_params = solve_arrhenius_params(df_disc, temps_list, rates_list, strain_targets)
    if not arr_params:
        print("ERROR: Could not solve Arrhenius parameters (need enough points per strain).")
        return 1

    eps_arr, poly_dict = build_poly_dict(arr_params, poly_degree=6)

    # Predict on discrete set
    pred_list = []
    for _, row in df_disc.iterrows():
        sigma = arrhenius_predict(row[COL_STRAIN], row[COL_SR], row[COL_TK], poly_dict)
        pred_list.append(sigma)

    y_true = df_disc[COL_STRESS].values
    y_pred = np.array(pred_list, dtype=float)
    valid = np.isfinite(y_pred) & (y_pred > 0)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    metrics = compute_metrics(y_true, y_pred)
    print("Metrics:")
    print(f"  R    = {metrics['R']:.4f}")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  AARE = {metrics['AARE(%)']:.2f}%")
    print(f"  RMSE = {metrics['RMSE']:.2f} MPa")

    if args.output:
        setup_style()
        os.makedirs(args.output, exist_ok=True)
        # Predicted vs Experimental
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, label="Points")
        mx = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([0, mx], [0, mx], "r--", lw=1.5, label="y = x")
        ax.set_xlabel("Experimental (MPa)")
        ax.set_ylabel("Predicted (MPa)")
        ax.set_title("Arrhenius: Predicted vs Experimental")
        ax.legend()
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        save_fig(fig, "arrhenius_predicted_vs_experimental", args.output)
        plt.close(fig)
        print(f"Figure saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
