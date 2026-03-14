#!/usr/bin/env python3
"""
Temperature extrapolation: train on 800-980°C, test on 1010°C (or highest temp in data).
Compares Arrhenius, NN-Direct, NN-Arrhenius-Only, NN-PhysicsInit.
Uses demo data by default.
"""

import argparse
import os
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.data_loader import load_from_excel, load_from_csv, generate_demo_data
from lib.arrhenius import (
    interpolate_stress_pchip,
    solve_arrhenius_params,
    build_poly_dict,
    arrhenius_predict,
)
from lib.metrics import evaluate_method
from lib.plot_utils import setup_style, save_fig, METHOD_COLORS, METHOD_ORDER

# NN (torch) imports
from lib.nn_models import (
    StressNet,
    Normalizer,
    prepare_features,
    train_model,
    predict_nn,
    transform_target_for_training,
)

COL_TEMP = "Temperature"
COL_TK = "T_K"
COL_SR = "StrainRate"
COL_STRAIN = "TrueStrain"
COL_STRESS = "FlowStress"

SEED = 42


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def main():
    parser = argparse.ArgumentParser(description="Temperature extrapolation experiment")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    set_seed()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data is None:
        df_all = generate_demo_data(seed=SEED, n_temps=3, n_rates=3, n_strains=5)
        temps_all = sorted(df_all[COL_TEMP].unique().tolist())
        TRAIN_TEMPS = temps_all[:-1]  # e.g. [800, 900]
        TEST_TEMPS = [temps_all[-1]]   # e.g. [950]
        print("Using synthetic demo data.")
    else:
        path = args.data
        df_all = load_from_csv(path) if path.lower().endswith(".csv") else load_from_excel(path)
        TRAIN_TEMPS = [800, 850, 900, 950, 980]
        TEST_TEMPS = [1010]
        # Use only temps present in data
        present = sorted(df_all[COL_TEMP].unique().tolist())
        TRAIN_TEMPS = [t for t in TRAIN_TEMPS if t in present]
        TEST_TEMPS = [t for t in TEST_TEMPS if t in present]
        if not TEST_TEMPS:
            TEST_TEMPS = [max(present)]
        if not TRAIN_TEMPS:
            TRAIN_TEMPS = present[:-1] if len(present) > 1 else present

    df_train_raw = df_all[df_all[COL_TEMP].isin(TRAIN_TEMPS)].copy()
    df_test_raw = df_all[df_all[COL_TEMP].isin(TEST_TEMPS)].copy()
    rates_list = sorted(df_all[COL_SR].unique().tolist())
    strain_targets = np.arange(0.05, 0.65, 0.05)

    df_train = interpolate_stress_pchip(df_train_raw, TRAIN_TEMPS, strain_targets, rates_list=rates_list)
    df_test = interpolate_stress_pchip(df_test_raw, TEST_TEMPS, strain_targets, rates_list=rates_list)

    if len(df_train) < 5 or len(df_test) < 2:
        print("Too few points after interpolation. Abort.")
        return 1

    # Arrhenius on train only
    arr_params = solve_arrhenius_params(df_train, TRAIN_TEMPS, rates_list, strain_targets)
    if not arr_params:
        print("Could not solve Arrhenius. Abort.")
        return 1
    eps_arr, poly_dict = build_poly_dict(arr_params, 6)

    # Predictions
    arr_pred_train = np.array([arrhenius_predict(r[COL_STRAIN], r[COL_SR], r[COL_TK], poly_dict) for _, r in df_train.iterrows()])
    arr_pred_test = np.array([arrhenius_predict(r[COL_STRAIN], r[COL_SR], r[COL_TK], poly_dict) for _, r in df_test.iterrows()])

    y_train = df_train[COL_STRESS].values
    y_test = df_test[COL_STRESS].values
    X_train = prepare_features(df_train)
    X_test = prepare_features(df_test)

    use_log = True
    y_train_tgt = transform_target_for_training(y_train, use_log_stress=use_log)
    y_test_tgt = transform_target_for_training(y_test, use_log_stress=use_log)
    norm_X = Normalizer()
    norm_y = Normalizer()
    # Synthetic data for PhysicsInit (build before fitting norm_y so one normalizer for all)
    N_SYN = 500
    syn_T = np.random.uniform(min(TRAIN_TEMPS) - 20, max(TEST_TEMPS) + 20, N_SYN)
    syn_TK = syn_T + 273.15
    syn_sr = np.exp(np.random.uniform(np.log(0.01), np.log(2), N_SYN))
    syn_eps = np.random.uniform(eps_arr.min(), eps_arr.max(), N_SYN)
    syn_stress = np.array([arrhenius_predict(eps, sr, tk, poly_dict) for eps, sr, tk in zip(syn_eps, syn_sr, syn_TK)])
    valid = np.isfinite(syn_stress) & (syn_stress > 0)
    X_syn = np.column_stack([1000 / syn_TK[valid], np.log(syn_sr[valid]), syn_eps[valid]])
    y_syn_log = transform_target_for_training(syn_stress[valid], use_log_stress=use_log)

    norm_X.fit(np.vstack([X_train, X_test]))
    norm_y.fit(np.concatenate([y_train_tgt, y_test_tgt, y_syn_log]))
    X_train_n = norm_X.transform(X_train)
    X_test_n = norm_X.transform(X_test)
    y_train_n = norm_y.transform(y_train_tgt)
    y_test_n = norm_y.transform(y_test_tgt)
    y_syn = norm_y.transform(y_syn_log)

    # Split train for validation
    n = len(X_train_n)
    idx = np.random.permutation(n)
    va_size = max(1, n // 5)
    tr_idx, va_idx = idx[va_size:], idx[:va_size]
    X_tr, X_va = X_train_n[tr_idx], X_train_n[va_idx]
    y_tr, y_va = y_train_n[tr_idx], y_train_n[va_idx]

    # NN-Direct
    model_direct = StressNet().to(device)
    train_model(model_direct, X_tr, y_tr, X_val=X_va, y_val=y_va, epochs=300, patience=80, verbose_every=100, device=device)
    nn_direct_train = predict_nn(model_direct, X_train, norm_X, norm_y, use_log_stress_target=use_log, device=device)
    nn_direct_test = predict_nn(model_direct, X_test, norm_X, norm_y, use_log_stress_target=use_log, device=device)

    # NN-Arrhenius-Only
    model_arronly = StressNet().to(device)
    train_model(model_arronly, X_syn, y_syn, epochs=200, verbose_every=100, device=device)
    nn_arronly_train = predict_nn(model_arronly, X_train, norm_X, norm_y, use_log_stress_target=use_log, device=device)
    nn_arronly_test = predict_nn(model_arronly, X_test, norm_X, norm_y, use_log_stress_target=use_log, device=device)

    # NN-PhysicsInit
    model_phys = StressNet().to(device)
    train_model(model_phys, X_syn, y_syn, epochs=200, verbose_every=100, device=device)
    train_model(model_phys, X_tr, y_tr, X_val=X_va, y_val=y_va, epochs=250, lr=5e-4, patience=60, verbose_every=100, device=device)
    nn_phys_train = predict_nn(model_phys, X_train, norm_X, norm_y, use_log_stress_target=use_log, device=device)
    nn_phys_test = predict_nn(model_phys, X_test, norm_X, norm_y, use_log_stress_target=use_log, device=device)

    denom_floor = 20.0
    print("\nTemperature extrapolation results")
    print("-" * 70)
    for name, ytr, yte in [
        ("Arrhenius", arr_pred_train, arr_pred_test),
        ("NN-Direct", nn_direct_train, nn_direct_test),
        ("NN-Arrhenius-Only", nn_arronly_train, nn_arronly_test),
        ("NN-PhysicsInit", nn_phys_train, nn_phys_test),
    ]:
        rtr = evaluate_method(name, y_train, ytr, eps=df_train[COL_STRAIN].values, denom_floor=denom_floor)
        rte = evaluate_method(name, y_test, yte, eps=df_test[COL_STRAIN].values, denom_floor=denom_floor)
        print(f"{name:20s} | Train R²={rtr['R2']:.4f} AARE={rtr['AARE(%)']:.2f}% | Test R²={rte['R2']:.4f} AARE={rte['AARE(%)']:.2f}%")

    if args.output:
        setup_style()
        os.makedirs(args.output, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, nn_phys_test, alpha=0.7, s=25, label="NN-PhysicsInit", color=METHOD_COLORS["NN-PhysicsInit"])
        mx = max(y_test.max(), nn_phys_test.max()) * 1.05
        ax.plot([0, mx], [0, mx], "k--", lw=1)
        ax.set_xlabel("Experimental (MPa)")
        ax.set_ylabel("Predicted (MPa)")
        ax.set_title("Temperature extrapolation: NN-PhysicsInit")
        ax.legend()
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        save_fig(fig, "temperature_extrapolation_physicsinit", args.output)
        plt.close(fig)
        print(f"Figure saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
