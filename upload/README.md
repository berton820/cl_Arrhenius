# TC4 Hot Deformation Constitutive Modeling (NN-PhysicsInit)

Code release for the paper on physics-prior-initialized neural networks for robust extrapolation of flow stress in TC4 titanium alloy under hot deformation.

## Environment

- **Python**: 3.8, 3.9, or 3.10 recommended.
- **Install**:
  ```bash
  pip install -r requirements.txt
  ```
  Or with conda:
  ```bash
  conda create -n tc4 python=3.9
  conda activate tc4
  pip install -r requirements.txt
  ```

## Data

**This repository does not include real experimental data.** You can:

1. **Run with built-in demo** (no file): scripts use a tiny in-code synthetic set (~10–20 points) so all pipelines run without any data file.
2. **Use the included sample set** (for easy reproduction): `data/demo_tc4_100.csv` — about 100 rows (5 temperatures × 4 strain rates × 5 strain points), Zener–Hollomon-style synthetic data. Example:
   ```bash
   python run_arrhenius_baseline.py --data data/demo_tc4_100.csv --output out
   python run_temperature_extrapolation.py --data data/demo_tc4_100.csv --output out
   python run_strain_rate_extrapolation.py --data data/demo_tc4_100.csv --output out
   ```
   To regenerate this CSV: `python data/generate_sample_data.py` (fixed seed, reproducible).
3. **Use your own data**: provide a CSV or Excel path via `--data`.

- **CSV format**: columns `Temperature`, `StrainRate`, `TrueStrain`, `FlowStress`.
- **Excel format**: one sheet per temperature (sheet name = temperature in °C); each sheet has strain rate in row 0 and pairs of columns (strain, stress).

## Usage

Run each experiment from the `upload` directory:

```bash
# 1. Arrhenius baseline (strain-compensated constitutive model)
python run_arrhenius_baseline.py [--data path/to/data.csv] [--output figures_dir]

# 2. Temperature extrapolation (train 800–980°C, test 1010°C; four methods)
python run_temperature_extrapolation.py [--data path] [--output figures_dir]

# 3. Strain-rate extrapolation (train ε̇≤1 s⁻¹, test 10 s⁻¹; four methods)
python run_strain_rate_extrapolation.py [--data path] [--output figures_dir]
```

Without `--data`, the demo dataset is used. Without `--output`, figures are not saved (or go to a default folder).

## Methods

| Script | Description |
|--------|-------------|
| `run_arrhenius_baseline.py` | Fits a strain-compensated Arrhenius (Sellars–Tegart) model and evaluates R, R², AARE, RMSE. |
| `run_temperature_extrapolation.py` | Compares Arrhenius, NN-Direct, NN-Arrhenius-Only, and **NN-PhysicsInit** under temperature extrapolation. |
| `run_strain_rate_extrapolation.py` | Same four methods under strain-rate extrapolation. |

## File structure

```
upload/
├── README.md
├── requirements.txt
├── data/                 # optional: place your data here
├── lib/
│   ├── data_loader.py    # load Excel/CSV, generate demo data
│   ├── arrhenius.py      # Arrhenius solve & predict
│   ├── nn_models.py      # StressNet, training, prediction
│   ├── metrics.py        # R, R², AARE, RMSE
│   └── plot_utils.py     # plotting style and save
├── run_arrhenius_baseline.py
├── run_temperature_extrapolation.py
└── run_strain_rate_extrapolation.py
```

## Citation

If you use this code, please cite our paper (journal and details to be updated upon publication).

## License

See LICENSE file (if present). Otherwise use under common academic attribution terms.
