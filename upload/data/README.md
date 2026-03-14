# Data directory

- **`demo_tc4_100.csv`** — Sample dataset (~100 rows) for quick reproduction of the scripts.
  - **Columns**: `Temperature` (°C), `StrainRate` (s⁻¹), `TrueStrain`, `FlowStress` (MPa).
  - **Grid**: 5 temperatures (800, 850, 900, 950, 1010), 4 strain rates (0.001, 0.01, 0.1, 1.0), 5 strain points per (T, rate).
  - Synthetic data from a Zener–Hollomon-type relation (see `generate_sample_data.py`), not real experiments.

- **`generate_sample_data.py`** — Script to regenerate `demo_tc4_100.csv` (fixed seed 42, reproducible).

Run from project root with this data:

```bash
python run_arrhenius_baseline.py --data data/demo_tc4_100.csv
python run_temperature_extrapolation.py --data data/demo_tc4_100.csv
python run_strain_rate_extrapolation.py --data data/demo_tc4_100.csv
```
