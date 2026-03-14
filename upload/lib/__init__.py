# lib: shared modules for TC4 constitutive modeling
# nn_models (torch) is not imported here so that run_arrhenius_baseline can run without torch.

from .data_loader import load_from_excel, load_from_csv, generate_demo_data
from .arrhenius import solve_arrhenius_params, build_poly_dict, arrhenius_predict, interpolate_stress_pchip
from .metrics import compute_metrics, evaluate_method
from .plot_utils import setup_style, save_fig, METHOD_COLORS, METHOD_ORDER

__all__ = [
    "load_from_excel",
    "load_from_csv",
    "generate_demo_data",
    "solve_arrhenius_params",
    "build_poly_dict",
    "arrhenius_predict",
    "interpolate_stress_pchip",
    "compute_metrics",
    "evaluate_method",
    "setup_style",
    "save_fig",
    "METHOD_COLORS",
    "METHOD_ORDER",
]
