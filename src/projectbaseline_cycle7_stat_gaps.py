"""Feature helpers retained from the original submission recipe."""
from __future__ import annotations
import pandas as pd
ESTIMATED = ('p1_hp_total_estimated', 'p2_hp_total_estimated', 'hp_total_diff_estimated')

def _base(model, base52, dynamics, interactions, windows, volatility):
    drop = set(ESTIMATED if model != 'CatBoost' else ('hp_total_diff_estimated',))
    X = base52.drop(columns=sorted(drop))
    if model in ('LR_L2', 'SVM_linear'):
        return pd.concat([X, dynamics, interactions, windows[['p1_hp_late_minus_early', 'p2_hp_late_minus_early']]], axis=1)
    if model == 'RandomForest':
        return pd.concat([X, dynamics, windows[['p1_hp_mean_t26_30', 'p2_hp_mean_t26_30']]], axis=1)
    if model == 'XGBoost':
        return pd.concat([X, dynamics, windows[['p1_hp_mean_t21_30', 'p2_hp_mean_t21_30']]], axis=1)
    vol = volatility[['p1_hp_abs_delta_mean_same_name', 'p2_hp_abs_delta_mean_same_name', 'p1_hp_decrease_rate_same_name', 'p2_hp_decrease_rate_same_name']]
    return pd.concat([X, windows[['p1_hp_late_minus_early', 'p2_hp_late_minus_early']], vol], axis=1)
