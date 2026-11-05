"""Feature helpers retained from the original submission recipe."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from .refit_goal70_pm2_submission2_all13_v1 import feature_block, item_frame, raw_bases, statused_active_names
from .goal70_species_hp_v34 import FIELDS, fit_lookup, profile, transform_profile

def _c09_matrices(records, ids, *, lookup=None, conflicts=None, encoder=None):
    train_base, train_status_moves = raw_bases(records)
    profiles = {i: profile(row) for i, row in enumerate(records, 1)}
    if lookup is None:
        lookup, conflicts = fit_lookup(profiles, list(profiles))
    p2 = pd.DataFrame([transform_profile(profiles[i], lookup) for i in range(1, len(records) + 1)], columns=FIELDS)
    status = statused_active_names(records)
    block = feature_block(records, ids)
    cat = pd.concat([train_base['CatBoost'], p2, status, train_status_moves, block], axis=1)
    fit_encoder = encoder is None
    if fit_encoder:
        encoder = OneHotEncoder(handle_unknown='ignore', min_frequency=5, sparse_output=False)
    items = item_frame(records)
    item_values = encoder.fit_transform(items) if fit_encoder else encoder.transform(items)
    encoded = pd.DataFrame(item_values, columns=encoder.get_feature_names_out())
    xgb = pd.concat([train_base['XGBoost'], p2, train_status_moves, encoded, block], axis=1)
    lr = pd.concat([xgb, status], axis=1)
    matrices = {'CatBoost': cat, 'XGBoost': xgb, 'LogisticRegression_C10': lr}
    for name, matrix in matrices.items():
        if not matrix.columns.is_unique or {'battle_id', 'player_won'} & set(matrix.columns):
            raise ValueError(f'Invalid C09 design matrix for {name}')
        if not np.isfinite(matrix.apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)).all():
            raise ValueError(f'Non-finite values in C09 matrix for {name}')
    return (matrices, lookup, conflicts, encoder)
