"""Feature helpers retained from the original submission recipe."""
from __future__ import annotations
import numpy as np

def volatility_features(battle):
    result = {}
    for side in ('p1', 'p2'):
        timeline = battle['battle_timeline']
        deltas = []
        for before, after in zip(timeline, timeline[1:]):
            a, b = (before[f'{side}_pokemon_state'], after[f'{side}_pokemon_state'])
            if a['name'] == b['name']:
                deltas.append(float(b['hp_pct']) - float(a['hp_pct']))
        result[f'{side}_hp_abs_delta_mean_same_name'] = float(np.mean(np.abs(deltas))) if deltas else 0.0
        result[f'{side}_hp_decrease_rate_same_name'] = float(np.mean(np.asarray(deltas) < 0)) if deltas else 0.0
        result[f'{side}_hp_delta_pair_count'] = len(deltas)
    assert len(result) == 6 and np.isfinite(list(result.values())).all()
    return result
