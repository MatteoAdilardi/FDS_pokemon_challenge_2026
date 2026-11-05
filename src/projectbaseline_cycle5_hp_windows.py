"""Feature helpers retained from the original submission recipe."""
from __future__ import annotations
import numpy as np

def window_features(battle):
    timeline = battle['battle_timeline']
    if len(timeline) != 30 or [f.get('turn') for f in timeline] != list(range(1, 31)):
        raise ValueError('Expected ordered snapshots for turns 1..30.')
    result = {}
    for side in ('p1', 'p2'):
        hp = np.asarray([float(f[f'{side}_pokemon_state']['hp_pct']) for f in timeline], dtype=float)
        result[f'{side}_hp_mean_t21_30'] = float(hp[20:30].mean())
        result[f'{side}_hp_mean_t26_30'] = float(hp[25:30].mean())
        result[f'{side}_hp_late_minus_early'] = float(hp[20:30].mean() - hp[0:10].mean())
    assert len(result) == 6 and np.isfinite(list(result.values())).all()
    return result
