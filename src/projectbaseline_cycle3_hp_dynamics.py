"""Feature helpers retained from the original submission recipe."""
from __future__ import annotations
import numpy as np

def transition_features(battle):
    result = {}
    for side in ('p1', 'p2'):
        pairs = []
        timeline = battle['battle_timeline']
        for before, after in zip(timeline, timeline[1:]):
            a, b = (before[f'{side}_pokemon_state'], after[f'{side}_pokemon_state'])
            if a['name'] == b['name']:
                pairs.append(float(b['hp_pct']) - float(a['hp_pct']))
        result[f'{side}_hp_delta_same_name_mean'] = float(np.mean(pairs)) if pairs else 0.0
        result[f'{side}_hp_delta_same_name_pair_count'] = len(pairs)
    assert len(result) == 4 and all((np.isfinite(v) for v in result.values()))
    return result
