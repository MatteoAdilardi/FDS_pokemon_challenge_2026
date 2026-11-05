"""Feature helpers retained from the original submission recipe."""
from __future__ import annotations
from collections import Counter
import numpy as np
STATS = ('base_hp', 'base_atk')

def weighted_observed_hp_features(battle):
    team = battle.get('p1_team_details')
    timeline = battle.get('battle_timeline')
    if not isinstance(team, list) or len(team) != 6 or (not isinstance(timeline, list)):
        raise ValueError('Expected six P1 cards and a timeline.')
    last = {}
    for frame in timeline:
        state = frame['p1_pokemon_state']
        last[state['name']] = float(state['hp_pct'])
    counts = Counter((card['name'] for card in team))
    observed = [c for c in team if counts[c['name']] == 1 and c['name'] in last]
    result = {}
    for stat in STATS:
        full = sum((float(c[stat]) for c in team))
        known = sum((float(c[stat]) for c in observed))
        result[f'p1_observed_hp_weighted_{stat}'] = sum((float(c[stat]) * last[c['name']] for c in observed)) / known if known else np.nan
        result[f'p1_observed_stat_coverage_{stat}'] = known / full if full > 0 else np.nan
    assert len(result) == 4 and np.isfinite(np.asarray(list(result.values()), dtype=float)).all()
    return result
