"""Feature helpers retained from the original submission recipe."""
from __future__ import annotations
import numpy as np
STATS = ('base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe', 'level')
SIDES = ('p1', 'p2')

def features(battle):
    row = {}
    team, lead, timeline = (battle['p1_team_details'], battle['p2_lead_details'], battle['battle_timeline'])
    for stat in STATS:
        values = [p[stat] for p in team]
        row.update({f'p1_{stat}_mean': float(np.mean(values)), f'p1_{stat}_min': min(values), f'p1_{stat}_max': max(values), f'p2_lead_{stat}': lead[stat]})
    for side in SIDES:
        state = timeline[-1][f'{side}_pokemon_state']
        row[f'{side}_hp_turn30'] = state['hp_pct']
        row[f'{side}_status_turn30'] = int(state['status'] != 'nostatus')
        row[f'{side}_boosts_turn30'] = sum(state['boosts'].values())
        latest = {}
        for turn in timeline:
            pokemon = turn[f'{side}_pokemon_state']
            latest[pokemon['name']] = pokemon['hp_pct']
        hp = np.asarray(list(latest.values()), dtype=float)
        row[f'{side}_n_observed'] = len(hp)
        row[f'{side}_hp_mean_observed'] = float(hp.mean())
        row[f'{side}_hp_min_observed'] = float(hp.min())
        row[f'{side}_hp_sum_observed'] = float(hp.sum())
        row[f'{side}_n_fainted_observed'] = int((hp <= 0).sum())
    row['hp_diff_turn30'] = row['p1_hp_turn30'] - row['p2_hp_turn30']
    row['boosts_diff_turn30'] = row['p1_boosts_turn30'] - row['p2_boosts_turn30']
    row['hp_mean_diff_observed'] = row['p1_hp_mean_observed'] - row['p2_hp_mean_observed']
    row['hp_sum_diff_observed'] = row['p1_hp_sum_observed'] - row['p2_hp_sum_observed']
    row['fainted_diff_observed'] = row['p1_n_fainted_observed'] - row['p2_n_fainted_observed']
    for side in SIDES:
        row[f'{side}_hp_total_estimated'] = row[f'{side}_hp_sum_observed'] + 6 - row[f'{side}_n_observed']
    row['hp_total_diff_estimated'] = row['p1_hp_total_estimated'] - row['p2_hp_total_estimated']
    assert len(row) == 52 and all((np.isfinite(v) for v in row.values()))
    return row
