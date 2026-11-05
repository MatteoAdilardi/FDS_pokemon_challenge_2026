"""Feature helpers retained from the original submission recipe."""
import json
import numpy as np
import pandas as pd
from .goal70_submission2_first30_features_9810_v324 import FEATURES as ENS_FEATURES, extract as extract_ens
from .projectbaseline_cycle11_weighted_observed_hp import weighted_observed_hp_features
from .projectbaseline_cycle117_p1_items_only_replication import P1_ITEMS
from .projectbaseline_cycle3_hp_dynamics import transition_features
from .projectbaseline_cycle5_hp_windows import window_features
from .projectbaseline_cycle6_hp_volatility import volatility_features
from .projectbaseline_cycle7_stat_gaps import _base
from .projectbaseline_cycle94_catboost_status_replication import status_fractions
from .projectbaseline_model_importance_v1 import features as base52_features
from personal_model_2.features import create_simple_features
PM2_FEATURES = ('p2_worst_hp_turn', 'hp_trend_acceleration_diff', 'p1_status_immediately_after_switch', 'p2_status_immediately_after_switch', 'p1_status_duration_total', 'p2_status_duration_total')
ALL_FEATURES = (*PM2_FEATURES, *ENS_FEATURES)
STATUS_FEATURE = 'p2_unique_active_names_with_nonterminal_status'
STATUS_COLS = ('p1_status_move_fraction_observed', 'p2_status_move_fraction_observed')
WEIGHTED_COLS = ('p1_observed_hp_weighted_base_hp', 'p1_observed_hp_weighted_base_atk', 'p1_observed_stat_coverage_base_hp', 'p1_observed_stat_coverage_base_atk')
BAD_STATUS = {'tox', 'slp', 'par', 'psn', 'frz', 'brn'}
KNOWN_STATUS = {'nostatus', 'tox', 'slp', 'fnt', 'par', 'psn', 'frz', 'brn'}

def read_jsonl(path, training=False):
    records, labels = ([], [])
    with path.open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: record is not an object')
            if training:
                labels.append(int(row.pop('player_won')))
            elif 'player_won' in row:
                raise ValueError(f'{path}:{line_no}: test target unexpectedly present')
            timeline = row.get('battle_timeline') or []
            if len(timeline) < 30:
                raise ValueError(f'{path}:{line_no}: expected at least 30 timeline snapshots')
            row['battle_timeline'] = timeline[:30]
            records.append(row)
    return (records, np.asarray(labels, dtype=int) if training else None)

def raw_bases(records):
    base = pd.DataFrame([base52_features(row) for row in records])
    dynamics = pd.DataFrame([transition_features(row) for row in records])
    interactions = pd.DataFrame({'p1_def_x_hp_delta': base.p1_base_def_mean * dynamics.p1_hp_delta_same_name_mean, 'p2_lead_def_x_hp_delta': base.p2_lead_base_def * dynamics.p2_hp_delta_same_name_mean})
    windows = pd.DataFrame([window_features(row) for row in records])
    volatility = pd.DataFrame([volatility_features(row) for row in records])
    bases = {name: _base(name, base, dynamics, interactions, windows, volatility) for name in ('CatBoost', 'XGBoost')}
    weighted = pd.DataFrame([weighted_observed_hp_features(row) for row in records])[list(WEIGHTED_COLS)]
    bases['CatBoost'] = pd.concat([bases['CatBoost'], weighted], axis=1)
    status_moves = pd.DataFrame([status_fractions(row) for row in records])[list(STATUS_COLS)]
    return (bases, status_moves)

def item_frame(records):
    out = []
    for row in records:
        team = row.get('p1_team_details')
        if not isinstance(team, list) or len(team) != 6:
            raise ValueError('Expected six initial P1 cards')
        out.append({f'p1_item_slot_{slot}': team[slot - 1].get('item') or '__unknown__' for slot in range(1, 7)})
    return pd.DataFrame(out, columns=P1_ITEMS)

def statused_active_names(records):
    result = []
    for row in records:
        names = set()
        for snapshot in row['battle_timeline']:
            state = snapshot['p2_pokemon_state']
            status = state['status']
            if status not in KNOWN_STATUS:
                raise ValueError(f'Unexpected status code: {status}')
            if status in BAD_STATUS:
                names.add(state['name'])
        result.append(len(names))
    return pd.DataFrame({STATUS_FEATURE: result})

def feature_block(records, ids):
    simple = create_simple_features(records)
    if not np.array_equal(simple.battle_id.to_numpy(dtype=int), np.asarray(ids, dtype=int)):
        raise ValueError('PM2 battle_id alignment check failed')
    pm2 = simple.loc[:, list(PM2_FEATURES)].apply(pd.to_numeric, errors='raise')
    ens = pd.DataFrame([extract_ens(row) for row in records], columns=ENS_FEATURES)
    block = pd.concat([pm2.reset_index(drop=True), ens.reset_index(drop=True)], axis=1)
    if list(block.columns) != list(ALL_FEATURES) or not np.isfinite(block.to_numpy(dtype=float)).all():
        raise ValueError('Feature block is incomplete or contains non-finite values')
    return block
