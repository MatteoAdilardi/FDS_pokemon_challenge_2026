"""Feature helpers retained from the original submission recipe."""
from collections import defaultdict
import numpy as np
FIELDS = ['p2_lookup_hp_weighted_fraction', 'p2_lookup_hp_known_fraction', 'p2_lookup_known_base_hp_sum']

def profile(record):
    cards = [*record['p1_team_details'], record['p2_lead_details']]
    observations = []
    for card in cards:
        name, hp = (card['name'], float(card['base_hp']))
        if not isinstance(name, str) or not name or (not np.isfinite(hp)) or (hp <= 0):
            raise ValueError('Invalid initial base HP observation')
        observations.append((name, hp))
    frames = record['battle_timeline']
    if [frame['turn'] for frame in frames] != list(range(1, 31)):
        raise ValueError('Expected 30 ordered snapshots')
    last = {}
    for frame in frames:
        state = frame['p2_pokemon_state']
        hp = float(state['hp_pct'])
        if not isinstance(state['name'], str) or not state['name'] or (not np.isfinite(hp)) or (not 0 <= hp <= 1):
            raise ValueError('Invalid P2 observation')
        last[state['name']] = hp
    return {'cards': observations, 'lead': observations[-1], 'last': last}

def fit_lookup(profiles, train_ids):
    values = defaultdict(set)
    for row_id in train_ids:
        for name, hp in profiles[int(row_id)]['cards']:
            values[name].add(hp)
    conflicts = sorted((name for name, observed in values.items() if len(observed) != 1))
    return ({name: next(iter(observed)) for name, observed in values.items() if len(observed) == 1}, conflicts)

def transform_profile(observed, lookup):
    local_name, local_hp = observed['lead']
    known = []
    for name, hp in observed['last'].items():
        base_hp = local_hp if name == local_name else lookup.get(name)
        if base_hp is not None:
            known.append((hp, base_hp))
    denominator = sum((base for _, base in known))
    return dict(zip(FIELDS, [sum((hp * base for hp, base in known)) / denominator if denominator else np.nan, len(known) / len(observed['last']), denominator]))
