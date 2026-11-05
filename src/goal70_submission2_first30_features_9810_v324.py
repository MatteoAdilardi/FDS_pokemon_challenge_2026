"""Feature helpers retained from the original submission recipe."""
from collections import Counter
import numpy as np
FEATURES = ('active_entropy_diff', 'status_turns_advantage', 'hp_gap_peak_turn_share', 'hp_gap_sign_flips', 'attacks_rate_diff', 'substitute_late_share_diff', 'sleep_streak_max_diff')
STATUS_SEVERITY = {'nostatus': 0, 'par': 1, 'brn': 1, 'psn': 1, 'tox': 2, 'frz': 2, 'slp': 3, 'fnt': 0}

def longest_streak(values, wanted):
    best = run = 0
    for value in values:
        run = run + 1 if value == wanted else 0
        best = max(best, run)
    return best

def extract(battle):
    turns = (battle.get('battle_timeline') or [])[:30]
    n = len(turns)
    states = {side: [turn.get(f'{side}_pokemon_state') or {} for turn in turns] for side in ('p1', 'p2')}
    names = {side: [str(s.get('name', '')).lower() for s in rows] for side, rows in states.items()}
    statuses = {side: [str(s.get('status', 'nostatus')).lower() for s in rows] for side, rows in states.items()}
    hp = {side: np.asarray([float(s.get('hp_pct', 0.0)) for s in rows], dtype=float) for side, rows in states.items()}
    gap = hp['p2'] - hp['p1']
    counts = {side: Counter(names[side]) for side in ('p1', 'p2')}

    def entropy(counter):
        total = sum(counter.values())
        if not total:
            return 0.0
        p = np.asarray(list(counter.values()), dtype=float) / total
        return float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    p1_moves = [turn.get('p1_move_details') or {} for turn in turns]
    p2_moves = [turn.get('p2_move_details') or {} for turn in turns]
    E = max(1, n // 3) if n else 1
    late_start = max(0, n - E)
    p1_effects = [set(s.get('effects') or []) for s in states['p1']]
    p2_effects = [set(s.get('effects') or []) for s in states['p2']]
    signs = np.sign(gap)
    values = {'active_entropy_diff': entropy(counts['p1']) - entropy(counts['p2']), 'status_turns_advantage': sum((STATUS_SEVERITY.get(s, 0) for s in statuses['p1'])) - sum((STATUS_SEVERITY.get(s, 0) for s in statuses['p2'])), 'hp_gap_peak_turn_share': float(np.argmax(gap) + 1) / n if n else 0.0, 'hp_gap_sign_flips': sum((1 for a, b in zip(signs, signs[1:]) if a != 0 and b != 0 and (a != b))), 'attacks_rate_diff': (sum((bool(m.get('name')) for m in p1_moves)) - sum((bool(m.get('name')) for m in p2_moves))) / n if n else 0.0, 'substitute_late_share_diff': (sum(('substitute' in e for e in p1_effects[late_start:])) - sum(('substitute' in e for e in p2_effects[late_start:]))) / max(1, n - late_start) if n else 0.0, 'sleep_streak_max_diff': longest_streak(statuses['p1'], 'slp') - longest_streak(statuses['p2'], 'slp')}
    result = {k: float(values[k]) for k in FEATURES}
    assert all((np.isfinite(v) for v in result.values()))
    return result
