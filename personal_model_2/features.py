"""Feature helpers retained from the original submission recipe."""
import numpy as np
import pandas as pd
FEATURES = ('p2_worst_hp_turn', 'hp_trend_acceleration_diff', 'p1_status_immediately_after_switch', 'p2_status_immediately_after_switch', 'p1_status_duration_total', 'p2_status_duration_total')

def _max_status_duration(status_by_turn):
    return sum((status_by_turn[turn] != 'nostatus' for turn in range(2, 31)))

def _status_after_switch(status_by_turn, name_by_turn):
    total = 0
    for turn in range(2, 31):
        previous_name = name_by_turn[turn - 1]
        current_name = name_by_turn[turn]
        if current_name != previous_name and previous_name and current_name:
            following_turn = turn + 1 if turn < 30 else turn
            following_status = status_by_turn.get(following_turn, status_by_turn[turn])
            if following_status != status_by_turn[turn] and following_status != 'nostatus':
                total += 1
    return total

def create_simple_features(data):
    rows = []
    for battle in data:
        timeline = battle.get('battle_timeline') or []
        by_turn = {int(snapshot.get('turn', index + 1)): snapshot for index, snapshot in enumerate(timeline[:30])}
        hp, names, statuses = ({}, {}, {})
        for side in ('p1', 'p2'):
            hp_by_turn, name_by_turn, status_by_turn = ({}, {}, {})
            for turn in range(1, 31):
                snapshot = by_turn.get(turn)
                if snapshot is not None:
                    state = snapshot.get(f'{side}_pokemon_state', {})
                    hp_by_turn[turn] = float(state.get('hp_pct', 0) or 0)
                    name_by_turn[turn] = (state.get('name') or '').lower()
                    status_by_turn[turn] = (state.get('status') or 'nostatus').lower()
                else:
                    hp_by_turn[turn] = hp_by_turn.get(turn - 1, 0.0)
                    name_by_turn[turn] = name_by_turn.get(turn - 1, '')
                    status_by_turn[turn] = status_by_turn.get(turn - 1, 'nostatus')
            hp[side], names[side], statuses[side] = (hp_by_turn, name_by_turn, status_by_turn)
        turns = np.arange(1, 31, dtype=float)
        accelerations = {}
        for side in ('p1', 'p2'):
            values = np.asarray([hp[side][turn] for turn in range(1, 31)], dtype=float)
            first_slope = float(np.polyfit(turns[:15], values[:15], 1)[0])
            second_slope = float(np.polyfit(turns[15:], values[15:], 1)[0])
            accelerations[side] = second_slope - first_slope
        rows.append({'battle_id': battle.get('battle_id'), 'p2_worst_hp_turn': int(np.argmin([hp['p2'][turn] for turn in range(1, 31)]) + 1), 'hp_trend_acceleration_diff': accelerations['p1'] - accelerations['p2'], 'p1_status_immediately_after_switch': _status_after_switch(statuses['p2'], names['p2']), 'p2_status_immediately_after_switch': _status_after_switch(statuses['p1'], names['p1']), 'p1_status_duration_total': _max_status_duration(statuses['p2']), 'p2_status_duration_total': _max_status_duration(statuses['p1'])})
    return pd.DataFrame(rows, columns=['battle_id', *FEATURES])
