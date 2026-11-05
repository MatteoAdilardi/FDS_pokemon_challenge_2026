"""The two user-code predictors consumed by the selected V334 recipes."""
import numpy as np


def extract_user_features(battle):
    turns = (battle.get('battle_timeline') or [])[:30]
    lead = battle.get('p2_lead_details') or {}
    states = []
    for turn in turns:
        p1 = turn.get('p1_pokemon_state') or {}
        p2 = turn.get('p2_pokemon_state') or {}
        try:
            hp = float(p2.get('hp_pct', 0.0) or 0.0)
        except (TypeError, ValueError):
            hp = 0.0
        states.append(((p1.get('name'), p2.get('name')), hp))
    deltas = []
    if states:
        pair, start_hp = states[0]
        for index in range(1, len(states)):
            current_pair, hp = states[index]
            if current_pair != pair:
                deltas.append(states[index - 1][1] - start_hp)
                pair, start_hp = current_pair, hp
        deltas.append(states[-1][1] - start_hp)
    return {
        'encounter_hp_delta_p2_mean': float(np.mean(deltas)) if deltas else 0.0,
        'p2_lead_type_fighting': int('fighting' in [
            str(value).lower() for value in (lead.get('types') or [])
        ]),
    }
