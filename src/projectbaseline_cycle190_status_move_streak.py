"""Feature helpers retained from the original submission recipe."""
FEATURES = ('p1_trailing_status_move_turns', 'p2_trailing_status_move_turns')

def move_recency_features(record):
    frames = record['battle_timeline']
    if [frame['turn'] for frame in frames] != list(range(1, 31)):
        raise ValueError('Expected ordered turns 1..30')
    result, counts = ({}, {})
    for side in ('p1', 'p2'):
        count = 0
        for frame in reversed(frames):
            move = frame[f'{side}_move_details']
            if move is None:
                break
            if not isinstance(move, dict) or move.get('category') not in ('status', 'physical', 'special'):
                raise ValueError('Expected an observed move category')
            if move['category'] != 'status':
                break
            count += 1
        result[FEATURES[0 if side == 'p1' else 1]] = count
        counts[f'{side}_recorded_status_moves_30'] = sum((frame[f'{side}_move_details'] is not None and frame[f'{side}_move_details']['category'] == 'status' for frame in frames))
    return (result, counts)
