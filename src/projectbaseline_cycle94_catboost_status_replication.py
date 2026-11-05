"""Feature helpers retained from the original submission recipe."""
STATUS = ('p1_status_move_fraction_observed', 'p2_status_move_fraction_observed')

def status_fractions(record):
    result = {}
    for side in ('p1', 'p2'):
        moves = [frame[f'{side}_move_details'] for frame in record['battle_timeline'][:30]]
        moves = [move for move in moves if move is not None]
        assert moves and all((move['category'] in ('status', 'physical', 'special') for move in moves))
        result[f'{side}_status_move_fraction_observed'] = sum((move['category'] == 'status' for move in moves)) / len(moves)
    assert all((0 <= result[name] <= 1 for name in STATUS))
    return result
