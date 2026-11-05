"""Feature helpers retained from the original submission recipe."""
from __future__ import annotations
import numpy as np

def _recent_move_difference(record: dict) -> float:
    frames = record['battle_timeline']
    if [frame.get('turn') for frame in frames] != list(range(1, 31)):
        raise ValueError('Expected ordered turns 1..30')
    window = frames[20:30]
    return float(np.mean([int(frame.get('p1_move_details') is not None) - int(frame.get('p2_move_details') is not None) for frame in window]))
