"""Load the arrays used by Placement training."""
import json
from pathlib import Path
import numpy as np


def load_placement_cache(directory):
    directory = Path(directory)
    rows = json.loads((directory / 'rows.json').read_text())
    splits = [r.get('split') for r in rows]
    counts = [splits.count(s) for s in ('train', 'valid', 'test')]
    if splits != ['train'] * counts[0] + ['valid'] * counts[1] + ['test'] * counts[2] or not all(counts[:2]):
        raise ValueError('Cache rows must be ordered train, valid, then optional test, with explicit splits.')
    train_targets = {r['target'] for r in rows[:counts[0]]}
    if train_targets & {r['target'] for r in rows[counts[0]:]}:
        raise ValueError('Training and held-out target concepts overlap.')
    with np.load(directory / 'inputs.npz') as archive:
        inputs = {k: archive[k] for k in archive.files}
    features = np.load(directory / 'features.npy')
    bank = np.load(directory / 'path_features.npy')
    memory, roles, gold, hard = (inputs[k] for k in ('memory', 'roles', 'gold', 'hard8'))
    if not (len(features) == len(memory) == len(gold) == len(rows)):
        raise ValueError('Cache rows and feature arrays must align.')
    if memory.shape != roles.shape or gold.shape != hard.shape or features.shape[:2] != gold.shape:
        raise ValueError('Inconsistent cache candidate or memory dimensions.')
    if memory.dtype.kind not in 'iu' or memory.min() < 0 or memory.max() >= len(bank):
        raise ValueError('Memory indices must address the supplied path feature bank.')
    if not np.isin(roles, [0, 1, 2, 3]).all() or not (roles != 3).any(axis=1).all():
        raise ValueError('Each query needs at least one valid memory path.')
    return rows, counts, inputs, features, bank
