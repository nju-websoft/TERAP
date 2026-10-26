"""Filesystem bindings, split checks and checkpoint saving."""
import json
import os
from pathlib import Path


def path_bindings(recipe, names):
    root = Path(os.environ.get('RELEASE_ROOT', Path(__file__).resolve().parents[2]))
    work = Path(os.environ.get('RELEASE_WORK', root / 'work')) / recipe
    if recipe == 'retrieval_temporal':
        work = work / os.environ.get('PLACEMENT_DATASET', 'ieee')
    overrides = json.loads(os.environ.get('RELEASE_PATHS', '{}')).get(recipe, {})
    result = {}
    for name in names:
        if name in ('R', 'ROOT'):
            default = root
        elif name == 'MODEL' or (name == 'BASE' and recipe in ('mesh_query', 'mesh_memory', 'ieee_query', 'ieee_memory', 'qmesh_encoder', 'retrieval_ieee')):
            default = root / 'models' / 'bert'
        elif name in ('O', 'OUT', 'ROOT_O'):
            default = work
        elif name == 'CACHE_ROOT':
            default = work / 'cache'
        else:
            default = work / name.lower()
        result[name] = Path(overrides.get(name, default))
    if recipe == 'retrieval_ieee' and 'SRC' not in overrides:
        result['SRC'] = result['OUT']
    return result


def checked_read(reader):
    def read(path):
        value = reader(path)
        split = Path(path).stem
        if split in ('train', 'valid', 'test') and isinstance(value, list):
            for row in value:
                if 'split' in row and row['split'] != split:
                    raise ValueError('Row split conflicts with its source file.')
                row['split'] = split
        if Path(path).name == 'rows.json' and isinstance(value, list):
            if any(row.get('split') not in ('train', 'valid', 'test') for row in value):
                raise ValueError('Cached rows must explicitly identify their split.')
        return value
    return read


def save_checkpoint(path, state):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)
