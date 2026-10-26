import hashlib
import json
import os
from pathlib import Path
DATASET = os.environ.get('PLACEMENT_DATASET', 'ieee_static')
KS = [1, 2, 4, 8, 16, 32]
CONFIGS = {'default': {'dim': 384, 'layers': 1, 'heads': 8, 'ff': 768, 'pair': 'none', 'lam': 0.0, 'hard': 'top8', 'memory_slots': 64}}

def read(p):
    return json.loads(Path(p).read_text())

def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda : f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def metric(scores, gold, rows):
    import numpy as np
    from collections import defaultdict
    order = np.argsort(-scores, axis=1, kind='stable')
    ranked = np.take_along_axis(gold, order, axis=1)
    rank = np.where(ranked, np.arange(1, ranked.shape[1] + 1)[None], np.inf).min(1)
    counts = np.asarray([len(set(r['gold_indices'])) for r in rows])
    groups = defaultdict(list)
    for (i, r) in enumerate(rows):
        v = {f'H{k}': float(rank[i] <= k) for k in KS}
        v.update({f'R{k}': float(ranked[i, :k].sum() / counts[i]) for k in KS})
        v['MRR32'] = float(1 / rank[i] if rank[i] <= 32 else 0.0)
        groups[r['target']].append(v)
    return {k: float(np.mean([np.mean([v[k] for v in group]) for group in groups.values()])) for k in next(iter(groups.values()))[0]}
from ..runtime import checked_read
read = checked_read(read)
from ..runtime import path_bindings
globals().update(path_bindings('placement_static', ['CACHE_ROOT', 'MODEL', 'O', 'ROOT_O']))
