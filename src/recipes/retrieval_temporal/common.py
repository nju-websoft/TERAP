import os, json, hashlib
from pathlib import Path
DATASET = os.environ.get('PLACEMENT_DATASET', 'ieee')
VARIANT = os.environ.get('PLACEMENT_VARIANT', 'default')
KS = [1, 2, 4, 8, 16, 32, 64, 128, 200, 512, 1024]

def read(p):
    return json.loads(Path(p).read_text())

def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False))
    tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while (b := f.read(8 * 1024 * 1024)):
            h.update(b)
    return h.hexdigest()

def key(kind, m):
    return tuple((m[k] for k in (['R1024', 'H1024', 'MRR1024'] if kind == 'be' else ['H32', 'R32', 'MRR1024'])))

def runname(seed, epochs, kind, branch='valid'):
    return f'{DATASET}_{VARIANT}_s{seed}_{kind}'
from ..runtime import checked_read
read = checked_read(read)
from ..runtime import path_bindings
globals().update(path_bindings('retrieval_temporal', ['BASE', 'DATA', 'GRAPH', 'MODEL', 'O', 'R', 'SRC']))
