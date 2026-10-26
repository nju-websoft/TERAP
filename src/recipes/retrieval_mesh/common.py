import os, json, hashlib
from pathlib import Path
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
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def key(kind, m):
    return tuple((m[k] for k in (['R1024', 'H1024', 'MRR'] if kind == 'be' else ['H32', 'R32', 'MRR'])))

def runname(seed, epochs, kind, branch='valid'):
    return f's{seed}_e{epochs}_{kind}' + ('_standard' if kind == 'ce' else '')
from ..runtime import checked_read
read = checked_read(read)
from ..runtime import path_bindings
globals().update(path_bindings('retrieval_mesh', ['DATA', 'GRAPH', 'MODEL', 'O', 'R', 'SRC']))
