import math
import numpy as np

def admission_metrics(q, c, y, tq, tc, ty):
    from sklearn.metrics import roc_auc_score

    def area(x, z, lo):
        keep = x > lo
        xx = np.r_[lo, x[keep]]
        zz = np.r_[np.interp(lo, x, z), z[keep]]
        return float(np.trapezoid(zz, xx) / (1 - lo))

    def single(z, y):
        ix = np.argsort(-z, kind='stable')
        s = z[ix]
        yy = y[ix]
        ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
        x = np.r_[0, np.cumsum(yy)[ends] / yy.sum()]
        u = np.r_[1, 1 - np.cumsum(~yy)[ends] / (~yy).sum()]
        return dict(AUROC=float(roc_auc_score(y, z)), pAUROC=area(x, u, 0.8))
    n = int(y.sum())
    best = [None] * n
    for a in np.unique(q[y]):
        mask = q >= a
        pos = np.sort(c[y & mask])[::-1]
        neg = np.sort(c[~y & mask])
        for b in np.unique(pos):
            count = int((pos >= b).sum())
            rej = int((~y).sum() - (len(neg) - np.searchsorted(neg, b, side='left')))
            key = (rej, count, float(a), float(b))
            for t in range(count):
                if best[t] is None or key > best[t]:
                    best[t] = key
    out = {}
    pairs = list(set(((v[2], v[3]) for v in best)))
    selected = best[math.ceil(0.9 * n) - 1]
    for sp, qq, cc, yy in [('valid', q, c, y), ('test', tq, tc, ty)]:
        raw = [(0.0, 1.0), (1.0, 0.0)]
        for a, b in pairs:
            passed = (qq >= a) & (cc >= b)
            raw.append((float(passed[yy].mean()), float((~passed[~yy]).mean())))
        x = np.array(sorted(set((a for a, b in raw))))
        u = np.array([np.mean([b for a, b in raw if a == v]) for v in x])
        passed = (qq >= selected[2]) & (cc >= selected[3])
        out[sp] = dict(pAUROC=area(x, u, 0.8), AUROC=area(x, u, 0), NR=float(passed[yy].mean()), UR=float((~passed[~yy]).mean()), single_memory=single(cc, yy), query_threshold=selected[2], memory_threshold=selected[3])
    return out


def cascade(query_scores, memory_scores, query_threshold, memory_threshold):
    return (np.asarray(query_scores) >= query_threshold) & (np.asarray(memory_scores) >= memory_threshold)


def placement_metrics(rankings, rows, ks=(1, 2, 4, 8, 16, 32)):
    """UID rankings; average queries within target, then average targets."""
    if len(rankings) != len(rows) or not rows:
        raise ValueError('Expected one ranking per nonempty sample row.')
    groups = {}
    for order, row in zip(rankings, rows):
        if len(set(order)) != len(order):
            raise ValueError('Duplicate candidate UID.')
        gold = set(row['gold'])
        if not gold:
            raise ValueError('Empty gold parent set.')
        positions = [i + 1 for i, uid in enumerate(order) if uid in gold]
        rank = min(positions, default=math.inf)
        values = {f'H@{k}': float(rank <= k) for k in ks}
        values.update({f'R@{k}': sum(p <= k for p in positions) / len(gold) for k in ks})
        values['MRR@32'] = 1 / rank if rank <= 32 else 0.0
        values['MRR'] = 1 / rank
        groups.setdefault(row['target'], []).append(values)
    return {key: float(np.mean([np.mean([v[key] for v in group]) for group in groups.values()]))
            for key in values}


def evaluate_placement(model, examples, batch_size=128):
    from .pipeline import predict_placement
    rankings = predict_placement(model, examples, batch_size)
    return {'metrics': placement_metrics(rankings, [e['row'] for e in examples]),
            'ids': [e['id'] for e in examples], 'rankings': rankings}


def evaluate_admission(query_model, memory_model, validation, test, batch_size=128):
    from .pipeline import predict_admission
    arrays = []
    for examples in (validation, test):
        labels = np.asarray([e['label'] for e in examples], dtype=bool)
        if not labels.any() or labels.all():
            raise ValueError('Admission evaluation requires both classes.')
        arrays.append((predict_admission(query_model, examples, batch_size),
                       predict_admission(memory_model, examples, batch_size), labels))
    return admission_metrics(*arrays[0], *arrays[1])
