"""Dataset-adapted formal BE/CE; validation-only checkpoint selection."""
import argparse, collections, concurrent.futures, math, shutil, os
import numpy as np, torch
from torch.nn import functional as F
from .common import *
from ...configuration import parameter as hp, inputs_for_recipe
from .model import Network, pair

def metrics(top, rows):
    groups = collections.defaultdict(list)
    for (p, r) in zip(top, rows):
        g = set(r['gold_indices'])
        rank = [j + 1 for (j, x) in enumerate(p) if int(x) in g]
        groups[r['target']].append({**{f'H{k}': float(any((j <= k for j in rank))) for k in KS}, **{f'R{k}': sum((j <= k for j in rank)) / len(g) for k in KS}, 'MRR': 1 / min(rank) if rank else 0.0, 'MRR32': 1 / min(rank) if rank and min(rank) <= 32 else 0.0, 'MRR1024': 1 / min(rank) if rank and min(rank) <= 1024 else 0.0})
    return {k: float(np.mean([np.mean([x[k] for x in xx]) for xx in groups.values()])) for k in next(iter(groups.values()))[0]} | dict(concepts=len(groups), queries=len(rows))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seed', type=int)
    ap.add_argument('epochs', type=int)
    ap.add_argument('kind', choices=['be', 'ce'])
    ap.add_argument('--micro', type=int, required=True)
    ap.add_argument('--lr', type=float, default=hp('lr'))
    a = ap.parse_args()
    kind = a.kind
    name = runname(a.seed, a.epochs, kind, 'valid')
    D = O / 'runs' / name
    D.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    optimstep = 0
    rows = {s: read(DATA / f'{s}.json') for s in ['train', 'valid', 'test']}
    rows = rows
    train = rows['train']
    years = sorted({r['year'] for rr in rows.values() for r in rr})
    qt = {y: np.load(SRC / f'{y}_tokens.npy') for y in years}
    pt = {y: np.load(BASE / f'cache/parent_tokens_{y}.npy') for y in years}
    graphs = {y: read(GRAPH / f'{y - 1}_index.json') for y in years}
    freq = collections.Counter(((r['year'], r['target']) for r in train))
    edges = [(i, p, 1 / (len(r['gold_indices']) * freq[r['year'], r['target']])) for (i, r) in enumerate(train) for p in r['gold_indices']]
    scale = len(edges) / sum((e[2] for e in edges))
    edges = [(i, p, w * scale) for (i, p, w) in edges]
    effective = hp('batch_size')
    assert effective % a.micro == 0
    steps = math.ceil(len(edges) / effective)
    total = steps * a.epochs
    model = Network(kind, 256).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=hp('weight_decay'))
    hard = None
    be_source = O / 'runs' / runname(a.seed, a.epochs, 'be') / 'selected_valid'
    base = None
    if kind == 'ce':
        hard = np.load(be_source / 'mined.npz')['top']
        assert len(hard) == len(train)
        assert all((r.get('masked_index', -1) not in hard[i] for (i, r) in enumerate(train)))
        base = {s: np.load(be_source / f'{s}.npz')['full'] for s in ['valid']}
        assert all((base[s].shape[0] == len(rows[s]) and base[s].shape[1] >= 1024 for s in base))
    last_status = [None, 0.0]

    @torch.inference_mode()
    def retrieve_many(splitrows, tag, k):
        model.eval()
        width = k if k is not None else max((len(pt[r['year']]) for rr in splitrows.values() for r in rr))
        top = {s: np.full((len(rr), width), -1, np.int32) for (s, rr) in splitrows.items()}
        done = 0
        total_ = sum(map(len, splitrows.values()))
        for y in sorted({r['year'] for rr in splitrows.values() for r in rr}):
            pieces = []
            for lo in range(0, len(pt[y]), 128):
                pieces.append(model.encode(torch.as_tensor(pt[y][lo:lo + 128], device='cuda'), 'parent'))
            bank = torch.cat(pieces)
            del pieces
            for (s, rr) in splitrows.items():
                ii = [i for (i, r) in enumerate(rr) if r['year'] == y]
                for lo in range(0, len(ii), 128):
                    jj = ii[lo:lo + 128]
                    q = model.encode(torch.as_tensor(np.stack([qt[y][rr[i]['token_index']] for i in jj]), device='cuda'), 'query')
                    sim = (q @ bank.T).cpu().numpy()
                    for (bi, ri) in enumerate(jj):
                        hidden = rr[ri].get('masked_index', -1)
                        if hidden >= 0:
                            sim[bi, hidden] = -np.inf
                    order = np.argsort(-sim, axis=1, kind='stable')
                    for (bi, ri) in enumerate(jj):
                        hidden = rr[ri].get('masked_index', -1)
                        rank = order[bi]
                        rank = rank[rank != hidden]
                        rank = rank[:k] if k is not None else rank
                        top[s][ri, :len(rank)] = rank
                    done += len(jj)
            del bank
        for (s, rr) in splitrows.items():
            assert all((r.get('masked_index', -1) < 0 or r['masked_index'] not in top[s][i] for (i, r) in enumerate(rr))), 'masked leaf leaked into BE results'
        return top

    @torch.inference_mode()
    def rerank(rr, candidates, tag):
        model.eval()
        ranked = candidates.copy()
        candidates = candidates[:, :1024]
        tasks = [np.arange(i, min(i + 2, len(rr))) for i in range(0, len(rr), 2)]

        def prep(ii):
            qs = []
            ps = []
            for i in ii:
                r = rr[i]
                qs.extend([qt[r['year']][r['token_index']]] * 1024)
                ps.extend(pt[r['year']][candidates[i]])
            return pair(qs, ps)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            f = pool.submit(prep, tasks[0])
            for (j, ii) in enumerate(tasks):
                (ids, types) = f.result()
                if j + 1 < len(tasks):
                    f = pool.submit(prep, tasks[j + 1])
                scores = np.concatenate([model(torch.as_tensor(ids[k:k + 128], device='cuda'), types=torch.as_tensor(types[k:k + 128], device='cuda')).cpu().numpy() for k in range(0, len(ids), 128)]).reshape(len(ii), 1024)
                order = np.lexsort((candidates[ii], -scores), axis=1)
                ranked[ii, :1024] = np.take_along_axis(candidates[ii], order, axis=1)
        return ranked

    def closure(nodes, adj):
        found = set()
        todo = list(nodes)
        while todo:
            u = todo.pop()
            if u not in found:
                found.add(u)
                todo.extend(adj[u])
        return found
    pools = {}

    def neg(qi, epoch, count):
        r = train[qi]
        gold = set(r['gold_indices'])
        forbidden = gold | ({r['masked_index']} if r.get('masked_index', -1) >= 0 else set())
        g = graphs[r['year']]
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, epoch, qi]))
        out = []
        cachekey = (r['year'], r['target'])
        if cachekey not in pools:
            pools[cachekey] = ({s for p in gold for gp in g['parents'][p] for s in g['children'][gp]}, closure(gold, g['parents']), closure(gold, g['children']))

        def add(pool, n, label):
            pp = sorted(set(map(int, pool)) - forbidden - set(out))
            if pp:
                pick = rng.choice(pp, min(n, len(pp)), replace=False).tolist()
                out.extend(pick)
        budget = [10, 4, 2, 2] if count == 20 else rng.multinomial(count, [0.5, 0.2, 0.1, 0.1, 0.1])[:4]
        add(hard[qi] if hard is not None else [], budget[0], 'BE_hard')
        for (pool, n, label) in zip(pools[cachekey], budget[1:], ['sibling', 'ancestor', 'descendant']):
            add(pool, n, label)
        while len(out) < count:
            n = int(rng.integers(len(g['uids'])))
            if n not in forbidden and n not in out:
                out.append(n)
        return out
    cfg = dict(name=name, kind=kind, seed=a.seed, epochs=a.epochs, lr=a.lr, dim=256, effective_batch=effective, micro=a.micro, train_queries=len(train), train_edges=len(edges), train_concepts=len(freq), warmup_fraction=hp('warmup_fraction'), lr_schedule='configured linear warmup then decay to zero', selection_BE=['R1024', 'H1024', 'MRR1024'], selection_CE=['H32', 'R32', 'MRR1024'], valid=len(rows['valid']), test=len(rows['test']), BE_topK=1024, CE_train_hard_pool=hp('hard_pool_k'), train_negatives=4 if kind == 'be' else 20, train_years=sorted({r['year'] for r in train}), BE_source_branch='valid' if kind == 'ce' else None, BE_source_sha=sha(be_source / 'best.pt') if kind == 'ce' else None, scope='validation-only checkpoint selection', negative_recipe='standard negatives', evaluation_schedule=dict(every_epochs=hp('eval_every_epochs')), dataset=DATASET, variant=VARIANT)
    write(D / 'config.json', cfg)
    history = []
    best = {}
    start_epoch = 1
    if (D / 'last.pt').exists():
        checkpoint = torch.load(D / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        optimstep = int(checkpoint['optimizer_step'])
        start_epoch = int(checkpoint['epoch']) + 1
        history = [x for x in read(D / 'history.json') if x['optimizer_step'] <= optimstep]
        write(D / 'history.json', history)
        for split in ['valid']:
            selected = read(D / f'selected_{split}/selection.json')
            assert selected['optimizer_step'] <= optimstep
            best[split] = key(kind, selected[split])
        if kind == 'be' and start_epoch <= a.epochs:
            hard = retrieve_many({'train': train}, 'resume_mine_hard_negatives', hp('hard_pool_k'))['train']
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    for epoch in range(start_epoch, a.epochs + 1):
        order = np.random.default_rng(a.seed + epoch).permutation(len(edges))
        model.train()
        loss_values = []
        for step in range(steps):
            ei = order[step * effective:(step + 1) * effective]
            opt.zero_grad(set_to_none=True)
            lossvalue = 0.0
            for lo in range(0, len(ei), a.micro):
                block = [edges[i] for i in ei[lo:lo + a.micro]]
                qs = []
                ps = []
                weights = []
                for (qi, p, w) in block:
                    r = train[qi]
                    ns = neg(qi, epoch, 4 if kind == 'be' else 20)
                    qs.append(qt[r['year']][r['token_index']])
                    ps.append(pt[r['year']][[p] + ns])
                    weights.append(w)
                q = np.stack(qs)
                p = np.stack(ps)
                if kind == 'be':
                    v = model(torch.as_tensor(q, device='cuda'), torch.as_tensor(p, device='cuda'))
                    losses = F.cross_entropy(v, torch.zeros(len(q), device='cuda', dtype=torch.long), reduction='none')
                else:
                    (ids, types) = pair(np.repeat(q, 21, axis=0), p.reshape(-1, p.shape[-1]))
                    v = model(torch.as_tensor(ids, device='cuda'), types=torch.as_tensor(types, device='cuda')).reshape(len(q), 21)
                    losses = F.softplus(v[:, 1:] - v[:, :1]).mean(1)
                loss = (losses * torch.tensor(weights, device='cuda')).sum() / len(ei)
                assert torch.isfinite(loss)
                loss.backward()
                lossvalue += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp('grad_clip'))
            fraction = (optimstep + 1) / total
            lr = a.lr * min(fraction / max(hp('warmup_fraction'), 1e-12), max(0.0, (1 - fraction) / (1 - hp('warmup_fraction'))))
            for group in opt.param_groups:
                group['lr'] = lr
            opt.step()
            optimstep += 1
            loss_values.append(lossvalue)
        if epoch % hp('eval_every_epochs') == 0 or epoch == a.epochs:
            preds = retrieve_many({s: rows[s] for s in ['valid']}, 'evaluation', None) if kind == 'be' else {s: rerank(rows[s], base[s], f'eval_{s}') for s in ['valid']}
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                values = list(pool.map(lambda s: metrics(preds[s], rows[s]), ['valid']))
            record = dict(epoch=epoch, optimizer_step=optimstep, train_loss=float(np.mean(loss_values)), valid=values[0])
            history.append(record)
            write(D / 'history.json', history)
            for split in ['valid']:
                score = key(kind, record[split])
                dest = D / f'selected_{split}'
                dest.mkdir(exist_ok=True)
                if split not in best or score > best[split]:
                    best[split] = score
                    torch.save(model.state_dict(), dest / 'best.pt.tmp')
                    (dest / 'best.pt.tmp').replace(dest / 'best.pt')
                    write(dest / 'selection.json', dict(record, selection_split=split, default_enabled=True))
                    for s in preds:
                        np.savez_compressed(dest / f'{s}.npz', top=preds[s][:, :1024], full=preds[s])
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch, optimizer_step=optimstep, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), D / 'last.pt.tmp')
            (D / 'last.pt.tmp').replace(D / 'last.pt')
            write(D / 'history.json', history)
        if kind == 'be' and epoch < a.epochs:
            hard = retrieve_many({'train': train}, 'mine_hard_negatives', hp('hard_pool_k'))['train']
        model.train()
    model.load_state_dict(torch.load(D / 'selected_valid/best.pt', map_location='cpu', weights_only=True))
    if kind == 'be':
        test_pred = retrieve_many({'test': rows['test']}, 'final_test', None)['test']
    else:
        test_base = np.load(be_source / 'test.npz')
        base['test'] = test_base['full'] if 'full' in test_base else test_base['top']
        test_pred = rerank(rows['test'], base['test'], 'final_test')
    np.savez_compressed(D / 'selected_valid/test.npz', top=test_pred[:, :1024], full=test_pred)
    write(D / 'test_metrics.json', metrics(test_pred, rows['test']))
    if kind == 'be':
        mined = {}
        for split in ['valid']:
            dest = D / f'selected_{split}'
            h = sha(dest / 'best.pt')
            if h in mined:
                shutil.copyfile(mined[h], dest / 'mined.npz')
            else:
                model.load_state_dict(torch.load(dest / 'best.pt', map_location='cpu', weights_only=True))
                top = retrieve_many({'train': train}, f'mine_selected_{split}', hp('hard_pool_k'))['train']
                np.savez_compressed(dest / 'mined.npz', top=top)
                mined[h] = dest / 'mined.npz'
if __name__ == '__main__':
    main()
