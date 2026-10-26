"""IEEE CE training and configurable frozen retrieval."""
import argparse, math, collections
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from .common import *
from ...configuration import parameter as hp, inputs_for_recipe
from .be import Retriever

class CE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE, local_files_only=True, add_pooling_layer=False, attn_implementation='sdpa')
        self.classifier = nn.Linear(768, 1)

    def forward(self, ids, types):
        length = int(ids.ne(0).sum(1).max())
        ids = ids[:, :length]
        types = types[:, :length]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not getattr(self, 'inference_fp32', False)):
            h = self.encoder(input_ids=ids, attention_mask=ids.ne(0), token_type_ids=types).last_hidden_state[:, 0]
        return self.classifier(h.float()).squeeze(-1)

def pair(qt, pt):
    ids = np.zeros((len(qt), 356), np.int64)
    types = np.zeros_like(ids)
    for (i, (q, p)) in enumerate(zip(qt, pt)):
        q = q[q != 0]
        p = p[p != 0][1:]
        n = len(q) + len(p)
        assert n <= 356
        ids[i, :len(q)] = q
        ids[i, len(q):n] = p
        types[i, len(q):n] = 1
    return (torch.as_tensor(ids, device='cuda'), torch.as_tensor(types, device='cuda'))

@torch.inference_mode()
def encode(net, t, role):
    out = []
    for lo in range(0, len(t), 128):
        out.append(net.encode(torch.as_tensor(t[lo:lo + 128], device='cuda', dtype=torch.long), role).cpu().numpy())
    return np.concatenate(out)

def mining():
    torch.set_num_threads(2)
    net = Retriever().cuda().eval()
    net.load_state_dict(torch.load(SRC / 'be/best.pt', map_location='cpu', weights_only=True))
    g = read(DATA / 'be_graph.json')
    qt = np.load(DATA / 'be_query_tokens.npy')
    pt = np.load(DATA / 'S23_parent_tokens.npy')
    bank = np.array(g['bank'])
    pv = encode(net, pt[bank], 'parent')
    qs = g['train'] + g['valid']
    qv = encode(net, qt[qs], 'query')
    scores = qv @ pv.T
    tops = {}
    valid_queries = set(g['valid'])
    for (i, q) in enumerate(qs):
        order = bank[np.argsort(-scores[i], kind='stable')]
        tops[str(q)] = order[order != q][:1024 if q in valid_queries else hp('hard_pool_k')].tolist()
    write(OUT / 'ce/mining.json', tops)

def train():
    torch.set_num_threads(2)
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group('nccl')
    torch.manual_seed(hp('seed'))
    g = read(DATA / 'be_graph.json')
    tops = read(OUT / 'ce/mining.json')
    qt = np.load(DATA / 'be_query_tokens.npy')
    pt = np.load(DATA / 'S23_parent_tokens.npy')
    bank = set(g['bank'])
    parents = g['parents']
    children = [[] for _ in parents]
    for (c, pp) in enumerate(parents):
        for p in pp:
            children[p].append(c)

    def closure(nodes, links):
        todo = list(nodes)
        seen = set()
        while todo:
            n = todo.pop()
            for j in links[n]:
                if j not in seen:
                    seen.add(j)
                    todo.append(j)
        return seen

    def negative(q, rng):
        gold = set(parents[q])
        allowed = bank - gold - {q}
        out = []
        groups = [(tops[str(q)], 10), ({s for p in gold for gp in parents[p] for s in children[gp]}, 4), (closure(gold, parents), 2), (closure(gold, children), 2)]
        for (group, n) in groups:
            options = sorted(set(group) & allowed - set(out))
            rng.shuffle(options)
            out.extend(options[:n])
        rest = sorted(allowed - set(out))
        out.extend(rng.choice(rest, 20 - len(out), replace=False).tolist())
        assert not set(out) & gold
        return out
    net = CE().cuda()
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank]) if world > 1 else net
    opt = torch.optim.AdamW(model.parameters(), lr=hp('lr'), weight_decay=hp('weight_decay'))
    edges = [(q, p, 1 / len(parents[q])) for q in g['train'] for p in parents[q]]
    local = hp('micro')
    globalbatch = local * world
    steps = math.ceil(len(edges) / globalbatch)
    epochs = hp('epochs')
    history = []
    best = (-1.0, -1.0, -1.0)
    if rank == 0:
        logdir = OUT / 'runs/retriever_CE'
        logdir.mkdir(parents=True, exist_ok=True)
        write(logdir / 'config.json', dict(seed=hp('seed'), stage='IEEE2022 CE pretraining', epochs=epochs, steps=epochs * steps, global_query_batch=globalbatch, learning_rate=hp('lr'), selection='validation H32, R32, MRR1024; earliest tie'))

    @torch.inference_mode()
    def evaluate(epoch):
        net.eval()
        sums = np.zeros(5)
        mine = g['valid'][rank::world]
        for (i, q) in enumerate(mine):
            top = tops[str(q)]
            z = []
            for lo in range(0, len(top), 128):
                ps = top[lo:lo + 128]
                z.extend(net(*pair(np.repeat(qt[q:q + 1], len(ps), axis=0), pt[ps])).cpu().tolist())
            order = np.array(top)[np.lexsort((top, -np.array(z)))]
            m = retrieval_metrics(order, parents[q])
            sums += np.array([m[k] for k in ['H32', 'R32', 'MRR1024', 'H1024', 'R1024']])
        sums = torch.tensor(sums, device='cuda')
        if world > 1:
            dist.all_reduce(sums)
        return dict(zip(['H32', 'R32', 'MRR1024', 'H1024', 'R1024'], (sums.cpu().numpy() / len(g['valid'])).tolist()))
    for epoch in range(epochs):
        rng = np.random.default_rng(hp('seed') + epoch)
        order = np.resize(rng.permutation(len(edges)), steps * globalbatch)
        model.train()
        for step in range(steps):
            idx = order[step * globalbatch + rank * local:step * globalbatch + (rank + 1) * local]
            qs = []
            ps = []
            weights = []
            for j in idx:
                (q, p, w) = edges[j]
                qs.extend([q] * 21)
                ps.extend([p] + negative(q, rng))
                weights.append(w)
            opt.zero_grad(set_to_none=True)
            z = model(*pair(qt[qs], pt[ps])).reshape(local, 21)
            loss = (F.softplus(z[:, 1:] - z[:, :1]).mean(1) * torch.tensor(weights, device='cuda')).mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp('grad_clip'))
            frac = (epoch * steps + step + 1) / (epochs * steps)
            for group in opt.param_groups:
                group['lr'] = hp('lr') * min(frac / max(hp('warmup_fraction'), 1e-12), max(0.0, (1 - frac) / (1 - hp('warmup_fraction'))))
            opt.step()
        if (epoch + 1) % hp('eval_every_epochs') != 0 and epoch + 1 != epochs:
            continue
        metric = evaluate(epoch + 1)
        if rank == 0:
            history.append(dict(epoch=epoch + 1, validation=metric))
            write(OUT / 'ce/history.json', history)
            if retrieval_key('ce', metric) > best:
                best = retrieval_key('ce', metric)
                torch.save(net.state_dict(), OUT / 'ce/best.pt')
                write(OUT / 'ce/selection.json', history[-1])
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()

@torch.inference_mode()
def retrieve(shard):
    cfg = inputs_for_recipe('training')
    be_k = cfg['be_k']
    ce_k = cfg['ce_k']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = read(DATA / 'rows.json')
    qt = np.load(DATA / 'query_tokens.npy')
    ix = np.array_split(np.arange(len(rows)), 4)[shard]
    out = OUT / 'retrieval'
    out.mkdir(exist_ok=True)
    parents = {w: np.load(DATA / f'{w}_parent_tokens.npy') for w in ['S23', 'S24', 'S25']}
    be = Retriever().cuda().eval()
    be.load_state_dict(torch.load(SRC / 'be/best.pt', map_location='cpu', weights_only=True))
    qv = encode(be, qt[ix], 'query')
    top = np.empty((len(ix), be_k), np.int32)
    for (w, pt) in parents.items():
        loc = np.array([j for (j, i) in enumerate(ix) if rows[i]['window'] == w], int)
        if not len(loc):
            continue
        pv = encode(be, pt, 'parent')
        scores = qv[loc] @ pv.T
        top[loc] = np.argsort(-scores, axis=1, kind='stable')[:, :be_k]
    del be
    torch.cuda.empty_cache()
    ce = CE().cuda().eval()
    ce.inference_fp32 = True
    ce.load_state_dict(torch.load(OUT / 'ce/best.pt', map_location='cpu', weights_only=True))
    ranked = np.empty((len(ix), ce_k), np.int32)
    scores32 = np.empty_like(ranked, np.float32)
    batch = hp('batch_size')
    for (j, i) in enumerate(ix):
        pt = parents[rows[i]['window']]
        (ids, types) = pair(np.repeat(qt[i:i + 1], be_k, axis=0), pt[top[j]])
        z = torch.cat([ce(ids[lo:lo + batch], types[lo:lo + batch]) for lo in range(0, be_k, batch)]).cpu().numpy()
        order = np.lexsort((top[j], -z))[:ce_k]
        ranked[j] = top[j, order]
        scores32[j] = z[order]
    np.savez(out / f'{shard}.npz', indices=ix, top=ranked, scores=scores32, be_top=top)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['mining', 'train', 'retrieve'])
    p.add_argument('--shard', type=int, default=0)
    a = p.parse_args()
    {'mining': mining, 'train': train, 'retrieve': lambda : retrieve(a.shard)}[a.stage]()
