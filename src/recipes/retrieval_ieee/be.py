"""Multi-parent bi-encoder training on the IEEE 2022 graph."""
import argparse, os, math
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.distributed.nn.functional import all_gather
from .common import *
from ...configuration import parameter as hp, inputs_for_recipe
from .model import Encoder

class Retriever(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = Encoder()
        self.query = nn.Linear(768, 256)
        self.parent = nn.Linear(768, 256)

    def encode(self, t, role):
        return F.normalize(getattr(self, role)(self.base(t)), dim=-1)

    def forward(self, q, p, positive, negative):
        q = self.encode(q, 'query')
        p = self.encode(p, 'parent')
        if dist.is_initialized():
            p = torch.cat(all_gather(p))
        score = q.float() @ p.float().T / 0.05
        ln = torch.logsumexp(score.masked_fill(~negative, -torch.inf), 1)
        assert positive.any(1).all() and negative.any(1).all()
        return ((F.softplus(ln[:, None] - score) * positive).sum(1) / positive.sum(1)).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=hp('epochs'))
    a = ap.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group('nccl')
    torch.manual_seed(hp('seed'))
    g = read(DATA / 'be_graph.json')
    qt = np.load(DATA / 'be_query_tokens.npy')
    pt = np.load(DATA / 'S23_parent_tokens.npy')
    net = Retriever().cuda()
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank]) if world > 1 else net
    opt = torch.optim.AdamW(model.parameters(), lr=hp('lr'), weight_decay=hp('weight_decay'))
    out = OUT / 'be'
    out.mkdir(exist_ok=True)
    batch = hp('batch_size')
    local = batch // world
    accum = hp('accumulation')
    steps = math.ceil(len(g['train']) / batch)
    history = []
    best = (-1.0, -1.0, -1.0)
    done = 0

    @torch.inference_mode()
    def evaluate():
        net.eval()
        vectors = {}
        for (role, t, ids) in [('parent', pt, g['bank']), ('query', qt, g['valid'])]:
            vs = []
            for lo in range(0, len(ids), 256):
                jj = ids[lo:lo + 256]
                vs.append(net.encode(torch.as_tensor(t[jj], device='cuda', dtype=torch.long), role).cpu().numpy())
            vectors[role] = np.concatenate(vs)
        scores = vectors['query'] @ vectors['parent'].T
        order = np.argsort(-scores, axis=1, kind='stable')
        bank = np.array(g['bank'])
        result = {}
        for k in [1, 10, 50, 100, 200]:
            result[f'Hit@{k}'] = float(np.mean([bool(set(bank[o[:k]]) & set(g['parents'][q])) for (o, q) in zip(order, g['valid'])]))
        ranked = [retrieval_metrics(bank[o], g['parents'][q]) for (o, q) in zip(order, g['valid'])]
        result.update({k: float(np.mean([m[k] for m in ranked])) for k in ranked[0]})
        return result
    for epoch in range(a.epochs):
        rng = np.random.default_rng(2000 + epoch)
        perm = rng.permutation(g['train'])
        perm = np.resize(perm, steps * batch)
        model.train()
        opt.zero_grad(set_to_none=True)
        for j in range(steps):
            qs = perm[j * batch:(j + 1) * batch]
            ps = []
            for q in qs:
                ps.append(int(rng.choice(g['parents'][q])))
                ps.extend(rng.choice(g['hard'][q], 4, replace=False).tolist())
            ps = np.array(ps)
            uq = np.r_[True, ps[1:] != ps[:-1]]
            seen = set()
            unique = np.array([not (p in seen or seen.add(int(p))) for p in ps], bool)
            mine = qs[rank * local:(rank + 1) * local]
            pm = ps[rank * local * 5:(rank + 1) * local * 5]
            pos = np.array([[p in g['parents'][q] for p in ps] for q in mine]) & unique[None]
            neg = ~np.array([[p in g['parents'][q] or p == q for p in ps] for q in mine]) & unique[None]
            loss = model(torch.as_tensor(qt[mine], device='cuda', dtype=torch.long), torch.as_tensor(pt[pm], device='cuda', dtype=torch.long), torch.as_tensor(pos, device='cuda'), torch.as_tensor(neg, device='cuda'))
            assert torch.isfinite(loss)
            group = min(accum, steps - j // accum * accum)
            (loss / group).backward()
            if (j + 1) % accum == 0 or j + 1 == steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp('grad_clip'))
                opt.step()
                opt.zero_grad(set_to_none=True)
                done += 1
        if world > 1:
            dist.barrier()
        if rank == 0 and ((epoch + 1) % hp('eval_every_epochs') == 0 or epoch + 1 == a.epochs):
            m = evaluate()
            history.append(dict(epoch=epoch + 1, optimizer_step=done, validation=m))
            write(out / 'history.json', history)
            if retrieval_key('be', m) > best:
                best = retrieval_key('be', m)
                torch.save(net.state_dict(), out / 'best.pt')
                write(out / 'selection.json', history[-1])
            print(history[-1], flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
