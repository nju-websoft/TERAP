import argparse, math, random
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from ...configuration import parameter as hp, inputs_for_recipe
from .model import Encoder, Head

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    ap.add_argument('--condition', default='with_weak')
    ap.add_argument('--seed', type=int, default=hp('seed'))
    ap.add_argument('--width', type=int, default=hp('width'))
    ap.add_argument('--depth', type=int, default=hp('depth'))
    ap.add_argument('--switch', type=int, default=hp('switch'))
    ap.add_argument('--steps', type=int, default=hp('steps'))
    ap.add_argument('--weak-start', type=int, default=hp('weak_start'))
    ap.add_argument('--eval-every', type=int, default=hp('eval_every'))
    ap.add_argument('--micro', type=int, default=hp('micro'))
    ap.add_argument('--chunk', type=int, default=hp('chunk'))
    ap.add_argument('--no-checkpoint', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    assert torch.cuda.device_count() == 1
    dest = OUT / 'runs' / a.key
    dest.mkdir(parents=True, exist_ok=True)
    net = Encoder().cuda()
    if a.no_checkpoint:
        net.encoder.gradient_checkpointing_disable()
    torch.manual_seed(a.seed)
    head = Head('true', a.width, a.depth).cuda()
    memory = True
    rows = read(DATA / 'rows.json')
    y = np.array([r['label'] for r in rows], np.float32)
    idx = {s: np.array([i for (i, r) in enumerate(rows) if r['split'] == s and (not (s == 'train' and a.condition == 'without_weak' and r['weak']))]) for s in ['train', 'valid']}
    pos = idx['train'][y[idx['train']] == 1]
    neg = idx['train'][y[idx['train']] == 0]
    strong = neg[np.array([not rows[i]['weak'] for i in neg])]
    qt = np.load(DATA / 'query_tokens.npy')
    pt = np.load(DATA / 'path_tokens.npy')
    plen = np.count_nonzero(pt, axis=1)
    qcache = torch.from_numpy(np.load(DATA / 'raw_query.npy')).cuda()
    bank = torch.from_numpy(np.load(DATA / 'raw_bank.npy')).cuda()
    mem = {k: torch.as_tensor(v.astype(np.int64), device='cuda') for (k, v) in np.load(DATA / f'{a.condition}_memory_{a.seed}.npz').items() if k in ('memory', 'roles')}
    if mem['memory'].shape[1] != inputs_for_recipe('training')['slots']:
        raise ValueError('Memory cache slots differ from config; regenerate the cache')
    if (mem['roles'].ne(3).sum(1) > inputs_for_recipe('training')['max_paths']).any():
        raise ValueError('Memory cache active paths exceed config; regenerate the cache')
    net.requires_grad_(False)
    head.encq.requires_grad_(False)
    head.encm.requires_grad_(False)
    opt = torch.optim.AdamW([{'params': [p for p in head.parameters() if p.requires_grad], 'lr': hp('head_lr')}], weight_decay=hp('weight_decay'))
    online = False
    rng = np.random.default_rng(a.seed)

    def unfreeze():
        nonlocal online
        net.requires_grad_(True)
        opt.add_param_group({'params': list(net.parameters()), 'lr': hp('encoder_lr')})
        head.encq.requires_grad_(True)
        head.encm.requires_grad_(True)
        opt.add_param_group({'params': [p for mod in [head.encq, head.encm] for p in mod.parameters() if p.requires_grad], 'lr': hp('projection_lr')})
        online = True

    def encode_training_memory(ids, roles):
        valid = roles.ne(3)
        (unique, inv) = torch.unique(ids[valid], return_inverse=True)
        ui = unique.cpu().numpy()
        order = np.argsort(plen[ui], kind='stable')
        ordered = ui[order]
        vec = []
        for lo in range(0, len(ordered), a.chunk):
            vec.append(net(torch.as_tensor(pt[ordered[lo:lo + a.chunk]], device='cuda', dtype=torch.long)))
        vectors = torch.cat(vec)[np.argsort(order)]
        out = torch.zeros((*ids.shape, 768), device='cuda')
        out[valid] = vectors[inv]
        return out

    def update(ix):
        head.train()
        net.train(online)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for lo in range(0, len(ix), a.micro):
            jj = ix[lo:lo + a.micro]
            q = net(torch.as_tensor(qt[jj], device='cuda', dtype=torch.long)) if online else qcache[jj]
            m = roles = None
            ids = mem['memory'][jj]
            roles = mem['roles'][jj]
            m = encode_training_memory(ids, roles) if online else bank[ids]
            z = head(q, m, roles)
            loss = F.binary_cross_entropy_with_logits(z, torch.as_tensor(y[jj], device='cuda'))
            assert torch.isfinite(loss)
            (loss * len(jj) / len(ix)).backward()
            loss_sum += float(loss) * len(jj) / len(ix)
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], hp('grad_clip'))
        opt.step()
        return (loss_sum, float(gn))

    @torch.inference_mode()
    def enc_eval(tokens, ids, role):
        net.eval()
        lens = np.count_nonzero(tokens[ids], axis=1)
        order = np.argsort(lens, kind='stable')
        ordered = ids[order]
        out = []
        for lo in range(0, len(ids), 256):
            jj = ordered[lo:lo + 256]
            out.append(net(torch.as_tensor(tokens[jj], device='cuda', dtype=torch.long)))
        return torch.cat(out)[np.argsort(order)]

    @torch.inference_mode()
    def refresh():
        if not online:
            return
        ix = idx['valid']
        qcache[ix] = enc_eval(qt, ix, 'query')
        used = torch.unique(mem['memory'][ix]).cpu().numpy()
        bank[used] = enc_eval(pt, used, 'memory')

    @torch.inference_mode()
    def predict(ix):
        head.eval()
        out = []
        for lo in range(0, len(ix), 128):
            jj = ix[lo:lo + 128]
            m = roles = None
            m = bank[mem['memory'][jj]]
            roles = mem['roles'][jj]
            out.append(head(qcache[jj], m, roles).cpu().numpy())
        return np.concatenate(out)
    config = dict(vars(a), effective_batch=hp('batch_size'), head_lr=hp('head_lr'), encoder_lr=hp('encoder_lr'), projection_lr=hp('projection_lr'), train_new=len(pos), train_D=len(neg), train_strong_D=len(strong), initial_encoder='base BERT; no placement/admission fine-tuning', shared_encoder=True, query_memory_encoder_frozen_until=a.switch, retrieval='frozen BE+CE; prepared memory cache', memory=inputs_for_recipe('training'), selection='clean S25 pAUROC; encoder export restricted to jointly fine-tuned checkpoints')
    write(dest / 'config.json', config)
    history = []
    best = -1
    best_ck = None
    step_done = 0

    def snapshot():
        return dict(head={k: v.detach().cpu().clone() for (k, v) in head.state_dict().items()}, encoder={k: v.detach().cpu().clone() for (k, v) in net.state_dict().items()}, optimizer_step=step_done, online=online)

    def state():
        return dict(**snapshot(), optimizer=opt.state_dict(), scheduler=None, step=step_done, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate(), history=history, best=best)

    def save_ck(path, value):
        tmp = path.with_suffix('.tmp')
        torch.save(value, tmp)
        tmp.replace(path)

    def validate(step, final=False):
        nonlocal best, best_ck
        refresh()
        ix = idx['valid']
        z = predict(ix)
        m = metrics(z, y[ix])
        m['loss'] = float(F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(y[ix])))
        arrays = {'logit': z, 'admission': 1 / (1 + np.exp(-np.clip(z, -80, 80)))}
        allout = {'valid': dict(metrics=m, arrays=arrays)}
        item = dict(optimizer_step=step, phase='joint' if online else 'frozen', valid=m)
        history.append(item)
        write(dest / 'history.json', history)
        if online and (not final) and (item['valid']['pAUROC'] > best):
            best = item['valid']['pAUROC']
            best_ck = dict(head={k: v.detach().cpu().clone() for (k, v) in head.state_dict().items()}, encoder={k: v.detach().cpu().clone() for (k, v) in net.state_dict().items()} if online else None, optimizer_step=step, online=online)
            write(dest / 'selection.json', item)
            save_ck(dest / 'best.pt', best_ck)
            for (split, v) in allout.items():
                np.savez(dest / f'{split}.npz', **v['arrays'])
        save_ck(dest / 'last.pt', state())
    for step in range(a.steps):
        if step == a.switch:
            unfreeze()
        ix = np.where(rng.integers(0, 2, hp('batch_size')), rng.choice(pos, hp('batch_size')), rng.choice(strong if step < a.weak_start else neg, hp('batch_size')))
        (loss, gn) = update(ix)
        step_done = step + 1
        if step_done % a.eval_every == 0 or step_done == a.steps:
            validate(step_done)
    assert best_ck is not None
    torch.save(best_ck, dest / 'best.pt')
    save_ck(dest / 'last.pt', state())
    result = {s: metrics(np.load(dest / f'{s}.npz')['logit'], yy) for (s, yy) in [('valid', y[idx['valid']])]}
    result['selection'] = read(dest / 'selection.json')
    write(dest / 'metrics.json', result)
if __name__ == '__main__':
    main()
