from ..cache import load_placement_cache
'Whole-path CARE with validation-only checkpoint selection.'
import argparse, math
import numpy as np
import torch
from .common import *
from ...configuration import parameter as hp, inputs_for_recipe
from .model import CARE, loss_fn

def run(seed):
    variant = "default"
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    cfg = dict(CONFIGS[variant])
    cfg.update({k: hp(k) for k in ['lr', 'epochs', 'schedule', 'dropout']})
    epochs = cfg['epochs']
    schedule = cfg['schedule']
    name = f'{DATASET}_{variant}_s{seed}'
    dest = ROOT_O / 'runs' / name
    dest.mkdir(parents=True, exist_ok=True)
    cache = CACHE_ROOT / f's{seed}'
    (rows, counts, z, features, path_features) = load_placement_cache(cache)
    (n, nv, nt) = counts
    c = torch.as_tensor(features, device='cuda')
    bank = torch.as_tensor(path_features, device='cuda')
    mi = torch.as_tensor(z['memory'], device='cuda', dtype=torch.long)
    roles = torch.as_tensor(z['roles'], device='cuda')
    gold = torch.as_tensor(z['gold'], device='cuda')
    hard = torch.as_tensor(z['hard8'], device='cuda')
    assert gold[:n].any(1).all() and roles.ne(3).any(1).all()
    sizes = inputs_for_recipe('training')
    if roles.shape[1] != sizes['slots'] or gold.shape[1] != sizes['ce_k'] or (roles.ne(3).sum(1) > sizes['max_paths']).any():
        raise ValueError('Cache sizes differ from config; regenerate the cache')
    head = torch.load(cache / 'CE_head.pt', map_location='cpu', weights_only=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed + 817)
    cfg['memory_slots'] = inputs_for_recipe('training')['slots']
    model = CARE(cfg, head).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=hp('wd'))
    steps_epoch = n // hp('batch_size')
    total = epochs * steps_epoch
    step = 0
    history = []
    best = {k: (-1.0, -1.0) for k in ['valid']}
    snapshots = {}
    warmup = 0 if schedule == 'constant' else int(hp('warmup') * total)

    def lr_factor(t):
        if schedule == 'constant':
            return 1.0
        if t < warmup:
            return (t + 1) / max(1, warmup)
        return max(hp('min_lr_factor'), (total - t) / max(1, total - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    cfg.update(warmup_steps=warmup, validation_cadence='configured optimizer-step interval')
    cfg['memory_slots'] = inputs_for_recipe('training')['slots']
    cfg.update(name=name, model=f'{DATASET}_{variant}', dataset=DATASET, seed=seed, epochs=epochs, batch=hp('batch_size'), micro=hp('batch_size'), steps_per_epoch=steps_epoch, drop_last=True, optimizer_total=total, eval_every=hp('eval_every'), lr_schedule=schedule, weight_decay=hp('wd'), clip=hp('grad_clip'), role=False, BE_frozen=True, CE_frozen=True, path_frozen=True, inputs=inputs_for_recipe('training'), selection='validation H1 then MRR32, trained steps only, earliest tie')
    cfg.update(path_encoder='MeSH Admission fine-tuned, Qmesh heldout supervision removed')
    write(dest / 'config.json', cfg)
    if (dest / 'last.pt').exists():
        ck = torch.load(dest / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        step = ck['step']
        rng.bit_generator.state = ck['rng']
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
        history = ck['history']
        best = ck['best']
        snapshots = ck['snapshots']
        for (key, snap) in snapshots.items():
            save_branch(dest, key, snap)

    @torch.inference_mode()
    def infer(lo, hi):
        model.eval()
        out = []
        for pos in range(lo, hi, 64):
            ii = torch.arange(pos, min(pos + 64, hi), device='cuda')
            jj = ii
            out.append(model(c[ii], bank[mi[jj]], roles[jj]).cpu().numpy())
        return np.concatenate(out)

    def state():
        return dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), step=step, optimizer_step=step, epoch=step / steps_epoch, batch_index=step % steps_epoch, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), history=history, best=best, snapshots=snapshots)

    def evaluate():
        pred = {'valid': infer(n, n + nv)}
        values = metric(pred['valid'], z['gold'][n:n + nv], rows[n:n + nv])
        if 'valid_mrr_tail' in z:
            values['MRR'] = values['MRR32'] + float(z['valid_mrr_tail'])
        metrics = {'valid': values}
        record = dict(optimizer_step=step, epoch=step / steps_epoch, valid=values)
        key = 'valid'
        score = (values['H1'], values['MRR32'])
        if step > 0 and score > best[key]:
            best[key] = score
            snapshots[key] = dict(model={k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}, selection=dict(record, selection_split=key), scores=pred)
            save_branch(dest, key, snapshots[key])
        history.append(record)
        write(dest / 'history.json', history)
        torch.save(state(), dest / 'last.tmp.pt')
        (dest / 'last.tmp.pt').replace(dest / 'last.pt')
    if step == 0:
        evaluate()
    while step < total:
        epoch = step // steps_epoch + 1
        cursor = step % steps_epoch * hp('batch_size')
        order = np.random.default_rng(seed * 10000 + epoch).permutation(n)
        ii = torch.as_tensor(order[cursor:cursor + hp('batch_size')], device='cuda')
        perm = torch.as_tensor(np.stack([rng.permutation(gold.shape[1]) for _ in range(hp('batch_size'))]), device='cuda')
        model.train()
        opt.zero_grad(set_to_none=True)
        s = model(c[ii].gather(1, perm[:, :, None].expand(-1, -1, 768)), bank[mi[ii]], roles[ii])
        (loss, ll, lp) = loss_fn(s, gold[ii].gather(1, perm), cfg, hard[ii].gather(1, perm), hard[ii].gather(1, perm))
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp('grad_clip'))
        opt.step()
        scheduler.step()
        step += 1
        if step % hp('eval_every') == 0 or step == total:
            evaluate()
    torch.save(state(), dest / 'last.tmp.pt')
    (dest / 'last.tmp.pt').replace(dest / 'last.pt')
    result = {}
    result = {k: {'selection': v['selection']} for (k, v) in snapshots.items()}
    write(dest / 'result.json', result)

def save_branch(dest, key, snap):
    d = dest / f'selected_{key}'
    d.mkdir(exist_ok=True)
    torch.save(snap['model'], d / 'best.tmp.pt')
    (d / 'best.tmp.pt').replace(d / 'best.pt')
    write(d / 'selection.json', snap['selection'])
    for (s, p) in snap['scores'].items():
        np.save(d / f'{s}_scores.npy', p)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('seed', type=int)
    a = ap.parse_args()
    run(a.seed)
