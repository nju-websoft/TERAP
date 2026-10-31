from ..cache import load_placement_cache
"""Train the MeSH Placement model."""
import sys, math, io, copy
import numpy as np
import torch
from .common import *
from ...configuration import parameter as hp, inputs_for_recipe
from .model import CARE, loss_fn
from .optimizers import build

def run(spec):
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    seed = spec['seed']
    name = spec['name']
    dest = O / 'runs' / name
    dest.mkdir(parents=True, exist_ok=True)
    cache = CACHE_ROOT / f's{seed}'
    (rows, counts, z, features, bank) = load_placement_cache(cache)
    (n, nv, nt) = counts
    (memory, roles, gold, hard) = (z[k] for k in ('memory', 'roles', 'gold', 'hard8'))
    bounds = {'valid': (n, n + nv)}
    training = np.arange(n)
    cehead = torch.load(cache / 'CE_head.pt', map_location='cpu', weights_only=True)
    c = torch.as_tensor(features, device='cuda')
    mb = torch.as_tensor(bank, device='cuda')
    mi = torch.as_tensor(memory, device='cuda', dtype=torch.long)
    rr = torch.as_tensor(roles, device='cuda')
    gg = torch.as_tensor(gold, device='cuda')
    hh = torch.as_tensor(hard, device='cuda')
    del features, bank
    assert gold[training].any(1).all()
    sizes = inputs_for_recipe('training')
    if roles.shape[1] != sizes['slots'] or gold.shape[1] != sizes['ce_k'] or (roles != 3).sum(1).max() > sizes['max_paths']:
        raise ValueError('Cache sizes differ from config; regenerate the cache')
    sets = {s: np.arange(lo, hi) for (s, (lo, hi)) in bounds.items()}
    cfg = dict(CONFIGS['default'], dropout=hp('dropout'), lr=spec['lr'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed + 817)
    cfg['memory_slots'] = inputs_for_recipe('training')['slots']
    model = CARE(cfg, cehead).cuda()
    opt = build(model, spec['optimizer'], spec['lr'])
    steps_epoch = len(training) // hp('batch_size')
    epochs = spec['epochs']
    total = epochs * steps_epoch
    step = 0
    history = []
    best = {}
    loss_total = 0.0
    cfg.update(spec, model=name, optimizer_total=total, steps_per_epoch=steps_epoch, selection='validation H1 then MRR32; earliest tie; step0 excluded')
    write(dest / 'config.json', cfg)

    @torch.inference_mode()
    def infer(ii):
        model.eval()
        out = []
        for lo in range(0, len(ii), 64):
            ix = torch.as_tensor(ii[lo:lo + 64], device='cuda')
            jx = ix
            out.append(model(c[ix], mb[mi[jx]], rr[jx]).cpu().numpy())
        return np.concatenate(out)

    def checkpoint(path):
        state = {'model': model.state_dict(), 'optimizer_step': step, 'spec': spec}
        buffer = io.BytesIO()
        torch.save(state, buffer)
        payload = buffer.getvalue()
        path.parent.mkdir(exist_ok=True, parents=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(payload)
        tmp.replace(path)

    def evaluate():
        pred = {'valid': infer(sets['valid'])}
        values = metric(pred['valid'], gold[sets['valid']], [rows[i] for i in sets['valid']])
        rec = dict(optimizer_step=step, epoch=step / steps_epoch, metrics={'valid': values})
        key = 'valid'
        value = (values['H1'], values['MRR32'])
        if step and (key not in best or value > tuple(best[key]['key'])):
            best[key] = dict(key=value, **rec)
            branch = dest / 'selected_valid'
            branch.mkdir(exist_ok=True)
            write(branch / 'selection.json', best[key])
            np.savez_compressed(branch / 'predictions.npz', **pred)
            checkpoint(branch / 'best.pt')
        history.append(rec)
        write(dest / 'history.json', history)
    evaluate()
    integer_epochs = set()
    while step < total:
        epoch = step // steps_epoch
        cursor = step % steps_epoch * hp('batch_size')
        order = np.random.default_rng(seed * 10000 + epoch + 1).permutation(training)
        ii = torch.as_tensor(order[cursor:cursor + hp('batch_size')], device='cuda')
        factor = 1.0
        if spec.get('schedule') in ['cosine', 'linear']:
            f = (step + 1) / total
            warm = hp('warmup')
            factor = f / warm if f < warm else hp('min_lr_factor') + (1 - hp('min_lr_factor')) * (1 + math.cos(math.pi * (f - warm) / (1 - warm))) / 2 if spec['schedule'] == 'cosine' else max(hp('min_lr_factor'), (1 - f) / (1 - warm))
        groups = opt.param_groups if hasattr(opt, 'param_groups') else opt.muon.param_groups + opt.adamw.param_groups
        for group in groups:
            group.setdefault('initial_lr', group['lr'])
            group['lr'] = group['initial_lr'] * factor
            group['weight_decay'] = hp('wd')
        model.train()
        opt.zero_grad(set_to_none=True)
        perm = torch.as_tensor(np.stack([rng.permutation(gold.shape[1]) for _ in range(hp('batch_size'))]), device='cuda')
        micro = hp('micro')
        loss_total = 0.0
        for lo in range(0, hp('batch_size'), micro):
            jj = ii[lo:lo + micro]
            pp = perm[lo:lo + micro]
            mem = mb[mi[jj]]
            scores = model(c[jj].gather(1, pp[:, :, None].expand(-1, -1, 768)), mem, rr[jj])
            (loss, ll, lp) = loss_fn(scores, gg[jj].gather(1, pp), cfg, hh[jj].gather(1, pp), hh[jj].gather(1, pp))
            assert torch.isfinite(loss)
            (loss * len(jj) / hp('batch_size')).backward()
            loss_total += float(loss.detach()) * len(jj) / hp('batch_size')
        params = list(model.parameters())
        torch.nn.utils.clip_grad_norm_(params, hp('grad_clip'))
        opt.step()
        step += 1
        due = step % hp('eval_every') == 0
        if due or (step % steps_epoch == 0 and step // steps_epoch in integer_epochs) or step == total:
            evaluate()
    write(dest / 'result.json', dict(selections=best))
if __name__ == '__main__':
    spec = read(sys.argv[1])
    run(spec)
