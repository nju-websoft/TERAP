"""Explicit optimizer semantics; Muon hidden matrices + AdamW head/vectors."""
import math
import torch

class MuonWithAdamW:

    def __init__(self, model, cfg):
        muon = []
        adam = []
        self.names = {'muon': [], 'adamw': []}
        for (name, p) in model.named_parameters():
            key = 'muon' if p.ndim == 2 and (not name.startswith('score.')) else 'adamw'
            (muon if key == 'muon' else adam).append(p)
            self.names[key].append(name)
        assert muon and adam and (len({id(p) for p in muon + adam}) == len(list(model.parameters())))
        self.muon = torch.optim.Muon(muon, lr=cfg['lr'], weight_decay=cfg['wd'], momentum=cfg['momentum'], nesterov=True, ns_steps=5, adjust_lr_fn='match_rms_adamw')
        self.adamw = torch.optim.AdamW(adam, lr=cfg['lr'], weight_decay=cfg['wd'], betas=(0.9, cfg['beta2']))

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return dict(muon=self.muon.state_dict(), adamw=self.adamw.state_dict(), names=self.names)

    def load_state_dict(self, state):
        assert state['names'] == self.names
        self.muon.load_state_dict(state['muon'])
        self.adamw.load_state_dict(state['adamw'])

def build(model, cfg):
    name = cfg['optimizer']
    kw = dict(lr=cfg['lr'], weight_decay=cfg['wd'])
    if name == 'muon':
        return MuonWithAdamW(model, cfg)
    return {'adamw': torch.optim.AdamW}[name](model.parameters(), **kw, betas=(0.9, cfg['beta2']))

def lr_factor(cfg, step, total):
    if cfg['schedule'] == 'constant':
        return 1.0
    warm = int(total * cfg['warmup'])
    floor = cfg['min_lr_factor']
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    if cfg['schedule'] == 'cosine':
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    assert cfg['schedule'] == 'linear'
    return max(floor, 1 - (1 - floor) * progress)
