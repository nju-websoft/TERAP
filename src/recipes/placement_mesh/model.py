import torch
from torch import nn

class CARE(nn.Module):

    def __init__(self, cfg, head):
        super().__init__()
        d = cfg['dim']
        layers = cfg['layers']
        heads = cfg['heads']
        ff = cfg['ff']
        drop = cfg.get('dropout', 0.1)
        self.score = nn.Linear(d, 1)
        if d == 768:
            self.score.load_state_dict(head)
        self.qproj = nn.Identity() if d == 768 else nn.Linear(768, d)
        self.mproj = nn.Linear(768, d)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d)) for _ in range(layers)])
        self.fn = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])

    def forward(self, c, m, roles):
        h = self.qproj(c)
        mm = self.mproj(m)
        mask = roles.ne(3)
        for (i, (n, ff, fn)) in enumerate(zip(self.norm, self.ff, self.fn)):
            (v, _) = self.attn[i](h, mm, mm, key_padding_mask=~mask, need_weights=False)
            h = n(h + v)
            h = fn(h + ff(h))
        return self.score(h).squeeze(-1)

def loss_fn(s, g, cfg, hard2, hard8):
    listwise = (torch.logsumexp(s, 1) - torch.logsumexp(s.masked_fill(~g, -torch.inf), 1)).mean()
    kind = cfg.get('pair', 'none')
    if kind == 'none':
        return (listwise, listwise, s.new_zeros(()))
    mask = hard8
    mask = mask & ~g
    sg = s.masked_fill(~g, -torch.inf).max(1).values
    delta = s - sg[:, None]
    raw = torch.nn.functional.softplus(delta)
    counts = mask.sum(1)
    valid = counts > 0
    per = (raw * mask).sum(1) / counts.clamp_min(1)
    pair = per[valid].mean() if valid.any() else s.sum() * 0
    return (listwise + cfg.get('lam', 1.0) * pair, listwise, pair)
