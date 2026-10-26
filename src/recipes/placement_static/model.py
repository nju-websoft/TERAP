"""Cross-attention Placement model."""
import torch
from torch import nn
from torch.nn import functional as F

class CARE(nn.Module):

    def __init__(self, cfg, head):
        super().__init__()
        self.cfg = cfg
        d = cfg['dim']
        drop = cfg['dropout']
        self.score = nn.Linear(d, 1)
        if d == 768:
            self.score.load_state_dict(head)
        self.qproj = nn.Identity() if d == 768 else nn.Linear(768, d)
        self.mproj = nn.Linear(768, d)
        self.qnorm = nn.Identity()
        self.mnorm = nn.Identity()
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, cfg['heads'], dropout=drop, batch_first=True) for _ in range(cfg['layers'])])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg['layers'])])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, cfg['ff']), nn.GELU(), nn.Dropout(drop), nn.Linear(cfg['ff'], d)) for _ in range(cfg['layers'])])
        self.fn = nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg['layers'])])
        self.register_buffer('ce_weight', head['weight'].clone())
        self.register_buffer('ce_bias', head['bias'].clone())

    def forward(self, c, m, roles):
        h = self.qnorm(self.qproj(c))
        slots = self.cfg['memory_slots']
        m = m[:, :slots]
        roles = roles[:, :slots]
        mask = roles.ne(3)
        assert mask.any(1).all()
        mm = self.mnorm(self.mproj(m))
        for (i, (norm, ff, fn)) in enumerate(zip(self.norm, self.ff, self.fn)):
            (v, w) = self.attn[i](h, mm, mm, key_padding_mask=~mask, need_weights=False, average_attn_weights=False)
            h = norm(h + 1.0 * v)
            h = fn(h + ff(h))
        delta = self.score(h).squeeze(-1)
        score = delta
        return score

def loss_fn(s, g, cfg, hard2, hard8):
    listwise = (torch.logsumexp(s, 1) - torch.logsumexp(s.masked_fill(~g, -torch.inf), 1)).mean()
    if cfg['pair'] == 'none':
        return (listwise, listwise, s.new_zeros(()))
    mask = hard8
    mask = mask & ~g
    sg = s.masked_fill(~g, -torch.inf).max(1).values
    raw = F.softplus(s - sg[:, None])
    counts = mask.sum(1)
    valid = counts > 0
    per = (raw * mask).sum(1) / counts.clamp_min(1)
    pair = per[valid].mean() if valid.any() else s.sum() * 0
    return (listwise + cfg['lam'] * pair, listwise, pair)
