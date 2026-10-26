from ...configuration import inputs_for_recipe
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from .common import BASE

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE, local_files_only=True, add_pooling_layer=False, attn_implementation='sdpa')
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False, 'preserve_rng_state': True})

    def forward(self, tokens):
        tokens = tokens[:, :int(tokens.ne(0).sum(1).max())]
        mask = tokens.ne(0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = self.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
        return (h.float() * mask[:, :, None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]

class Head(nn.Module):

    def __init__(self, variant, d=256, depth=2):
        super().__init__()
        self.variant = variant
        self.d = d
        self.natural = False
        self.qproj = nn.Linear(d, d)
        self.mproj = nn.Linear(d, d)
        self.role = nn.Embedding(4, d)
        self.layers = nn.ModuleList([nn.MultiheadAttention(d, 8, dropout=0.1, batch_first=True) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d, d)) for _ in range(depth)])
        self.ffnorm = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))
        self.encq = nn.Linear(768, d, bias=False)
        self.encm = nn.Linear(768, d, bias=False)
        with torch.no_grad():
            self.encq.weight.copy_(torch.eye(d, 768))
            self.encm.weight.copy_(torch.eye(d, 768))

    def project_query(self, q):
        return self.qproj(F.normalize(self.encq(q), dim=-1))

    def project_memory(self, m, roles):
        return self.mproj(F.normalize(self.encm(m), dim=-1)) + self.role(roles)

    def forward(self, q, m=None, roles=None):
        q = self.project_query(q)
        mask = roles.ne(3)
        m = self.project_memory(m, roles)
        h = q[:, None]
        for (attn, norm, ff, fn) in zip(self.layers, self.norms, self.ff, self.ffnorm):
            (a, _) = attn(h, m, m, key_padding_mask=~mask, need_weights=False)
            h = norm(h + a)
            h = fn(h + ff(h))
        h = h[:, 0]
        return self.head(torch.cat([q, h, q - h, q * h], -1)).squeeze(-1)
