"""Query-only and memory-enhanced admission heads on BERT pooled features."""
import torch
from torch import nn
from torch.nn import functional as F

class AdmissionHead(nn.Module):

    def __init__(self, variant='true', d=256, depth=2, use_role=True, memory_slots=None, heads=8, attn_scale=1.0):
        super().__init__()
        self.attn_scale = attn_scale
        self.model_config = dict(variant=variant, d=d, depth=depth, use_role=use_role, memory_slots=memory_slots, heads=heads, attn_scale=attn_scale)
        self.use_role = use_role
        self.variant = variant
        self.d = d
        if variant not in ('true', 'natural_linear'):
            raise ValueError('Unsupported Admission model.')
        self.natural = variant == 'natural_linear'
        if self.natural:
            self.head = nn.Linear(768, 1)
            return
        self.qproj = nn.Linear(d, d)
        self.mproj = nn.Linear(d, d)
        self.role = nn.Embedding(4, d)
        self.layers = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True) for _ in range(depth)])
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
        return self.mproj(F.normalize(self.encm(m), dim=-1)) + (self.role(roles) if self.use_role else 0)

    def forward(self, q, m=None, roles=None):
        if self.natural:
            return self.head(q).squeeze(-1)
        q = self.project_query(q)
        mask = roles.ne(3)
        m = self.project_memory(m, roles)
        h = q[:, None]
        for (attn, norm, ff, fn) in zip(self.layers, self.norms, self.ff, self.ffnorm):
            (a, _) = attn(h * self.attn_scale, m, m, key_padding_mask=~mask, need_weights=False)
            h = norm(h + a)
            h = fn(h + ff(h))
        h = h[:, 0]
        return self.head(torch.cat([q, h, q - h, q * h], -1)).squeeze(-1)

class AdmissionModel(nn.Module):
    """Shared token encoder plus either a Query-only or CARE classification head."""

    def __init__(self, encoder, head, path_chunk=64):
        super().__init__()
        (self.encoder, self.head, self.path_chunk) = (encoder, head, path_chunk)

    def forward(self, query, memory=None, roles=None):
        if not any((p.requires_grad for p in self.encoder.parameters())):
            self.encoder.eval()
        q = self.encoder(query)
        if self.head.natural:
            return self.head(q)
        if memory is None or roles is None or (not roles.ne(3).any(1).all()):
            raise ValueError('CARE requires at least one valid memory path per query.')
        valid = roles.ne(3)
        (unique, inverse) = torch.unique(memory[valid], dim=0, return_inverse=True)
        vectors = torch.cat([self.encoder(unique[i:i + self.path_chunk]) for i in range(0, len(unique), self.path_chunk)])
        m = vectors.new_zeros((*roles.shape, vectors.shape[-1]))
        m[valid] = vectors[inverse]
        return self.head(q, m, roles)
