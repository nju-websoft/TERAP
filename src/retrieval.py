"""Bi-encoder and cross-encoder; pair() builds joint BERT inputs."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from .encoder import autocast_for

class RetrievalModel(nn.Module):

    def __init__(self, kind, model_path, dim=256):
        super().__init__()
        self.kind = kind
        self.dim = dim
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True, add_pooling_layer=False, attn_implementation='sdpa')
        if kind not in ('be', 'ce'):
            raise ValueError(kind)
        if kind == 'be':
            self.query = nn.Linear(768, dim)
            self.parent = nn.Linear(768, dim)
        else:
            self.classifier = nn.Linear(768, 1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Accept the IEEE Admission recipe's nested BERT checkpoint keys."""
        if self.kind == 'be' and any(k.startswith('base.encoder.') for k in state_dict):
            from collections import OrderedDict
            def name(key):
                return 'encoder.' + key[len('base.encoder.'):] if key.startswith('base.encoder.') else key
            converted = OrderedDict((name(k), v) for k, v in state_dict.items())
            if hasattr(state_dict, '_metadata'):
                converted._metadata = OrderedDict(
                    (('encoder' if k == 'base.encoder' else name(k)), v)
                    for k, v in state_dict._metadata.items() if k != 'base')
            state_dict = converted
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def encode(self, ids, role):
        mask = ids.ne(0)
        with autocast_for(ids):
            h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        h = (h.float() * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        return F.normalize(getattr(self, role)(h), dim=-1)

    def forward(self, q, p=None, types=None):
        if self.kind == 'be':
            return torch.einsum('bd,bkd->bk', self.encode(q, 'query'), self.encode(p.flatten(0, 1), 'parent').reshape(p.shape[0], p.shape[1], -1)) / 0.05
        return self.classifier(self.joint_features(q, types)).squeeze(-1)

    def joint_features(self, ids, types=None):
        # Historical IEEE Admission CE inference explicitly disables BF16.
        with (torch.autocast('cuda', enabled=False) if getattr(self, 'inference_fp32', False) else autocast_for(ids)):
            h = self.encoder(input_ids=ids, attention_mask=ids.ne(0), token_type_ids=types).last_hidden_state[:, 0]
        return h.float()

def pair(qs, ps):
    ids = np.zeros((len(qs), 356), np.int64)
    types = np.zeros_like(ids)
    for i, (q, p) in enumerate(zip(qs, ps)):
        q = q[q != 0]
        p = p[p != 0][1:]
        n = len(q) + len(p)
        assert n <= 356
        ids[i, :len(q)] = q
        ids[i, len(q):n] = p
        types[i, len(q):n] = 1
    return (ids, types)
