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
