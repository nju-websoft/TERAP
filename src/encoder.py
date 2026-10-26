"""Shared BERT masked-mean encoder (padding token ID 0)."""
from contextlib import nullcontext
import torch
from torch import nn
from transformers import AutoModel

def autocast_for(tokens):
    return torch.autocast('cuda', dtype=torch.bfloat16) if tokens.is_cuda else nullcontext()

class Encoder(nn.Module):
    def __init__(self, model_path, gradient_checkpointing=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True,
            add_pooling_layer=False, attn_implementation='sdpa')
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
                'use_reentrant': False, 'preserve_rng_state': True})

    def forward(self, tokens):
        if not tokens.ne(0).any(1).all():
            raise ValueError('Each sequence must contain at least one non-padding token.')
        tokens = tokens[:, :int(tokens.ne(0).sum(1).max())]
        mask = tokens.ne(0)
        with autocast_for(tokens):
            h = self.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
        return (h.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
