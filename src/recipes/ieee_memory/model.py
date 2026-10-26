from ...configuration import inputs_for_recipe
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from .common import BASE
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=AutoModel.from_pretrained(BASE,local_files_only=True,add_pooling_layer=False,attn_implementation='sdpa');self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
 def forward(self,tokens):
  tokens=tokens[:,:int(tokens.ne(0).sum(1).max())];mask=tokens.ne(0)
  with torch.autocast('cuda',dtype=torch.bfloat16):h=self.encoder(input_ids=tokens,attention_mask=mask).last_hidden_state
  return (h.float()*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
from ...admission import AdmissionHead

class Head(AdmissionHead):
 def __init__(self, variant, d=256, depth=2, heads=8, attn_scale=1.):
  super().__init__(variant, d, depth, use_role=False, memory_slots=None, heads=heads, attn_scale=attn_scale)
  self.no_role = True
