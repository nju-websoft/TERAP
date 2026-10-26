"""Unchanged low_lr BE/CE architectures for the mask32 dataset."""
import argparse,math,collections,time,concurrent.futures
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from .common import *
class Network(nn.Module):
 def __init__(self,kind,dim=256):
  super().__init__();self.kind=kind;self.dim=dim;self.encoder=AutoModel.from_pretrained(MODEL,local_files_only=True,add_pooling_layer=False,attn_implementation='sdpa')
  if kind=='be':self.query=nn.Linear(768,dim);self.parent=nn.Linear(768,dim)
  else:self.classifier=nn.Linear(768,1)
 def encode(self,ids,role):
  mask=ids.ne(0)
  with torch.autocast('cuda',dtype=torch.bfloat16):h=self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state
  h=(h.float()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None];return F.normalize(getattr(self,role)(h),dim=-1)
 def forward(self,q,p=None,types=None):
  if self.kind=='be':return torch.einsum('bd,bkd->bk',self.encode(q,'query'),self.encode(p.flatten(0,1),'parent').reshape(p.shape[0],p.shape[1],-1))/.05
  with torch.autocast('cuda',dtype=torch.bfloat16):h=self.encoder(input_ids=q,attention_mask=q.ne(0),token_type_ids=types).last_hidden_state[:,0]
  return self.classifier(h.float()).squeeze(-1)
def pair(qs,ps):
 ids=np.zeros((len(qs),356),np.int64);types=np.zeros_like(ids)
 for i,(q,p) in enumerate(zip(qs,ps)):
  q=q[q!=0];p=p[p!=0][1:];n=len(q)+len(p);assert n<=356;ids[i,:len(q)]=q;ids[i,len(q):n]=p;types[i,len(q):n]=1
 return ids,types
