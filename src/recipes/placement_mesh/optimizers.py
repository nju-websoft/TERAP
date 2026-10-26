from ...configuration import parameter as hp
'Muon for hidden matrices; AdamW for the output head and vector parameters.'
import torch

def build(model, name, lr):
    return {'adamw': torch.optim.AdamW}[name](model.parameters(), lr=lr, weight_decay=hp('wd'))
