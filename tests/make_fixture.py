"""Build a disposable small corpus and randomly initialized BERT for integration tests."""
import json
import shutil
import sys
from pathlib import Path
import torch
from transformers import BertConfig, BertModel, BertTokenizer

source=Path(__file__).resolve().parents[1]
root=Path(sys.argv[1]).resolve()
if root.exists():raise FileExistsError(root)
root.mkdir(parents=True)
shutil.copytree(source/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__'))
model=root/'models/bert';model.mkdir(parents=True)
vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','query',':','node','root','new','negative','definition','candidate','parent','ancestor','paths',';','|','>']+[str(i) for i in range(1100)]
(model/'vocab.txt').write_text('\n'.join(vocab)+'\n')
tokenizer=BertTokenizer(vocab_file=str(model/'vocab.txt'),do_lower_case=True)
tokenizer.save_pretrained(model)
torch.manual_seed(42)
BertModel(BertConfig(vocab_size=len(vocab),hidden_size=768,num_hidden_layers=1,num_attention_heads=12,intermediate_size=64),add_pooling_layer=False).save_pretrained(model)
records={f'n{i}':dict(name=f'node {i}',definition='definition',parents=[] if i<1000 else [f'n{i%10}'],paths=[[]] if i<1000 else [[f'n{i%10}']]) for i in range(1035)}
def write(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x))
def lines(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
for dataset in ['TaxoAP-MeSH','TaxoAP-IEEE','QEN-MeSH','IEEE-static']:
 d=root/'dataset'/dataset
 years=range(2015,2026) if dataset=='TaxoAP-MeSH' else range(2019,2025) if dataset=='TaxoAP-IEEE' else ['static']
 for year in years:write(d/'taxonomy'/f'{year}.json',dict(records=records))
 for split,offset in [('train',0),('valid',20),('test',22)]:
  if dataset=='TaxoAP-MeSH' and split!='train':offset+=4
  rr=[]
  for i in range(4 if split=='train' else 2):
   y=(2016 if split=='train' else 2025 if split=='valid' else 2026) if dataset=='TaxoAP-MeSH' else (2023 if split=='train' else 2024 if split=='valid' else 2025) if dataset=='TaxoAP-IEEE' else 0
   rr.append(dict(sample_id=f'{split}{i}',id=f'{split}{i}',target=f'n{1000+offset+i}',year=y,name=f'node {1000+offset+i}',gold=[f'n{(offset+i)%10}'],masked=True,query=f'Query: node {1000+offset+i} [SEP] definition',split=split))
  lines(d/'placement'/f'{split}.jsonl',rr)
  if dataset.startswith('TaxoAP'):
   for i,r in enumerate(rr):
    r.update(label=i%2,weak=False,paragraph='new node',start=0,end=3,name='new')
   lines(d/'admission'/f'{split}.jsonl',rr)
held=[f'n{i}' for i in [1020,1021,1022,1023]]
write(root/'src/recipes/qmesh_encoder/exclusions.json',dict(targets=held,uids=held,names=[f'node {i}' for i in [1020,1021,1022,1023]]))
print(root)
