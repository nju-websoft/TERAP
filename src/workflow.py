"""Prepare released data and evaluate the existing task recipes."""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

import numpy as np
import torch
from transformers import AutoTokenizer

from .configuration import default_config, load_config, training_config, input_settings
from .data import Corpus, Graph, load_rows, load_taxonomy, sample_id
from .encoder import Encoder
from .retrieval import RetrievalModel
from .pipeline import RetrievalEngine, prepare_placement
from .memory import build_memory
from .evaluation import placement_metrics, admission_metrics
from .recipes.runtime import path_bindings

DATASETS = {'mesh': 'TaxoAP-MeSH', 'ieee': 'TaxoAP-IEEE', 'qmesh': 'QEN-MeSH', 'static': 'IEEE-static'}
ROOT = Path(os.environ.get('RELEASE_ROOT', Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get('RELEASE_WORK', ROOT / 'work'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def config(recipe, directory=None):
    return Path(directory) / (recipe + '.json') if directory else default_config(recipe)


def paths(recipe):
    return path_bindings(recipe, ['DATA', 'SRC', 'GRAPH', 'BASE', 'MODEL', 'OUT', 'CACHE_ROOT'])


def corpus(dataset, model, admission=False):
    style = 'ieee_paths' if admission and dataset == 'ieee' else 'mesh_temporal' if dataset == 'mesh' else 'name_definition'
    return Corpus(ROOT / 'dataset' / DATASETS[dataset], AutoTokenizer.from_pretrained(model, local_files_only=True), style)


def encode_to_file(encoder, tokens, path, batch=32):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(tokens), 768))
    device = next(encoder.parameters()).device
    encoder.eval()
    with torch.inference_mode():
        for lo in range(0, len(tokens), batch):
            result[lo:lo+batch] = encoder(torch.as_tensor(tokens[lo:lo+batch], device=device, dtype=torch.long)).cpu().numpy()
    result.flush()
    return result


def init_configs(destination):
    """Generate editable starting configurations for each task."""
    from .recipes.catalog import RECIPES
    values = dict(seed=42, lr=1e-5, epochs=3, steps=1000, switch=0, weak_start=0,
                  projection_start=0, head_lr=1e-4, encoder_lr=1e-5, projection_lr=1e-4,
                  weight_decay=0.01, wd=0.01, batch_size=16, micro=2, chunk=16,
                  grad_clip=1.0, eval_every=100, eval_every_epochs=1, warmup=0.1,
                  warmup_fraction=0.1, min_lr_factor=0.1, dropout=0.1, beta2=0.999,
                  momentum=0.9, width=256, depth=2, heads=8, attn_scale=1.0,
                  optimizer='adamw', schedule='constant', hard_pool_k=128, accumulation=1, shard=0)
    for recipe in RECIPES:
        data = json.loads(default_config(recipe).read_text())
        data['training'] = {k: values[k] for k in data['training']}
        if recipe.startswith('placement_'):
            data['training'].update(lr=1e-4, epochs=5)
        if recipe == 'placement_static':
            data['training']['schedule'] = 'cosine'
        for tasks in data.get('inputs', {}).values():
            for phases in tasks.values():
                for phase in phases:
                    phases[phase] = dict(be_k=256, ce_k=32, slots=32, max_paths=32)
        target = Path(destination) / (recipe + '.json')
        if target.exists():
            raise FileExistsError(f'Refuse to replace {target}; choose a new configuration directory.')
        write(target, data)


def prepare_retrieval(dataset, model):
    implementation = 'retrieval_mesh' if dataset == 'mesh' else 'retrieval_static' if dataset == 'static' else 'retrieval_temporal'
    os.environ['PLACEMENT_DATASET'] = 'ieee_static' if dataset == 'static' else dataset
    p = paths(implementation)
    c = corpus(dataset, model)
    rows = {s: c.rows('placement', s) for s in ('train', 'valid', 'test')}
    all_rows = sum(rows.values(), [])
    for year in sorted({r.get('year', 0) for r in all_rows}):
        subset = [r for r in all_rows if r.get('year', 0) == year]
        graph = c.graph(subset[0])
        for i, row in enumerate(subset):
            row['year'] = year
            row['token_index'] = i
            row['gold_indices'] = [graph.index[u] for u in row['gold']]
            row['masked_index'] = graph.index[row['target']] if row.get('masked', False) else -1
        p['SRC'].mkdir(parents=True, exist_ok=True)
        np.save(p['SRC'] / f'{year}_tokens.npy', c.queries(subset))
        target = p['GRAPH'] / f'{year-1}_tokens.npy' if dataset == 'mesh' else p['BASE'] / f'cache/parent_tokens_{year}.npy'
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, graph.tokens)
        write(p['GRAPH'] / f'{year-1}_index.json', dict(uids=graph.uids,
              parents=[sorted(graph.index[v] for v in graph.parents[u]) for u in graph.uids],
              children=[sorted(graph.index[v] for v in graph.children[u]) for u in graph.uids]))
    for split, rr in rows.items():
        write(p['DATA'] / f'{split}.json', rr)
    print(f'Prepared {implementation}: {len(all_rows)} rows', flush=True)


def prepare_ieee_graph(model):
    """2022 graph self-supervision: leaf holdout, multi-parent positives, lexical negatives."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    p = paths('retrieval_ieee')
    p['OUT'].mkdir(parents=True, exist_ok=True)
    c = corpus('ieee', model, admission=True)
    graph = c.graph({'year': 2023})
    parents = [sorted(graph.index[v] for v in graph.parents[u]) for u in graph.uids]
    leaves = [i for i, u in enumerate(graph.uids) if parents[i] and not graph.children[u]]
    rng = np.random.default_rng(20)
    rng.shuffle(leaves)
    valid = sorted(leaves[:max(1, len(leaves)//10)])
    held = set(valid)
    train = [i for i, pp in enumerate(parents) if pp and i not in held]
    bank = [i for i in range(len(graph.uids)) if i not in held]
    names = [graph.records[u]['name'] for u in graph.uids]
    tokens = np.asarray(c.tokenizer(['Query: '+n for n in names], max_length=128, padding='max_length', truncation=True)['input_ids'])
    x = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)).fit_transform(names)
    hard = []
    for start in range(0, len(names), 128):
        similarities = (x[start:start+128] @ x.T).toarray()
        for j, sim in enumerate(similarities):
            i = start+j
            allowed = np.array([k for k in bank if k != i and k not in parents[i]])
            hard.append(allowed[np.argsort(-sim[allowed], kind='stable')[:32]].tolist())
    write(p['DATA'] / 'be_graph.json', dict(train=train, valid=valid, bank=bank, parents=parents, hard=hard))
    np.save(p['DATA'] / 'be_query_tokens.npy', tokens)
    np.save(p['DATA'] / 'S23_parent_tokens.npy', graph.tokens)
    rows = sum([c.rows('admission', s) for s in ('train','valid','test')], [])
    for r in rows:
        r['weak'] = r.get('weak', False)
        r['window'] = 'S'+str(r['year'])[-2:]
    write(p['DATA'] / 'rows.json', rows)
    np.save(p['DATA'] / 'query_tokens.npy', c.queries(rows))
    for y in (2024,2025):
        np.save(p['DATA'] / f'S{str(y)[-2:]}_parent_tokens.npy', c.graph({'year': y}).tokens)


def retriever(c, model, be_path, ce_path, device, batch):
    be = RetrievalModel('be', model).to(device)
    ce = RetrievalModel('ce', model).to(device)
    be.load_state_dict(torch.load(be_path, map_location='cpu', weights_only=True))
    ce.load_state_dict(torch.load(ce_path, map_location='cpu', weights_only=True))
    return RetrievalEngine(c, be.eval(), ce.eval(), batch)


def normalized(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', str(text)).casefold()))


def qmesh_exclusion():
    root = ROOT / 'dataset' / DATASETS['qmesh']
    records = load_taxonomy(root)
    held = {r['target'] for s in ('valid','test') for r in load_rows(root,'placement',s)}
    exclusion = json.loads((Path(__file__).parent/'recipes/qmesh_encoder/exclusions.json').read_text())
    if held != set(exclusion['targets']):
        raise ValueError('QEN-MeSH split changed; regenerate the UID/alias exclusion mapping before encoder training.')
    held = set(exclusion['uids'])
    names = {normalized(x) for x in exclusion['names']}
    pat = re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(x) for x in sorted(names,key=len,reverse=True) if x)+r')(?!\w)')
    def blocked(row):
        return str(row.get('candidate_id',row.get('target'))) in held or any(normalized(row.get(k,'')) in names for k in ('name','official_or_previous_name')) or bool(pat.search(normalized(row.get('query',''))))
    return blocked, names


def prepare_admission_cache(recipe, cfg_path, model, be, ce, device, batch):
    dataset = 'ieee' if recipe.startswith('ieee') else 'mesh'
    c = corpus(dataset,model,admission=True)
    rows = sum([c.rows('admission',s) for s in ('train','valid','test')],[])
    rows = [dict(r, weak=r.get('weak',False)) for r in rows if r['split'] != 'test']
    excluded_names = set()
    if recipe == 'qmesh_encoder':
        blocked, excluded_names = qmesh_exclusion()
        rows = [r for r in rows if not blocked(r)]
    memory = not recipe.endswith('_query')
    cfg = training_config(recipe,cfg_path)
    p = paths(recipe)['DATA']
    p.mkdir(parents=True,exist_ok=True)
    if memory:
        sizes = input_settings('admission',dataset,'training',cfg_path)
        engine = retriever(c,model,be,ce,device,batch)
        texts, index, ids, roles, kept = [], {}, [], [], []
        for lo in range(0,len(rows),batch):
            block = rows[lo:lo+batch]
            ranked = engine.rank(block,sizes['be_k'],sizes['ce_k'],full=False)
            for row,result in zip(block,ranked):
                paths_,_ = build_memory(result['natural'],c.graph(row).records,sizes['max_paths'])
                paths_ = [t for t in paths_ if not any(normalized(n) in excluded_names for n in t.split('[SEP]'))]
                if not paths_:
                    if recipe == 'qmesh_encoder': continue
                    raise ValueError('No memory paths for '+str(row.get('id')))
                mi = np.zeros(sizes['slots'],np.int32); rr=np.full(sizes['slots'],3,np.int32)
                for j,t in enumerate(paths_):
                    if t not in index: index[t]=len(texts);texts.append(t)
                    mi[j]=index[t];rr[j]=0
                ids.append(mi);roles.append(rr);kept.append(row)
            print(f'{recipe}: retrieved {min(lo+batch,len(rows))}/{len(rows)}',flush=True)
        rows=kept
        del engine
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        pt=np.asarray(c.tokenizer(texts,max_length=226,padding='max_length',truncation=True)['input_ids'])
        np.save(p/'path_tokens.npy',pt)
        np.savez(p/f'without_weak_memory_{cfg["seed"]}.npz',memory=np.stack(ids),roles=np.stack(roles))
    qt=c.queries(rows)
    write(p/'rows.json',rows);np.save(p/'query_tokens.npy',qt)
    net=Encoder(model,False).to(device).eval()
    encode_to_file(net,qt,p/'raw_query.npy',batch)
    if memory: encode_to_file(net,pt,p/'raw_bank.npy',batch)
    print(f'Prepared {recipe}: {len(rows)} rows',flush=True)


def path_encoder(model, checkpoint, device):
    net=Encoder(model,False)
    if checkpoint:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state.get('encoder') is None:
            raise ValueError('Path checkpoint must contain a fine-tuned encoder; select a joint-stage checkpoint.')
        net.load_state_dict(state['encoder'])
    return net.to(device).eval()


def prepare_placement_cache(dataset,cfg_path,model,be,ce,path_checkpoint,device,batch):
    recipe='placement_'+dataset
    cfg=training_config(recipe,cfg_path)
    c=corpus(dataset,model)
    engine=retriever(c,model,be,ce,device,batch)
    encoder=path_encoder(model,path_checkpoint,device)
    sizes=input_settings('placement','ieee_static' if dataset=='static' else dataset,'training',cfg_path)
    p=paths(recipe)['CACHE_ROOT']/f's{cfg["seed"]}'
    p.mkdir(parents=True,exist_ok=True)
    rows=sum([c.rows('placement',s) for s in ('train','valid')],[])
    if len([r for r in rows if r['split']=='train'])<cfg['batch_size']:
        raise ValueError('Training split is smaller than batch_size.')
    shape=(len(rows),sizes['ce_k'],768)
    features=np.lib.format.open_memmap(p/'features.npy',mode='w+',dtype=np.float32,shape=shape)
    gold=[];hard=[];roles=[];memory=[];bank=[];bank_index={};path_cache={}
    for i,row in enumerate(rows):
        e=prepare_placement(c,[row],engine,encoder,training=row['split']=='train',
            be_k=sizes['be_k'],top_k=sizes['ce_k'],slots=sizes['slots'],max_paths=sizes['max_paths'],path_chunk=batch,path_cache=path_cache)[0]
        if len(e['candidates'])!=sizes['ce_k']:raise ValueError('ce_k exceeds available candidates.')
        features[i]=e['features'].numpy()
        valid=e['roles'].ne(3).numpy();mi=np.zeros(sizes['slots'],np.int32)
        for j,text in enumerate(e['path_texts']):
            if text not in bank_index:
                bank_index[text]=len(bank);bank.append(e['memory'][j].numpy())
            mi[j]=bank_index[text]
        memory.append(mi);roles.append(e['roles'].numpy());gold.append(e['gold'].numpy());hard.append(e['hard8'].numpy())
        row['gold_indices']=[c.graph(row).index[u] for u in row['gold']]
        if (i+1)%100==0:print(f'{recipe}: prepared {i+1}/{len(rows)}',flush=True)
    features.flush()
    write(p/'rows.json',rows)
    np.save(p/'path_features.npy',np.asarray(bank,np.float32))
    np.savez(p/'inputs.npz',memory=np.stack(memory),roles=np.stack(roles),gold=np.stack(gold),hard8=np.stack(hard))
    torch.save(engine.ce.classifier.state_dict(),p/'CE_head.pt')
    from .recipes.cache import load_placement_cache
    load_placement_cache(p)


def admission_model(recipe,cfg_path,model,checkpoint,device):
    from .admission import AdmissionHead,AdmissionModel
    cfg=training_config(recipe,cfg_path)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    memory=not recipe.endswith('_query')
    options=state.get('model_config',dict(variant='true' if memory else 'natural_linear',d=cfg.get('width',256),depth=cfg.get('depth',2),use_role=not recipe.startswith('ieee'),heads=cfg.get('heads',8),attn_scale=cfg.get('attn_scale',1.0)))
    head=AdmissionHead(**options)
    head.load_state_dict(state['head'])
    net=Encoder(model,False)
    if state['encoder'] is not None:net.load_state_dict(state['encoder'])
    return AdmissionModel(net,head).to(device).eval()


def evaluate_admission_run(dataset,config_dir,model,be,ce,query_checkpoint,memory_checkpoint,device,batch,output):
    from .pipeline import prepare_admission,predict_admission
    c=corpus(dataset,model,admission=True)
    engine=retriever(c,model,be,ce,device,batch)
    qr=dataset+'_query';mr=dataset+'_memory'
    q=admission_model(qr,config(qr,config_dir),model,query_checkpoint,device)
    m=admission_model(mr,config(mr,config_dir),model,memory_checkpoint,device)
    arrays=[];split_rows=[]
    for split in ('valid','test'):
        rows=c.rows('admission',split);split_rows.append(rows);qs=[];ms=[]
        for lo in range(0,len(rows),batch):
            examples=prepare_admission(c,rows[lo:lo+batch],engine,config_path=config(mr,config_dir))
            qs.extend(predict_admission(q,examples,batch));ms.extend(predict_admission(m,examples,batch))
        arrays.append((np.asarray(qs),np.asarray(ms),np.asarray([r['label'] for r in rows],bool)))
    metrics=admission_metrics(*arrays[0],*arrays[1])
    write(output,metrics)
    with Path(output).with_suffix('.predictions.jsonl').open('w') as stream:
        a,b=metrics['valid']['query_threshold'],metrics['valid']['memory_threshold']
        for rows,(qs,ms,_) in zip(split_rows,arrays):
            for r,qscore,mscore in zip(rows,qs,ms):
                stream.write(json.dumps(dict(id=sample_id(r),split=r['split'],query_score=float(qscore),memory_score=float(mscore),accepted=bool(qscore>=a and mscore>=b)))+'\n')


def evaluate_placement_run(dataset,cfg_path,model,be,ce,path_checkpoint,checkpoint,device,batch,output):
    recipe='placement_'+dataset
    c=corpus(dataset,model)
    engine=retriever(c,model,be,ce,device,batch)
    encoder=path_encoder(model,path_checkpoint,device)
    module=importlib.import_module('src.recipes.'+recipe+'.model')
    common=importlib.import_module('src.recipes.'+recipe+'.common')
    cfg=dict(common.CONFIGS['default'],dropout=training_config(recipe,cfg_path)['dropout'])
    cfg['memory_slots']=input_settings('placement','ieee_static' if dataset=='static' else dataset,'inference',cfg_path)['slots']
    net=module.CARE(cfg,engine.ce.classifier.state_dict()).to(device)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    net.load_state_dict(state.get('model',state));net.eval()
    from .pipeline import predict_placement
    out={};predictions=[]
    for split in ('valid','test'):
        rows=c.rows('placement',split);rankings=[]
        for lo in range(0,len(rows),batch):
            examples=prepare_placement(c,rows[lo:lo+batch],engine,encoder,config_path=cfg_path,path_chunk=batch)
            rankings.extend(predict_placement(net,examples,batch))
        out[split]=placement_metrics(rankings,rows)
        predictions.extend(dict(id=sample_id(r),split=split,ranking=ranking) for r,ranking in zip(rows,rankings))
    write(output,out)
    with Path(output).with_suffix('.predictions.jsonl').open('w') as stream:
        for r in predictions:stream.write(json.dumps(r)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['init-configs','download','retrieval-data','ieee-graph','admission-data','placement-data','evaluate-admission','evaluate-placement'])
    ap.add_argument('--dataset',choices=DATASETS)
    ap.add_argument('--recipe',choices=['mesh_query','mesh_memory','ieee_query','ieee_memory','qmesh_encoder'])
    ap.add_argument('--configs',type=Path,default=WORK/'configs')
    ap.add_argument('--model',type=Path,default=ROOT/'models/bert')
    ap.add_argument('--model-id',default='google-bert/bert-base-uncased')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--batch-size',type=int,default=16)
    for name in ['be','ce','path-checkpoint','checkpoint','query-checkpoint','memory-checkpoint','output']:
        ap.add_argument('--'+name,type=Path)
    a=ap.parse_args()
    if a.stage=='init-configs':init_configs(a.configs);return
    if a.stage=='download':
        from huggingface_hub import snapshot_download
        snapshot_download(a.model_id,local_dir=a.model,allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json'])
        return
    if not a.model.is_dir():ap.error('Download the BERT model first.')
    from transformers import AutoConfig
    model_config=AutoConfig.from_pretrained(a.model,local_files_only=True)
    if model_config.model_type!='bert' or model_config.hidden_size!=768:
        ap.error('Use a BERT model with hidden_size=768.')
    if a.stage not in ('retrieval-data','ieee-graph') and a.device.startswith('cuda') and not torch.cuda.is_available():ap.error('CUDA is unavailable; preparation/evaluation also support --device cpu.')
    if a.stage=='retrieval-data':
        if not a.dataset:ap.error('--dataset is required')
        prepare_retrieval(a.dataset,a.model)
    elif a.stage=='ieee-graph':prepare_ieee_graph(a.model)
    elif a.stage=='admission-data':
        if not a.recipe:ap.error('--recipe is required')
        if not a.recipe.endswith('_query') and (not a.be or not a.ce):ap.error('Memory preparation requires --be and --ce checkpoints.')
        prepare_admission_cache(a.recipe,config(a.recipe,a.configs),a.model,a.be,a.ce,a.device,a.batch_size)
    else:
        if not a.dataset or not a.be or not a.ce:ap.error('--dataset, --be and --ce are required')
        if a.dataset in ('mesh','qmesh') and 'placement' in a.stage and not a.path_checkpoint:ap.error('MeSH/QEN-MeSH Placement requires --path-checkpoint from its Admission encoder.')
        if a.stage=='placement-data':prepare_placement_cache(a.dataset,config('placement_'+a.dataset,a.configs),a.model,a.be,a.ce,a.path_checkpoint,a.device,a.batch_size)
        elif a.stage=='evaluate-placement':
            if not a.checkpoint or not a.output:ap.error('--checkpoint and --output are required')
            evaluate_placement_run(a.dataset,config('placement_'+a.dataset,a.configs),a.model,a.be,a.ce,a.path_checkpoint,a.checkpoint,a.device,a.batch_size,a.output)
        elif a.stage=='evaluate-admission':
            if a.dataset not in ('mesh','ieee') or not a.query_checkpoint or not a.memory_checkpoint or not a.output:ap.error('Admission evaluation needs mesh/ieee, both branch checkpoints and --output')
            evaluate_admission_run(a.dataset,a.configs,a.model,a.be,a.ce,a.query_checkpoint,a.memory_checkpoint,a.device,a.batch_size,a.output)

if __name__=='__main__':main()
