"""Execute the released preparation, training and final-evaluation stages in order."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .workflow import ROOT, WORK, config, training_config, write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['mesh','ieee','qmesh','static','all'],required=True)
    ap.add_argument('--configs',type=Path,default=WORK/'configs')
    ap.add_argument('--batch-size',type=int,default=16,help='Preparation/evaluation batch size, independent of training microbatch.')
    ap.add_argument('--plan',action='store_true',help='Print all commands without running them.')
    ap.add_argument('--start-at',type=int,default=1,help='Resume at a numbered command from --plan, keeping completed inputs unchanged.')
    a=ap.parse_args()
    a.configs=a.configs.resolve()
    scheduled=set();commands=[]
    def command(*args):commands.append([sys.executable,'-m',*map(str,args)])
    def prepare(stage,*args):command('src.workflow',stage,'--configs',a.configs,'--batch-size',a.batch_size,*args)
    def train(recipe):command('src.train',recipe,'--config',config(recipe,a.configs))
    def cfg(recipe):return training_config(recipe,config(recipe,a.configs))
    def retrieve(dataset):
        key='retrieve_'+dataset
        if key in scheduled:return retrieval_paths(dataset)
        scheduled.add(key)
        prepare('retrieval-data','--dataset',dataset)
        for kind in ('be','ce'):train(dataset+'_'+kind)
        return retrieval_paths(dataset)
    def retrieval_paths(dataset):
        be_cfg=cfg(dataset+'_be');ce_cfg=cfg(dataset+'_ce')
        if be_cfg['seed']!=ce_cfg['seed']:raise ValueError('BE and CE must use the same seed.')
        if dataset=='mesh':
            if be_cfg['epochs']!=ce_cfg['epochs']:raise ValueError('MeSH BE/CE epochs must match the existing recipe checkpoint naming.')
            root=WORK/'retrieval_mesh/runs';base=f's{be_cfg["seed"]}_e{be_cfg["epochs"]}'
            return root/(base+'_be/selected_valid/best.pt'),root/(base+'_ce_standard/selected_valid/best.pt')
        impl='retrieval_static' if dataset=='static' else 'retrieval_temporal'
        label='ieee_static' if dataset=='static' else dataset
        root=WORK/impl
        if dataset!='static':root=root/dataset
        root=root/'runs'
        return tuple(root/f'{label}_default_s{be_cfg["seed"]}_{k}/selected_valid/best.pt' for k in ('be','ce'))
    def admission(dataset):
        if 'admission_'+dataset in scheduled:return admission_paths(dataset)
        scheduled.add('admission_'+dataset)
        if dataset=='mesh':be,ce=retrieve('mesh')
        else:
            prepare('ieee-graph')
            for r in ('ieee_admission_be','ieee_admission_mining','ieee_admission_ce'):train(r)
            be,ce=WORK/'retrieval_ieee/be/best.pt',WORK/'retrieval_ieee/ce/best.pt'
        for suffix in ('query','memory'):
            recipe=dataset+'_'+suffix
            prepare('admission-data','--recipe',recipe,'--be',be,'--ce',ce)
            train(recipe)
        query,memory=admission_paths(dataset)
        prepare('evaluate-admission','--dataset',dataset,'--be',be,'--ce',ce,
                '--query-checkpoint',query,'--memory-checkpoint',memory,'--output',WORK/'results'/f'{dataset}_admission.json')
        return query,memory
    def admission_paths(dataset):
        qs=cfg(dataset+'_query')['seed'];ms=cfg(dataset+'_memory')['seed']
        qname=f'query_s{qs}' if dataset=='ieee' else f'mesh_query_s{qs}'
        return WORK/(dataset+'_query')/'runs'/qname/'best.pt',WORK/(dataset+'_memory')/'runs'/f'{dataset}_memory_s{ms}'/'best.pt'
    def placement(dataset):
        if 'placement_'+dataset in scheduled:return
        scheduled.add('placement_'+dataset)
        extra=[]
        if dataset=='mesh':extra=['--path-checkpoint',admission('mesh')[1]]
        elif dataset=='qmesh':
            mbe,mce=retrieve('mesh')
            prepare('admission-data','--recipe','qmesh_encoder','--be',mbe,'--ce',mce)
            train('qmesh_encoder')
            extra=['--path-checkpoint',WORK/'qmesh_encoder/runs'/f'qmesh_encoder_s{cfg("qmesh_encoder")["seed"]}'/'best.pt']
        be,ce=retrieve(dataset)
        recipe='placement_'+dataset
        prepare('placement-data','--dataset',dataset,'--be',be,'--ce',ce,*extra)
        train(recipe)
        seed=cfg(recipe)['seed']
        name=f'placement_mesh_s{seed}' if dataset=='mesh' else f'{"ieee_static" if dataset=="static" else dataset}_default_s{seed}'
        checkpoint=WORK/recipe/'runs'/name/'selected_valid/best.pt'
        prepare('evaluate-placement','--dataset',dataset,'--be',be,'--ce',ce,*extra,'--checkpoint',checkpoint,'--output',WORK/'results'/f'{dataset}_placement.json')
    for dataset in (['mesh','ieee','qmesh','static'] if a.dataset=='all' else [a.dataset]):
        if dataset in ('mesh','ieee'):admission(dataset)
        placement(dataset)
    import shlex
    if not 1 <= a.start_at <= len(commands):ap.error('--start-at is outside the plan')
    for number,argv in enumerate(commands,1):
        print(f'[{number}/{len(commands)}] '+shlex.join(argv),flush=True)
        if not a.plan and number >= a.start_at:
            subprocess.run(argv,cwd=ROOT,check=True)
    if not a.plan:print(f'Finished. Metrics: {WORK / "results"}',flush=True)

if __name__=='__main__':main()
