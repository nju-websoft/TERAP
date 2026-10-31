"""Run all workflow stages on the synthetic integration fixture."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
os.chdir(root);os.environ['RELEASE_ROOT']=str(root);os.environ['RELEASE_WORK']=str(root/'work')
def run(*args):subprocess.run([sys.executable,'-m',*map(str,args)],check=True)
run('src.workflow','init-configs')
for p in (root/'work/configs').glob('*.json'):
 d=json.loads(p.read_text());t=d['training']
 for k,v in dict(epochs=1,steps=2,batch_size=2,micro=1,chunk=4,eval_every=1,switch=0,projection_start=0,hard_pool_k=32).items():
  if k in t:t[k]=v
 for task in d.get('inputs',{}).values():
  for phases in task.values():
   for value in phases.values():value.update(be_k=8,ce_k=4,slots=4,max_paths=4)
 p.write_text(json.dumps(d))
run('src.run','--dataset','all','--batch-size','4')
results=list((root/'work/results').glob('*.json'))
assert len(results)==6,results
for p in results:
 assert p.with_suffix('.predictions.jsonl').is_file()
 d=json.loads(p.read_text());assert 'valid' in d and 'test' in d
(root/'complete.json').write_text(json.dumps(dict(status='passed',results=[p.name for p in results],stopped=False,hard_roof_triggered=False)))
print('ALL SIX END-TO-END WORKFLOWS PASSED',flush=True)
