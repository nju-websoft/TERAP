"""Run a training recipe using explicitly filled JSON settings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .configuration import training_config, input_settings, default_config
from .recipes.catalog import RECIPES


def command(recipe, config_path=None, spec_path=None):
    cfg = RECIPES[recipe]
    values = training_config(recipe, config_path)
    values.update(spec=str(spec_path or '<spec.json>'))
    return [sys.executable, '-m', cfg['module']] + [arg.format(**values) for arg in cfg['args']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', choices=RECIPES)
    parser.add_argument('--config', help='Override the selected recipe configuration file.')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    args.config = Path(args.config).resolve() if args.config else default_config(args.recipe)
    try:
        values = training_config(args.recipe, args.config)
        argv = command(args.recipe, args.config)
        if args.recipe in ('mesh_memory','ieee_memory','qmesh_encoder','ieee_admission_cache') or args.recipe.startswith('placement_'):
            if args.recipe.startswith('placement_'):
                task='placement';dataset={'placement_static':'ieee_static'}.get(args.recipe,args.recipe.removeprefix('placement_'))
            else:
                task='admission';dataset='ieee' if args.recipe.startswith('ieee') else 'mesh'
            input_settings(task,dataset,'training',args.config)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, **RECIPES[args.recipe]['env'])
    env.update(RELEASE_CONFIG=str(Path(args.config).resolve()), RELEASE_RECIPE=args.recipe)
    env.setdefault('RELEASE_ROOT',str(root))
    env.setdefault('HF_HUB_OFFLINE','1')
    env.setdefault('TRANSFORMERS_OFFLINE','1')
    spec = dict(values, name=f'placement_mesh_s{values.get("seed")}')
    if args.show:
        print(json.dumps(dict(command=argv, training=values, spec=spec if args.recipe=='placement_mesh' else None),indent=2))
        return
    if args.recipe=='placement_mesh':
        folder=Path(env.get('RELEASE_WORK',root/'work'))/'specs';folder.mkdir(parents=True,exist_ok=True)
        spec_path=folder/(spec['name']+'.json');spec_path.write_text(json.dumps(spec))
        argv=command(args.recipe,args.config,spec_path=spec_path)
    subprocess.run(argv,cwd=root,env=env,check=True)

if __name__=='__main__':
    main()
