"""Read user-supplied settings. Null fields have no implicit defaults."""
import json
import os
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=None)
def _read(path):
    return json.loads(Path(path).read_text())

def default_config(recipe):
    from .recipes.catalog import RECIPES
    package = RECIPES[recipe]['module'].split('.')[2:-1]
    return Path(__file__).resolve().parent.joinpath('recipes', *package, 'configs', recipe + '.json')

def load_config(path=None, recipe=None):
    if path is None:
        path = os.environ.get('RELEASE_CONFIG')
    if path is None:
        recipe = recipe or os.environ.get('RELEASE_RECIPE')
        if not recipe:
            raise ValueError('Select a recipe or pass its configuration path.')
        path = default_config(recipe)
    return _read(str(Path(path).resolve()))

def required(mapping, key, location):
    value = mapping.get(key)
    if value is None or value == '':
        raise ValueError(f'Fill {location}.{key} in the recipe configuration before running.')
    return value

def training_config(recipe=None, path=None):
    recipe = recipe or os.environ.get('RELEASE_RECIPE')
    if not recipe:
        raise ValueError('Select a recipe with python -m src.train RECIPE --config PATH.')
    values = load_config(path, recipe)['training']
    for key in values:
        value = required(values, key, 'training.' + recipe)
        if key in ('seed','shard','steps','epochs','eval_every','eval_every_epochs','batch_size','micro','chunk','accumulation','width','depth','heads','hard_pool_k'):
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key in ('seed','shard') else 1):
                raise ValueError(f'training.{recipe}.{key} must be an integer in its valid range.')
        if key in ('lr','head_lr','encoder_lr','projection_lr','grad_clip') and (isinstance(value,bool) or not isinstance(value,(int,float)) or value <= 0):
            raise ValueError(f'training.{recipe}.{key} must be positive.')
        if key in ('warmup','warmup_fraction','dropout','min_lr_factor') and not 0 <= value < 1:
            raise ValueError(f'training.{recipe}.{key} must be in [0,1).')
        if key in ('switch','weak_start','projection_start') and (not isinstance(value,int) or isinstance(value,bool) or value<0):
            raise ValueError(f'training.{recipe}.{key} must be a nonnegative integer.')
        if key in ('wd','weight_decay') and (not isinstance(value,(int,float)) or isinstance(value,bool) or value<0):
            raise ValueError(f'training.{recipe}.{key} must be nonnegative.')
    return dict(values)

def parameter(name):
    values = training_config()
    return required(values, name, 'training.' + os.environ['RELEASE_RECIPE'])

def input_settings(task, dataset, phase='inference', path=None):
    if path is None and not os.environ.get('RELEASE_CONFIG'):
        recipe = ('placement_' + ('static' if dataset == 'ieee_static' else dataset)) if task == 'placement' else dataset + '_memory'
        path = default_config(recipe)
    values = load_config(path)['inputs'][task][dataset][phase]
    values = {k:required(values,k,f'inputs.{task}.{dataset}.{phase}') for k in ('be_k','ce_k','slots','max_paths')}
    if any(isinstance(v,bool) or not isinstance(v,int) or v < 1 for v in values.values()):
        raise ValueError('Input sizes must be positive integers.')
    if values['ce_k'] > values['be_k'] or values['max_paths'] > values['slots']:
        raise ValueError('Require ce_k <= be_k and max_paths <= slots.')
    return values

def inputs_for_recipe(phase='training'):
    recipe = os.environ.get('RELEASE_RECIPE', '')
    if recipe.startswith('placement_'):
        task='placement';dataset={'placement_static':'ieee_static'}.get(recipe,recipe.removeprefix('placement_'))
    else:
        task='admission';dataset='ieee' if recipe.startswith('ieee') else 'mesh'
    return input_settings(task,dataset,phase)
