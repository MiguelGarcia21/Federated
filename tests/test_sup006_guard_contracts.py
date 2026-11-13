"""Test just the two source validator functions on fabricated filesystem paths.

Intentionally avoids importing the Flower runner because this offline environment
has no flwr/ray dependencies. Does not write to repository or train models.
"""
from pathlib import Path
import ast
import copy
from tempfile import TemporaryDirectory
import yaml

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'src').is_dir() and (ROOT / 'PROPOSED_MANUAL_SOURCE_FILES').is_dir():
    ROOT = ROOT / 'PROPOSED_MANUAL_SOURCE_FILES' 


def get_validator(path, required_globals):
    content = (ROOT / path).read_text(encoding='utf-8')
    tree = ast.parse(content)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_config']
    assert len(matches) == 1
    n = copy.deepcopy(matches[0]); n.decorator_list = []
    ns = {'Path': Path, 'Dict': dict, 'Any': object}
    ns.update(required_globals)
    exec(compile(ast.Module(body=[n], type_ignores=[]), str(path), 'exec'), ns)
    return ns['validate_config']


central = get_validator('src/experiments/run_exp01_centralized.py', {
    'SUPPORTED_CHECKPOINT_METRICS': {'f1'},
})
federated = get_validator('src/experiments/run_exp02_fedavg.py', {
    'SUPPORTED_SPLIT_TYPES': {'iid', 'non_iid_label_skew'},
    'normalize_cache_extension': lambda x: x,
})


def must_fail(fn, c, expected_exception):
    try:
        fn(c)
    except expected_exception:
        return
    except Exception as exc:
        raise AssertionError(f'Expected {expected_exception}, got {type(exc)}: {exc}') from exc
    raise AssertionError(f'Expected {expected_exception}, but configuration passed')


with TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    csv = tmp / 'train.csv'; csv.write_text('dummy', encoding='utf-8')
    images = tmp / 'images'; images.mkdir()
    cache = tmp / 'cache'; cache.mkdir()
    for path in sorted((ROOT / 'configs').glob('g5_sup006_densenet121_*.yaml')):
        c = yaml.safe_load(path.read_text(encoding='utf-8'))
        c['train_csv'] = str(csv)
        c['train_image_dir'] = str(images)
        c['output_dir'] = str(tmp / 'sup006')
        if c['run_type'] == 'flower_fedavg':
            c['image_cache_dir'] = str(cache)
            fn = federated
        else:
            fn = central
        fn(c)
        wrong = dict(c, pretrained=True)
        must_fail(fn, wrong, ValueError)
        wrong = dict(c, sup006_architecture='wrong')
        must_fail(fn, wrong, ValueError)
        wrong = dict(c, experiment_id='g4_sup008_would_overwrite')
        must_fail(fn, wrong, ValueError)
        wrong = dict(c, model_name='other_net')
        must_fail(fn, wrong, ValueError)
        if c['run_type'] == 'flower_fedavg':
            wrong = dict(c, fl_num_clients=4)
            must_fail(fn, wrong, ValueError)
            wrong = dict(c, sup008_audit=False)
            must_fail(fn, wrong, ValueError)
        existing = tmp / 'sup006' / 'checkpoints' / c['experiment_id']
        existing.mkdir(parents=True, exist_ok=True)
        must_fail(fn, c, FileExistsError)
        existing.rmdir()
        # Historical ResNet-18 path remains allowed without SUP006 opt-in.
        legacy = dict(c)
        legacy['model_name'] = 'resnet18'
        legacy.pop('sup006_architecture')
        legacy['experiment_id'] = f'g4_sup008_legacy_{c["seed"]}'
        legacy['output_dir'] = str(tmp / 'sup008')
        fn(legacy)

print('PASS: six new opt-in configs, bad-ID/bad-model/pretrained guards,')
print('      five-client audit contract, no-overwrite guard, legacy ResNet acceptance')
