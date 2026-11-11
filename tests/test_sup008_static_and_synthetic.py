"""No training, images, GPU, checkpoints, Flower, or repository writes."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None


def read_flat_yaml(path):
    text = path.read_text(encoding='utf-8')
    if yaml is not None:
        return yaml.safe_load(text)
    # Minimal fallback for supplied flat SUP008 config keys; no YAML dependency.
    out = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        key, value = line.split(':', 1)
        value = value.strip().strip('"').strip("'")
        if value in ('true','false'):
            value = value == 'true'
        else:
            try:
                value = float(value) if any(c in value for c in ('.', 'e', 'E')) and value[0:1].isdigit() else int(value)
            except ValueError:
                pass
        out[key.strip()] = value
    return out

ROOT = Path(__file__).resolve().parents[1]


def check_configs():
    files = list(sorted((ROOT / 'configs').glob('g4_sup008_*.yaml')))
    assert len(files) == 6
    got = set()
    for filename in files:
        cfg = read_flat_yaml(filename)
        exp = cfg['experiment_id']
        assert exp == filename.stem
        assert exp not in got
        got.add(exp)
        assert cfg['seed'] in (2026, 31415)
        assert cfg['reuse_split_files'] is True
        assert cfg['canonical_train_split_path'].endswith('seed42_train_split.csv')
        assert cfg['canonical_val_split_path'].endswith('seed42_val_split.csv')
        assert cfg['output_dir'] == '/workspace/thesis-fl-xai/outputs/sup008'
        assert cfg['model_name'] == 'resnet18' and cfg['pretrained'] is False
        assert cfg['num_classes'] == 2 and cfg['image_size'] == 224
        assert cfg['learning_rate'] == 0.0001 and cfg['weight_decay'] == 0.01
        assert not any(k.startswith(('sup003_', 'sup007_')) for k in cfg)
        if cfg['run_type'] == 'centralized':
            assert cfg['epochs'] == 10 and cfg['batch_size'] == 32
            assert cfg['checkpoint_metric'] == 'f1' and cfg['save_predictions'] is True
            assert cfg['seed'] in (2026, 31415)
        else:
            assert cfg['fl_num_clients'] == 5 and cfg['fl_num_rounds'] == 50
            assert cfg['fl_local_epochs'] == 1 and cfg['batch_size'] == 16
            assert cfg['sup008_audit'] is True
            assert cfg['save_initial_checkpoint'] is True
            assert cfg['save_checkpoints_every'] == 5
            if cfg['split_type'] == 'iid':
                assert cfg['class_weight_scope'] == 'local'
            else:
                assert cfg['split_type'] == 'non_iid_label_skew'
                assert cfg['non_iid_skew_strength'] == 4.0
                assert cfg['class_weight_scope'] == 'global'
    return files

spec = importlib.util.spec_from_file_location('sup008_participation', ROOT / 'src/training/sup008_participation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
validate = module.validate_sup008_round


def good_round(sizes, roundnum=1):
    plan = {f'proxy_{i}': i for i in range(5)}
    res = [{'cid': f'proxy_{i}', 'logical_client_id': i, 'num_examples': sizes[i]} for i in range(5)]
    return dict(server_round=roundnum, expected_sizes=sizes, planned_by_proxy=plan,
                results=res, failures_count=0)


def must_fail(params, label):
    try:
        validate(**params)
    except RuntimeError:
        return label + ': PASS(expected rejection)'
    raise AssertionError('FAIL did not reject ' + label)


def main():
    files = check_configs()
    cases = []
    for label, sizes in [('IID', {i: 2400 for i in range(5)}),
                         ('NONIID', dict(enumerate([2996,2698,2400,2102,1804])))]:
        sample = good_round(sizes)
        audit = validate(**sample)
        assert audit['sample_total'] == 12000
        assert len(audit['aggregation_weight_by_logical_client']) == 5
        assert sum(audit['aggregation_weight_by_logical_client'].values()) == 1.0
        cases.append(label + ': PASS(exact five clients)')
        miss = good_round(sizes)
        miss['results'] = miss['results'][:-1]
        cases.append(must_fail(miss, label + ' missing client'))
        duplicate = good_round(sizes)
        duplicate['results'][-1] = dict(duplicate['results'][-1], logical_client_id=0)
        cases.append(must_fail(duplicate, label + ' duplicate logical partition'))
        mismatch = good_round(sizes)
        mismatch['results'][1] = dict(mismatch['results'][1], num_examples=sizes[1] - 1)
        cases.append(must_fail(mismatch, label + ' wrong sample count'))
        failure = good_round(sizes)
        failure['failures_count'] = 1
        cases.append(must_fail(failure, label + ' Flower reported failure'))
        remapped = good_round(sizes)
        remapped['results'][0] = dict(remapped['results'][0], logical_client_id=1)
        cases.append(must_fail(remapped, label + ' proxy identity mismatch'))
    print(json.dumps({'result': 'PASS', 'yaml_count': len(files), 'cases': cases}, indent=2))

if __name__=='__main__':
    main()
