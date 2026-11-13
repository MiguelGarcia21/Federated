"""SUP-006 source-only preflight for author/Master. No training, no network, no repository writes.
Run in a Python environment with torch, torchvision and PyYAML installed:
  python tests/test_sup006_proposal.py
Run after manually applying and changing path's root to your repository.
"""
from pathlib import Path
import importlib.util
import sys
import hashlib
import ast
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src' / 'models' / 'cnn.py'
if not SRC.is_file():
    raise RuntimeError('Expected applied source under src/models/cnn.py')
for rel in (
    'src/models/cnn.py',
    'src/experiments/run_exp01_centralized.py',
    'src/experiments/run_exp02_fedavg.py',
    'src/experiments/run_sup006_r50_export.py',
):
    ast.parse((ROOT / rel).read_text(encoding='utf-8'), filename=rel)

spec = importlib.util.spec_from_file_location('sup006_model_module', SRC)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import torch

torch.set_num_threads(2)
dense = module.create_model('densenet121', num_classes=2, pretrained=False)
dense.eval()
assert dense.classifier.out_features == 2
assert module.count_trainable_parameters(dense) == 6955906
assert len(dense.state_dict()) == 727
assert sum(t.numel() * t.element_size() for t in dense.state_dict().values()) == 28159184
with torch.no_grad():
    logits = dense(torch.randn(1, 3, 96, 96))
assert tuple(logits.shape) == (1, 2) and torch.isfinite(logits).all()
arrays = [t.detach().clone() for t in dense.state_dict().values()]
copy = module.create_model('densenet121', num_classes=2, pretrained=False)
from collections import OrderedDict
copy.load_state_dict(OrderedDict(zip(copy.state_dict(), arrays)), strict=True)
assert all(torch.equal(x, y) for x, y in zip(dense.state_dict().values(), copy.state_dict().values()))
try:
    module.create_model('densenet121', 2, pretrained=True)
except ValueError:
    pass
else:
    raise AssertionError('DenseNet pretrained=True must fail without downloading weights')
try:
    module.create_model('not-a-model', 2, pretrained=False)
except ValueError:
    pass
else:
    raise AssertionError('Unsupported architecture must fail')
resnet = module.create_model()
assert resnet.fc.out_features == 2
assert module.count_trainable_parameters(resnet) == 11177538

configs = sorted((ROOT / 'configs').glob('g5_sup006_densenet121_*.yaml'))
assert len(configs) == 6, len(configs)
for p in configs:
    c = yaml.safe_load(p.read_text(encoding='utf-8'))
    seed = int(c['seed'])
    assert seed in (2026, 31415)
    assert c['experiment_id'] == p.stem
    assert c['model_name'] == 'densenet121'
    assert c['sup006_architecture'] == 'densenet121'
    assert c['pretrained'] is False and c['num_classes'] == 2
    assert c['image_size'] == 224
    assert c['reuse_split_files'] is True
    assert Path(c['output_dir']).name == 'sup006'
    if c['run_type'] == 'centralized':
        assert c['batch_size'] == 32 and c['epochs'] == 10
        assert c['checkpoint_metric'] == 'f1'
    else:
        assert c['run_type'] == 'flower_fedavg'
        assert c['fl_algorithm'] == 'fedavg'
        assert c['fl_num_clients'] == 5 and c['fl_num_rounds'] == 50
        assert c['fl_local_epochs'] == 1 and c['batch_size'] == 16
        assert c['sup008_audit'] is True  # legacy mechanism reused, NOT SUP008 results
print('PASS: syntax, random-init DenseNet 2 logits/727 arrays, finite forward,')
print('      strict state reconstruction, ResNet-18 default, 6 YAML contracts')
print('PENDING: Flower integration, actual RunPod versions, CUDA/pilot, real splits')
