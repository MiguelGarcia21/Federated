"""Read-only RunPod environment and corpus preflight; no training or writes.

Usage from repo root (after author has manually installed proposed configs):
 python path/to/sup008_runpod_readonly_preflight.py --repo /workspace/thesis-fl-xai
"""
import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

IDS = (
 'g4_sup008_centralized_seed2026', 'g4_sup008_centralized_seed31415',
 'g4_sup008_iid_seed2026', 'g4_sup008_iid_seed31415',
 'g4_sup008_noniid_label_skew_seed2026', 'g4_sup008_noniid_label_skew_seed31415',
)


def command(*parts, cwd=None):
    try:
        p = subprocess.run(parts, cwd=cwd, capture_output=True, text=True, timeout=20)
        return {'returncode': p.returncode, 'stdout': p.stdout.strip()[:2000], 'stderr': p.stderr.strip()[:500]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def ids_at(csv_path: Path):
    with csv_path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert all(r.get('image_id') and r.get('binary_label') in ('0','1') for r in rows)
    assert len(set(r['image_id'] for r in rows)) == len(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = {'repo': str(repo), 'training_authorized': False, 'checks': {}}
    report['checks']['git_head'] = command('git','rev-parse','HEAD',cwd=repo)
    report['checks']['git_status'] = command('git','status','--short',cwd=repo)
    report['checks']['gpu'] = command('nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader',cwd=repo)
    report['python'] = sys.version.split()[0]
    packages = {}
    for dist in ('torch','torchvision','flwr','ray','numpy','pandas','scikit-learn','PyYAML','pydicom'):
        try: packages[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError: packages[dist] = None
    report['packages'] = packages
    report['checks']['gpu_cuda'] = {'verified': False}
    try:
        import torch
        report['checks']['gpu_cuda'] = {'verified': bool(torch.cuda.is_available()), 'version': torch.version.cuda}
    except Exception as exc:
        report['checks']['gpu_cuda']['error'] = str(exc)
    train = repo/'data/interim/exp01_centralized_vindr_binary_seed42_train_split.csv'
    val = repo/'data/interim/exp01_centralized_vindr_binary_seed42_val_split.csv'
    try:
        tr, va = ids_at(train), ids_at(val)
        assert len(tr)==12000 and len(va)==3000
        assert not (set(r['image_id'] for r in tr) & set(r['image_id'] for r in va))
        cache = repo/'data/processed/vindr_binary_224_png'
        ids = set(r['image_id'] for r in tr+va)
        missing = [i for i in ids if not (cache/(i+'.png')).is_file()]
        sample_dicom = [r['dicom_path'] for r in tr[:20] if 'dicom_path' in r]
        report['checks']['corpus'] = {
            'train':len(tr), 'validation':len(va), 'cache_dir_exists':cache.is_dir(),
            'cached_images_found':len(ids)-len(missing),
            'cached_images_missing':len(missing), 'missing_examples':missing[:5],
            'sample_raw_dicom_paths_existing':sum(Path(x).is_file() for x in sample_dicom),
            'sample_raw_dicom_paths_checked':len(sample_dicom),
        }
    except Exception as exc:
        report['checks']['corpus'] = {'error': str(exc)}
    report['checks']['free_disk'] = {}
    try:
        import shutil
        usage = shutil.disk_usage(repo)
        report['checks']['free_disk']={'free_gb':round(usage.free/1e9,2)}
    except Exception as exc:
        report['checks']['free_disk']={'error':str(exc)}
    outputs = repo/'outputs/sup008'
    report['checks']['would_overwrite'] = {
        k: any((outputs/sub/k).exists() for sub in ('checkpoints','sup008_evidence','heatmaps'))
           or (outputs/'metrics'/f'{k}_flower_fedavg_metrics.csv').exists()
        for k in IDS
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

if __name__=='__main__':
    sys.exit(main())
