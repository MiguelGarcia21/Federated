#!/usr/bin/env python3
r"""SUP-001: frozen nine-model inference on the full RSNA Stage-2 labeled cohort.

Stand-alone coordination script: place manually in the Federated repository root.
Does not edit repository modules, DICOMs, manifests, configurations or checkpoints.
Only creates new results in data-dir/sup001_full_inference (unless --out-dir).
Predictions are NOT a clinical result and are not yet independently metric-audited.

From thesis/Federated on Windows (CPU):
  python .\G2_SUP001_RSNA_full_inference.py --repo-root . --data-dir ..\test \
    --checkpoints-root ..\complete_checkpoints\checkpoints --device cpu --batch-size 16
On a CUDA environment with the same data, repository and checkpoints:
  python G2_SUP001_RSNA_full_inference.py --repo-root /workspace/Federated \
    --data-dir /workspace/test --checkpoints-root /workspace/checkpoints --device cuda --batch-size 32
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_SHA = 'b1268958d49b9d4d407bb1511f6fc1412c3c41931837cf1a1007d4725c3a2bf6'
DATASETS_SHA = '4a6b7c376058bcf2288e489be4b2f028f245f1f4cb10833abd097ea49b56ff6b'
MODEL_SHA = '47187e7ef7be179414d8c2493732dea1177e227a0813654f9c72768c15c0ef34'
CATEGORY_COUNTS = {'Normal': 8851, 'Lung Opacity': 6012, 'No Lung Opacity / Not Normal': 11821}
SPECS = (
    ('Centralized', 42, 'exp01_centralized_vindr_binary_seed42', 'centralized_resnet18_debug_best.pt', 'epoch', 2),
    ('Centralized', 7, 'exp01_centralized_vindr_binary_seed7', 'centralized_resnet18_best.pt', 'epoch', 10),
    ('Centralized', 123, 'exp01_centralized_vindr_binary_seed123', 'centralized_resnet18_best.pt', 'epoch', 9),
    ('IID', 42, 'exp02_fedavg_iid_5clients_seed42', 'flower_fedavg_round_50.pt', 'round', 50),
    ('IID', 7, 'exp02_fedavg_iid_5clients_seed7', 'flower_fedavg_round_50.pt', 'round', 50),
    ('IID', 123, 'exp02_fedavg_iid_5clients_seed123', 'flower_fedavg_round_50.pt', 'round', 50),
    ('Non-IID', 42, 'exp03_fedavg_noniid_5clients_seed42', 'flower_fedavg_round_50.pt', 'round', 50),
    ('Non-IID', 7, 'exp03_fedavg_noniid_5clients_seed7', 'flower_fedavg_round_50.pt', 'round', 50),
    ('Non-IID', 123, 'exp03_fedavg_noniid_5clients_seed123', 'flower_fedavg_round_50.pt', 'round', 50),
)
FIELDS = ('condition','seed','image_id','rsna_detailed_class','true_binary_label',
          'predicted_binary_label','probability_abnormal','confidence_predicted_class','checkpoint_sha256')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest().lower()


def read_manifest(data_dir: Path):
    path = data_dir / 'sup001_cohort_audit' / 'rsna_stage2_external_binary_manifest.csv'
    meta = path.with_name('rsna_stage2_external_cohort_audit.json')
    if not path.is_file() or not meta.is_file():
        raise FileNotFoundError('Frozen cohort manifest/audit JSON not found')
    if sha256(path) != MANIFEST_SHA:
        raise ValueError('Frozen manifest hash differs from approved cohort')
    if json.loads(meta.read_text(encoding='utf-8'))['sha256'][path.name] != MANIFEST_SHA:
        raise ValueError('Cohort audit and manifest hash do not match')
    result = []
    seen = set()
    counts = Counter()
    with path.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        required = {'rsna_exam_id','dicom_filename','rsna_detailed_class','binary_abnormal_label'}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError('Frozen manifest schema not recognized')
        for row in reader:
            iid = row['rsna_exam_id'].strip()
            fn = row['dicom_filename'].strip()
            cat = row['rsna_detailed_class'].strip()
            lab = row['binary_abnormal_label'].strip()
            if not iid or iid in seen or fn != iid + '.dcm' or '/' in fn or '\\' in fn:
                raise ValueError('Duplicate or unexpected image ID/file name in frozen manifest')
            if cat not in CATEGORY_COUNTS or lab != str(int(cat != 'Normal')):
                raise ValueError('Unapproved category or label assignment')
            seen.add(iid)
            counts[cat] += 1
            result.append((iid, fn, cat, int(lab)))
    if len(result) != 26684 or dict(counts) != CATEGORY_COUNTS:
        raise ValueError(f'Cohort size/class distribution unexpected: {len(result)}, {dict(counts)}')
    images = data_dir / 'stage_2_train_images'
    if not images.is_dir():
        raise FileNotFoundError(f'RSNA DICOM directory missing: {images}')
    for iid, fn, _, _ in result:
        p = images / fn
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f'Missing or empty DICOM: {p}')
    return result


def check_models(root: Path):
    qa_file = root / 'sup001_checkpoint_structure_qa.json'
    inventory_file = root / 'sup001_checkpoint_provenance.csv'
    if not qa_file.is_file() or not inventory_file.is_file():
        raise FileNotFoundError('Missing approved checkpoint QA JSON/provenance CSV')
    qa = json.loads(qa_file.read_text(encoding='utf-8'))
    if qa.get('status') != 'CHECKPOINT_STRUCTURE_QA_PASS' or qa.get('passed') != 9 or qa.get('failed') != 0:
        raise ValueError('Checkpoint structure QA is not an accepted 9/9 PASS')
    checks = {(r['condition'], int(r['seed'])):r for r in qa.get('results', [])}
    with inventory_file.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    inv = {(r['Condition'], int(r['Seed'])):r for r in rows}
    expected = {(c,s) for c,s,*_ in SPECS}
    if set(checks) != expected or set(inv) != expected or len(rows) != 9:
        raise ValueError('The checkpoint QA/inventory does not contain exactly nine expected identities')
    verified = {}
    for c,s,folder,filename,which,number in SPECS:
        meta, record = checks[(c,s)], inv[(c,s)]
        ck = root / folder / filename
        cfg = root / folder / 'config.yaml'
        if meta['status'] != 'PASS' or not all(meta[k] for k in
            ('strict_model_load','synthetic_forward_pass','file_hash_verified','config_hash_verified')):
            raise ValueError(f'Prior structure QA not approved: {c} seed {s}')
        if (meta['folder'],meta['checkpoint']) != (folder,filename) or (record['Folder'],record['Checkpoint']) != (folder,filename):
            raise ValueError(f'Checkpoint identity mismatch: {c} seed {s}')
        if meta['metadata'][which] != number:
            raise ValueError(f'Checkpoint epoch/round mismatch: {c} seed {s}')
        if not ck.is_file() or not cfg.is_file() or ck.stat().st_size != int(record['SizeBytes']):
            raise ValueError(f'Missing or altered checkpoint/config: {c} seed {s}')
        if sha256(ck) != record['CheckpointSHA256'].strip().lower():
            raise ValueError(f'Changed checkpoint hash: {c} seed {s}')
        if sha256(cfg) != record['ConfigSHA256'].strip().lower():
            raise ValueError(f'Changed config hash: {c} seed {s}')
        verified[(c,s)] = {'path':ck,'sha256':sha256(ck),'metadata_key':which,'metadata_value':number}
    return verified


def verify_smoke(data_dir: Path, predictions: Path) -> dict:
    smoke_dir = data_dir / 'sup001_inference_smoke_qa'
    sm_csv = smoke_dir / 'sup001_rsna_smoke_predictions.csv'
    sm_json = smoke_dir / 'sup001_rsna_smoke_qa.json'
    if not sm_csv.is_file() or not sm_json.is_file():
        raise FileNotFoundError('Accepted smoke CSV and JSON not found under data-dir/sup001_inference_smoke_qa')
    meta = json.loads(sm_json.read_text(encoding='utf-8'))
    if (meta.get('status') != 'RSNA_NINE_MODEL_SMOKE_QA_PASS' or
        meta.get('predictions_written') != 135 or
        meta.get('rsna_frozen_manifest_sha256') != MANIFEST_SHA or
        meta.get('predictions_csv_sha256') != sha256(sm_csv)):
        raise ValueError('Accepted smoke test does not match current sample')
    smoked = {}
    with sm_csv.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            key = (r['condition'],int(r['seed']),r['image_id'])
            if key in smoked:
                raise ValueError('Duplicate smoke prediction key')
            smoked[key] = r
    if len(smoked) != 135:
        raise ValueError('Smoke CSV is not complete')
    matched = 0
    max_delta = 0.0
    changed_predictions = 0
    with predictions.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            key = (r['condition'],int(r['seed']),r['image_id'])
            if key not in smoked:
                continue
            s = smoked.pop(key)
            if (s['true_binary_label'],s['rsna_detailed_class'],s['checkpoint_sha256']) != (r['true_binary_label'],r['rsna_detailed_class'],r['checkpoint_sha256']):
                raise ValueError(f'Smoke-to-full labels/checkpoint metadata mismatch at {key}')
            delta = abs(float(s['probability_abnormal']) - float(r['probability_abnormal']))
            max_delta = max(max_delta,delta)
            changed_predictions += (s['predicted_binary_label'] != r['predicted_binary_label'])
            matched += 1
    if smoked or matched != 135:
        raise ValueError(f'Full inference omitted smoke prediction(s): {len(smoked)} missing')
    if max_delta > 1e-3:
        raise ValueError(f'Smoke/full inference probability divergence exceeds 0.001: {max_delta}')
    return {'smoke_rows_compared': matched, 'max_probability_difference':max_delta,
            'predicted_class_differences':changed_predictions,
            'tolerance_probability':0.001,
            'note':'Minor CPU/GPU floating point discrepancies can occur; investigate any class differences.'}


def audit_pred_file(path: Path, cohort, specs):
    byid = {r[0]: (r[2],r[3]) for r in cohort}
    expected = {(c,s) for c,s,*_ in specs}
    seen = set()
    counts = Counter()
    n = 0
    min_prob = 1.0
    max_prob = 0.0
    by_model_pred = defaultdict(Counter)
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if list(reader.fieldnames or []) != list(FIELDS):
            raise ValueError('Full predictions schema mismatch')
        for row in reader:
            identity = (row['condition'],int(row['seed']))
            iid = row['image_id']
            key = (*identity, iid)
            if identity not in expected or iid not in byid or key in seen:
                raise ValueError(f'Unexpected/duplicate model-image prediction key: {key}')
            seen.add(key)
            cat,label = byid[iid]
            p = float(row['probability_abnormal'])
            conf = float(row['confidence_predicted_class'])
            pred = int(row['predicted_binary_label'])
            if (row['rsna_detailed_class'],int(row['true_binary_label'])) != (cat,label):
                raise ValueError('Full inference class/ground truth mismatch')
            if pred not in (0,1) or not math.isfinite(p) or not math.isfinite(conf) or not (0 <= p <= 1):
                raise ValueError('Invalid probability or prediction value')
            if abs(conf - (p if pred else 1-p)) > 1e-5 or pred != int(p > 0.5):
                # Exact tie is possible; torch.argmax returns class 0.
                raise ValueError('Prediction/confidence/probability inconsistency')
            min_prob = min(min_prob,p)
            max_prob = max(max_prob,p)
            counts[identity] += 1
            by_model_pred[identity][pred] += 1
            n += 1
    if n != 9*len(cohort) or len(seen) != n or set(counts) != expected or any(c != len(cohort) for c in counts.values()):
        raise ValueError(f'Full output incomplete: {n}, expected {9*len(cohort)}')
    return {'rows':n,'unique_model_image_keys':len(seen),'images_per_model':len(cohort),
            'models':len(expected),'min_probability_abnormal':min_prob,
            'max_probability_abnormal':max_prob,
            'per_model_predictions':[
                {'condition':c,'seed':s,'rows':counts[(c,s)],
                 'predicted_normal':by_model_pred[(c,s)][0],
                 'predicted_abnormal':by_model_pred[(c,s)][1]}
                for c,s,*_ in specs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--checkpoints-root', required=True, type=Path)
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch-size >=1 and num-workers >=0 required')
    data_dir = args.data_dir.resolve()
    repo_root = args.repo_root.resolve()
    ck_root = args.checkpoints_root.resolve()
    out_dir = (args.out_dir or data_dir / 'sup001_full_inference').resolve()
    dest_csv = out_dir / 'sup001_rsna_full_predictions.csv'
    dest_report = out_dir / 'sup001_rsna_full_inference_qa.json'
    partial_csv = out_dir / 'sup001_rsna_full_predictions.csv.partial'
    if dest_csv.exists() or dest_report.exists() or partial_csv.exists():
        raise FileExistsError(f'Output already exists in {out_dir}; keep previous outputs intact, choose a fresh --out-dir')

    cohort = read_manifest(data_dir)
    verified = check_models(ck_root)
    ds_path = repo_root / 'src/data/datasets.py'
    model_path = repo_root / 'src/models/cnn.py'
    if not ds_path.is_file() or sha256(ds_path) != DATASETS_SHA:
        raise ValueError('Repository dataset loader changed; repeat preprocessing review before inference')
    if not model_path.is_file() or sha256(model_path) != MODEL_SHA:
        raise ValueError('Repository ResNet model changed; repeat model audit before inference')
    # Require accepted smoke QA before starting the full run.
    smoke_root = data_dir / 'sup001_inference_smoke_qa'
    smoke_json = smoke_root / 'sup001_rsna_smoke_qa.json'
    smoke_csv = smoke_root / 'sup001_rsna_smoke_predictions.csv'
    if not smoke_json.is_file() or not smoke_csv.is_file():
        raise FileNotFoundError('Accepted smoke test report and predictions are required')
    smoke_meta = json.loads(smoke_json.read_text(encoding='utf-8'))
    if (smoke_meta.get('status') != 'RSNA_NINE_MODEL_SMOKE_QA_PASS' or
        smoke_meta.get('predictions_written') != 135 or
        smoke_meta.get('predictions_csv_sha256') != sha256(smoke_csv)):
        raise ValueError('Smoke evidence is not an accepted 135-row PASS')

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0,str(repo_root))
    from src.data.datasets import VinDrBinaryDataset, get_val_transform
    from src.models.cnn import create_model
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but CUDA is unavailable')
    device = torch.device('cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    if device.type == 'cpu':
        print('Device: CPU. For GPU use a CUDA-enabled PyTorch environment and --device cuda.',flush=True)
    else:
        print('Device:',torch.cuda.get_device_name(device),flush=True)

    frame = pd.DataFrame([
        {'image_id':iid, 'binary_label':lab, 'dicom_path':str(data_dir/'stage_2_train_images'/fn)}
        for iid,fn,cat,lab in cohort])
    ds = VinDrBinaryDataset(frame,transform=get_val_transform(224),image_source='dicom')
    loader = DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
                        pin_memory=(device.type=='cuda'),drop_last=False)
    expected_order = [r[0] for r in cohort]
    lookup = {r[0]:(r[2],r[3]) for r in cohort}

    # Nine frozen models stay resident. Each decoded input batch is shared identically.
    models = []
    for c,s,folder,filename,which,number in SPECS:
        ck = torch.load(verified[(c,s)]['path'],map_location='cpu',weights_only=True)
        if not isinstance(ck,dict) or 'model_state_dict' not in ck or int(ck[which]) != number:
            raise ValueError(f'Checkpoint content altered since structure QA: {c} seed {s}')
        model = create_model(model_name='resnet18',num_classes=2,pretrained=False)
        model.load_state_dict(ck['model_state_dict'],strict=True)
        model.eval().to(device)
        models.append((c,s,verified[(c,s)]['sha256'],model))
        del ck
    print('Loaded and verified nine frozen models. Starting full labeled RSNA cohort:',len(cohort),flush=True)

    out_dir.mkdir(parents=True,exist_ok=True)
    processed = 0
    try:
        with partial_csv.open('x',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=FIELDS)
            writer.writeheader()
            with torch.inference_mode():
                for tensors, labels, ids in loader:
                    if (tensors.dtype != torch.float32 or tuple(tensors.shape[1:]) != (3,224,224)
                            or not torch.isfinite(tensors).all().item()):
                        raise ValueError(f'Non-finite or malformed model inputs at index {processed}')
                    if list(ids) != expected_order[processed:processed+len(ids)]:
                        raise ValueError(f'DataLoader order mismatch at index {processed}')
                    if any(int(label) != lookup[iid][1] for label,iid in zip(labels.tolist(),ids)):
                        raise ValueError(f'Labels changed during loading at index {processed}')
                    x = tensors.to(device,non_blocking=(device.type=='cuda'))
                    for condition,seed,digest,model in models:
                        logits = model(x)
                        if tuple(logits.shape) != (len(ids),2) or not torch.isfinite(logits).all().item():
                            raise ValueError(f'Invalid logits for {condition} seed {seed} at {processed}')
                        probs = torch.softmax(logits.float(),dim=1)
                        if not torch.isfinite(probs).all().item() or not torch.allclose(
                              probs.sum(dim=1),torch.ones(len(ids),device=device),atol=1e-5,rtol=0):
                            raise ValueError(f'Invalid probabilities for {condition} seed {seed}')
                        ps = probs[:,1].cpu().tolist()
                        preds = probs.argmax(dim=1).cpu().tolist()
                        for iid,pred,p in zip(ids,preds,ps):
                            cat,lab = lookup[iid]
                            writer.writerow({'condition':condition,'seed':seed,'image_id':iid,
                                'rsna_detailed_class':cat,'true_binary_label':lab,
                                'predicted_binary_label':pred,'probability_abnormal':p,
                                'confidence_predicted_class':p if pred==1 else 1-p,
                                'checkpoint_sha256':digest})
                    processed += len(ids)
                    if processed%1000 < len(ids) or processed == len(cohort):
                        print(f'Processed {processed}/{len(cohort)} distinct images across all nine models',flush=True)
        if processed != len(cohort):
            raise ValueError(f'Only processed {processed}/{len(cohort)} images')
        summary = audit_pred_file(partial_csv,cohort,SPECS)
        smoke_summary = verify_smoke(data_dir,partial_csv)
        pred_sha = sha256(partial_csv)
        os.replace(partial_csv,dest_csv)
        report = {'status':'RSNA_FULL_NINE_MODEL_INFERENCE_QA_PASS',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Full frozen RSNA Stage-2 image/exam cohort; predictive metrics NOT yet independently audited',
            'rsna_frozen_manifest_sha256':MANIFEST_SHA,
            'datasets_module_sha256':DATASETS_SHA,'model_module_sha256':MODEL_SHA,
            'device':str(device),'torch_version':torch.__version__,
            'batch_size':args.batch_size,'num_workers':args.num_workers,
            'cohort_class_counts':CATEGORY_COUNTS,
            'mapping':{'Normal':0,'Lung Opacity':1,'No Lung Opacity / Not Normal':1},
            'output_csv':str(dest_csv),'output_sha256':pred_sha,
            'verified_checkpoints':[
                {'condition':c,'seed':s,'filename':filename,'sha256':verified[(c,s)]['sha256']}
                for c,s,folder,filename,which,number in SPECS],
            'output_audit':summary,'smoke_consistency':smoke_summary,
            'caveats':['RSNA label ontology differs from VinDr-CXR.',
                'RSNA challenge class distribution is enriched and not population prevalence.',
                'Image/exam IDs cannot establish independent patient groups.',
                'Historical centralized best-F1 checkpoint selection requires separate training metrics evidence.',
                'This report validates technical execution only; independent metrics verification follows.']}
        dest_report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(f'\nResult: {report["status"]}',flush=True)
        print('Predictions:',dest_csv,flush=True)
        print('Report:',dest_report,flush=True)
    except Exception:
        print(f'FAILED. Partial output (if present) left for diagnosis: {partial_csv}',file=sys.stderr)
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'RSNA FULL INFERENCE FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
