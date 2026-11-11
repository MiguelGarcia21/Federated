"""SUP-008 read-only post-training FL Round-50 prediction export.

Run ONLY after Master approves training and a completed new model exists.
Never updates a checkpoint, training metric, canonical split, or historical file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split_fingerprint(df) -> str:
    """Same sorted image_id<TAB>label<LF> method as the local preflight."""
    h = hashlib.sha256()
    pairs = ((str(r.image_id), str(int(r.binary_label))) for r in df.itertuples())
    for image_id, label in sorted(pairs):
        h.update(f'{image_id}\t{label}\n'.encode('utf-8'))
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Export SUP008 new FedAvg Round-50 predictions (no training).')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    # Imports deliberately follow argparse so --help works without ML dependencies.
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import yaml
    from src.data.preprocess_vindr import load_existing_full_split
    from src.models.cnn import create_model
    from src.training.evaluate import evaluate_model
    from src.training.train_flower_fedavg import (
        compute_binary_class_weights, make_loader_from_df,
    )

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Expected a YAML mapping')
    exp = str(config['experiment_id'])
    if (not exp.startswith('g4_sup008_')
            or config.get('run_type') != 'flower_fedavg'
            or config.get('split_type') not in ('iid', 'non_iid_label_skew')
            or config.get('sup008_audit') is not True
            or config.get('reuse_split_files') is not True
            or int(config.get('fl_num_rounds', 0)) != 50
            or int(config.get('fl_num_clients', 0)) != 5
            or str(config.get('image_source')) != 'cache'
            or str(config.get('class_weight_scope')) != (
                'local' if config.get('split_type') == 'iid' else 'global'
            )
            or any(k in config for k in ('sup003_condition','sup003_gamma_by_client','sup007_condition'))):
        raise ValueError('Not an approved original-condition SUP008 FL configuration')
    if int(config.get('seed', -1)) not in (2026, 31415):
        raise ValueError('Unexpected new seed')
    if (str(config.get('model_name')) != 'resnet18'
            or int(config.get('num_classes', 0)) != 2
            or config.get('pretrained') is not False):
        raise ValueError('Unexpected model configuration')

    output_root = Path(config['output_dir'])
    checkpoint_path = output_root / 'checkpoints' / exp / 'flower_fedavg_round_50.pt'
    metrics_path = output_root / 'metrics' / f'{exp}_flower_fedavg_metrics.csv'
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError('New Round-50 checkpoint and metrics CSV are required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable')
    device = torch.device(args.device)

    # Only load model tensors plus primitive metadata, never untrusted pickle code.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('round') != 50:
        raise ValueError('Checkpoint is not a Round-50 training snapshot')
    checkpoint_cfg = checkpoint.get('config')
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError('Checkpoint lacks embedded configuration')
    identity_keys = (
        'experiment_id', 'seed', 'split_type', 'class_weight_scope',
        'model_name', 'num_classes', 'pretrained', 'image_size',
        'image_source', 'image_cache_dir', 'canonical_train_split_path',
        'canonical_val_split_path', 'batch_size', 'learning_rate',
        'weight_decay', 'fl_num_rounds', 'fl_num_clients', 'fl_local_epochs',
    )
    for key in identity_keys:
        if checkpoint_cfg.get(key) != config.get(key):
            raise ValueError(f'Checkpoint/config mismatch: {key}')

    train_df, val_df, _ = load_existing_full_split(config)
    if len(train_df) != 12000 or len(val_df) != 3000:
        raise RuntimeError('Canonical train/validation lengths incorrect')
    expected_fingerprints = {
        'train': '5107ccd6639059d4fc66bf190f27fea727ecfa96f82e8b0033f9732accb4eed6',
        'validation': '60bbb6632f791d81eb46fbb3f9eef272f29005d431ec00282773a95c4de0a756',
    }
    if (split_fingerprint(train_df) != expected_fingerprints['train']
            or split_fingerprint(val_df) != expected_fingerprints['validation']):
        raise RuntimeError('Canonical split fingerprint differs from approved preflight')

    model = create_model(model_name='resnet18', num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device)
    val_loader = make_loader_from_df(config=config, df=val_df, shuffle=False)
    weights = compute_binary_class_weights(train_df, 'Complete federated training data')
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    observed, predictions = evaluate_model(
        model=model, loader=val_loader, criterion=criterion,
        device=device, return_predictions=True,
    )
    if len(predictions) != 3000 or predictions['image_id'].astype(str).nunique() != 3000:
        raise RuntimeError('Expected 3000 distinct per-image predictions')
    canonical = {str(r.image_id): int(r.binary_label) for r in val_df.itertuples()}
    actual = {str(r.image_id): int(r.true_label) for r in predictions.itertuples()}
    if actual != canonical:
        raise RuntimeError('Prediction image IDs/labels differ from canonical validation')
    probs = predictions['positive_probability'].to_numpy(dtype=float)
    confidences = predictions['prediction_confidence'].to_numpy(dtype=float)
    if (not np.isfinite(probs).all() or not np.isfinite(confidences).all()
            or np.any((probs < 0) | (probs > 1))
            or np.any((confidences < 0.5) | (confidences > 1))):
        raise RuntimeError('Non-finite/out-of-range probabilities')
    predicted = predictions['predicted_label'].to_numpy(dtype=int)
    if np.any(predicted != np.where(probs > 0.5, 1, 0)):
        raise RuntimeError('Predicted class differs from positive probability argmax')

    table = pd.read_csv(metrics_path)
    if 'round' not in table.columns:
        raise ValueError('Historical-style metrics CSV lacks round')
    row50 = table.loc[pd.to_numeric(table['round']) == 50]
    if len(row50) != 1:
        raise RuntimeError('Expected exactly one Round-50 metrics row')
    ref = row50.iloc[0]
    if (str(ref['experiment_id']) != exp or int(ref['seed']) != int(config['seed'])
            or int(ref['num_validation_samples']) != 3000
            or int(ref['num_clients']) != 5
            or str(ref['split_type']) != str(config['split_type'])
            or str(ref['class_weight_scope']) != str(config['class_weight_scope'])):
        raise RuntimeError('Metrics metadata differs from run configuration')

    differences = {}
    for metric in (
        'accuracy','precision','recall','specificity','f1','auc',
        'average_precision','balanced_accuracy','negative_predictive_value',
        'false_positive_rate','false_negative_rate','mcc','brier_score',
        'mean_positive_probability','mean_prediction_confidence',
        'positive_prediction_rate',
    ):
        if metric not in row50.columns:
            raise RuntimeError(f'Metric missing from recorded Round-50 metrics: {metric}')
        a, b = float(observed[metric]), float(ref[metric])
        if not math.isfinite(a) or not math.isfinite(b):
            raise RuntimeError(f'Nonfinite metric {metric}')
        differences[metric] = abs(a - b)
        if differences[metric] > 1e-5:
            raise RuntimeError(f'Metric mismatch {metric}: recomputed {a}, logged {b}')
    if 'centralized_loss' not in row50.columns:
        raise RuntimeError('Missing centralized_loss')
    loss_delta = abs(float(observed['loss']) - float(ref['centralized_loss']))
    if not math.isfinite(loss_delta) or loss_delta > 1e-4:
        raise RuntimeError('Round-50 loss discrepancy')

    evidence_dir = output_root / 'sup008_evidence' / exp
    csv_path = evidence_dir / f'{exp}_R50_predictions.csv'
    report_path = evidence_dir / f'{exp}_R50_metrics.json'
    if csv_path.exists() or report_path.exists():
        raise FileExistsError('SUP008 evidence output already exists; do not overwrite')
    evidence_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(csv_path, index=False)
    report = {
        'experiment_id': exp,
        'seed': int(config['seed']),
        'round': 50,
        'n_predictions': 3000,
        'device': str(device),
        'canonical_train_fingerprint': expected_fingerprints['train'],
        'canonical_val_fingerprint': expected_fingerprints['validation'],
        'source_sha256': digest(Path(__file__)),
        'config_sha256': digest(config_path),
        'checkpoint_sha256': digest(checkpoint_path),
        'logged_metrics_sha256': digest(metrics_path),
        'predictions_sha256': digest(csv_path),
        'recomputed_metrics': {k: float(v) for k, v in observed.items()},
        'absolute_metric_differences': differences,
        'absolute_loss_difference': loss_delta,
        'tolerance': {'metrics': 1e-5, 'loss': 1e-4},
        'qa_status': 'PASS',
    }
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f'PASS: {exp}, 3000 predictions, QA at {report_path}')


if __name__ == '__main__':
    main()
