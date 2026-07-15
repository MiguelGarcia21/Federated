from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_one_epoch(model, loader, criterion, optimizer, device) -> Dict[str, float]:
    """Train the model for one epoch.

    This preserves the Phase 5 behavior and return structure.
    Additional training-side metrics can be added later if they become
    necessary for a specific experiment.
    """
    model.train()

    total_loss_numerator = 0.0
    total_loss_denominator = 0.0

    all_preds = []
    all_labels = []

    for images, labels, _ in loader:
        images = images.to(
        device,
        non_blocking=True,
    )
    labels = labels.to(
        device,
        non_blocking=True,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    if criterion.weight is None:
        batch_loss_denominator = float(
            labels.numel()
        )
    else:
        batch_loss_denominator = float(
            criterion.weight[labels]
            .sum()
            .item()
        )

    total_loss_numerator += (
        float(loss.item())
        * batch_loss_denominator
    )

    total_loss_denominator += (
        batch_loss_denominator
    )

    preds = torch.argmax(logits, dim=1)

    all_preds.extend(preds.detach().cpu().numpy())
    all_labels.extend(labels.detach().cpu().numpy())

    if len(loader.dataset) == 0:
        raise ValueError("Training loader contains no samples.")

    if total_loss_denominator <= 0.0:
        raise ValueError(
            "Training loss denominator must be greater than zero."
        )

    avg_loss = (
        total_loss_numerator
        / total_loss_denominator
    )
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
    }


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    return_predictions: bool = False,
) -> Dict[str, Any] | Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate a binary classifier and return thesis-ready metrics.

    The original Phase 5 metric keys are preserved:

    - loss
    - accuracy
    - precision
    - recall
    - f1
    - auc

    Additional metrics are included for class imbalance, threshold behavior,
    calibration, error analysis, and later federated/XAI comparisons.

    Parameters
    ----------
    model:
        PyTorch classification model.

    loader:
        DataLoader yielding:
            images, labels, image_ids

    criterion:
        Loss function used for validation loss.

    device:
        PyTorch device.

    return_predictions:
        When False, return only the metrics dictionary.
        When True, return:
            metrics, predictions_dataframe

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary when return_predictions=False.

    Tuple[Dict[str, Any], pd.DataFrame]
        Metrics and per-image predictions when return_predictions=True.
    """
    model.eval()

    if len(loader.dataset) == 0:
        raise ValueError("Evaluation loader contains no samples.")

    total_loss_numerator = 0.0
    total_loss_denominator = 0.0

    all_preds = []
    all_labels = []
    all_positive_probs = []
    all_confidences = []
    all_image_ids = []

    for images, labels, image_ids in loader:
        images = images.to(
        device,
        non_blocking=True,
    )
    labels = labels.to(
        device,
        non_blocking=True,
    )

    logits = model(images)
    loss = criterion(logits, labels)

    probabilities = torch.softmax(logits, dim=1)
    positive_probs = probabilities[:, 1]
    preds = torch.argmax(probabilities, dim=1)
    confidences = probabilities.max(dim=1).values

    if criterion.weight is None:
        batch_loss_denominator = float(
            labels.numel()
        )
    else:
        batch_loss_denominator = float(
            criterion.weight[labels]
            .sum()
            .item()
        )

    total_loss_numerator += (
        float(loss.item())
        * batch_loss_denominator
    )

    total_loss_denominator += (
        batch_loss_denominator
    )

    all_preds.extend(preds.detach().cpu().numpy().tolist())
    all_labels.extend(labels.detach().cpu().numpy().tolist())
    all_positive_probs.extend(
            positive_probs.detach().cpu().numpy().tolist()
        )
    all_confidences.extend(
            confidences.detach().cpu().numpy().tolist()
        )
    all_image_ids.extend(list(image_ids))

    labels_array = np.asarray(all_labels, dtype=np.int64)
    preds_array = np.asarray(all_preds, dtype=np.int64)
    probs_array = np.asarray(all_positive_probs, dtype=np.float64)
    confidences_array = np.asarray(all_confidences, dtype=np.float64)

    if total_loss_denominator <= 0.0:
        raise ValueError(
            "Evaluation loss denominator must be greater than zero."
        )

    avg_loss = (
        total_loss_numerator
        / total_loss_denominator
    )

    tn, fp, fn, tp = confusion_matrix(
        labels_array,
        preds_array,
        labels=[0, 1],
    ).ravel()

    num_samples = int(len(labels_array))
    num_actual_positive = int(tp + fn)
    num_actual_negative = int(tn + fp)
    num_predicted_positive = int(tp + fp)
    num_predicted_negative = int(tn + fn)

    actual_positive_rate = (
        num_actual_positive / num_samples
        if num_samples > 0
        else np.nan
    )

    positive_prediction_rate = (
        num_predicted_positive / num_samples
        if num_samples > 0
        else np.nan
    )

    negative_prediction_rate = (
        num_predicted_negative / num_samples
        if num_samples > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    negative_predictive_value = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )

    metrics: Dict[str, Any] = {
        "loss": float(avg_loss),
        "accuracy": float(
            accuracy_score(labels_array, preds_array)
        ),
        "precision": float(
            precision_score(
                labels_array,
                preds_array,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels_array,
                preds_array,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels_array,
                preds_array,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels_array,
                preds_array,
            )
        ),
        "negative_predictive_value": float(
            negative_predictive_value
        ),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "mcc": float(
            matthews_corrcoef(
                labels_array,
                preds_array,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                labels_array,
                probs_array,
            )
        ),
        "mean_positive_probability": float(
            np.mean(probs_array)
        ),
        "mean_prediction_confidence": float(
            np.mean(confidences_array)
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "num_samples": num_samples,
        "num_actual_positive": num_actual_positive,
        "num_actual_negative": num_actual_negative,
        "actual_positive_rate": float(actual_positive_rate),
        "num_predicted_positive": num_predicted_positive,
        "num_predicted_negative": num_predicted_negative,
        "positive_prediction_rate": float(
            positive_prediction_rate
        ),
        "negative_prediction_rate": float(
            negative_prediction_rate
        ),
    }

    try:
        metrics["auc"] = float(
            roc_auc_score(
                labels_array,
                probs_array,
            )
        )
    except ValueError:
        metrics["auc"] = np.nan

    try:
        metrics["average_precision"] = float(
            average_precision_score(
                labels_array,
                probs_array,
            )
        )
    except ValueError:
        metrics["average_precision"] = np.nan

    if not return_predictions:
        return metrics

    predictions_df = pd.DataFrame({
        "image_id": all_image_ids,
        "true_label": labels_array,
        "predicted_label": preds_array,
        "positive_probability": probs_array,
        "prediction_confidence": confidences_array,
        "correct": labels_array == preds_array,
    })

    return metrics, predictions_df
