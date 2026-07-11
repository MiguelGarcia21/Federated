from pathlib import Path
from typing import Dict

import numpy as np
HEATMAP_EPSILON = 1e-12


def load_heatmap(path: str | Path) -> np.ndarray:
    """Load and validate a saved two-dimensional heatmap."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Heatmap file not found: {path}")

    try:
        heatmap = np.load(
            path,
            allow_pickle=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load heatmap from {path}: {exc}"
        ) from exc

    if heatmap.ndim != 2:
        raise ValueError(
            f"Expected a 2D heatmap at {path}, "
            f"but found shape {heatmap.shape}."
        )

    if not np.isfinite(heatmap).all():
        raise ValueError(
            f"Heatmap contains NaN or infinite values: {path}"
        )

    return heatmap.astype(
        np.float32,
        copy=False,
    )


def validate_heatmap_pair(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
) -> None:
    """Validate two heatmaps before comparing them."""
    if heatmap_a.ndim != 2 or heatmap_b.ndim != 2:
        raise ValueError(
            "Both heatmaps must be two-dimensional."
        )

    if heatmap_a.shape != heatmap_b.shape:
        raise ValueError(
            "Heatmap shapes must match. "
            f"Received {heatmap_a.shape} and {heatmap_b.shape}."
        )

    if not np.isfinite(heatmap_a).all():
        raise ValueError(
            "First heatmap contains non-finite values."
        )

    if not np.isfinite(heatmap_b).all():
        raise ValueError(
            "Second heatmap contains non-finite values."
        )


def pearson_correlation(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
) -> float:
    """Calculate Pearson correlation safely.

    A flat heatmap has no variation, so Pearson correlation is undefined.
    """
    flat_a = np.asarray(
        heatmap_a,
        dtype=np.float64,
    ).ravel()

    flat_b = np.asarray(
        heatmap_b,
        dtype=np.float64,
    ).ravel()

    if flat_a.shape != flat_b.shape:
        raise ValueError(
            "Heatmaps must have the same number of values for "
            f"Pearson correlation: {flat_a.shape} vs {flat_b.shape}"
        )

    if (
        np.std(flat_a) <= HEATMAP_EPSILON
        or np.std(flat_b) <= HEATMAP_EPSILON
    ):
        return float("nan")

    return float(
        np.corrcoef(flat_a, flat_b)[0, 1]
    )

def cosine_similarity(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
) -> float:
    """Calculate cosine similarity safely."""
    flat_a = np.asarray(
        heatmap_a,
        dtype=np.float64,
    ).ravel()

    flat_b = np.asarray(
        heatmap_b,
        dtype=np.float64,
    ).ravel()

    if flat_a.shape != flat_b.shape:
        raise ValueError(
            "Heatmaps must have the same number of values for "
            f"cosine similarity: {flat_a.shape} vs {flat_b.shape}"
        )

    norm_a = float(np.linalg.norm(flat_a))
    norm_b = float(np.linalg.norm(flat_b))

    if (
        norm_a <= HEATMAP_EPSILON
        or norm_b <= HEATMAP_EPSILON
    ):
        return float("nan")

    return float(
        np.dot(flat_a, flat_b)
        / (norm_a * norm_b)
    )


def mean_absolute_error(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
) -> float:
    """Compute mean absolute pixel difference."""
    validate_heatmap_pair(
        heatmap_a,
        heatmap_b,
    )

    return float(
        np.mean(
            np.abs(
                heatmap_a - heatmap_b
            )
        )
    )


def mean_squared_error(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
) -> float:
    """Compute mean squared pixel difference."""
    validate_heatmap_pair(
        heatmap_a,
        heatmap_b,
    )

    difference = heatmap_a - heatmap_b

    return float(
        np.mean(
            difference ** 2
        )
    )


def threshold_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert a normalized heatmap to a boolean salient-region mask."""
    if heatmap.ndim != 2:
        raise ValueError(
            "Heatmap must be two-dimensional."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if not np.isfinite(heatmap).all():
        raise ValueError(
            "Heatmap contains non-finite values."
        )

    return heatmap >= threshold


def intersection_over_union(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute IoU between thresholded salient regions."""
    validate_heatmap_pair(
        heatmap_a,
        heatmap_b,
    )

    mask_a = threshold_heatmap(
        heatmap_a,
        threshold,
    )
    mask_b = threshold_heatmap(
        heatmap_b,
        threshold,
    )

    intersection = int(
        np.logical_and(
            mask_a,
            mask_b,
        ).sum()
    )

    union = int(
        np.logical_or(
            mask_a,
            mask_b,
        ).sum()
    )

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


def dice_score(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Dice similarity between thresholded salient regions."""
    validate_heatmap_pair(
        heatmap_a,
        heatmap_b,
    )

    mask_a = threshold_heatmap(
        heatmap_a,
        threshold,
    )
    mask_b = threshold_heatmap(
        heatmap_b,
        threshold,
    )

    intersection = int(
        np.logical_and(
            mask_a,
            mask_b,
        ).sum()
    )

    total_positive_pixels = int(
        mask_a.sum() + mask_b.sum()
    )

    if total_positive_pixels == 0:
        return 1.0

    return float(
        2.0 * intersection
        / total_positive_pixels
    )


def compare_heatmaps(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the initial Phase 6 heatmap-stability metrics."""
    validate_heatmap_pair(
        heatmap_a,
        heatmap_b,
    )

    return {
        "pearson_correlation": pearson_correlation(
            heatmap_a,
            heatmap_b,
        ),
        "cosine_similarity": cosine_similarity(
            heatmap_a,
            heatmap_b,
        ),
        "mean_absolute_error": mean_absolute_error(
            heatmap_a,
            heatmap_b,
        ),
        "mean_squared_error": mean_squared_error(
            heatmap_a,
            heatmap_b,
        ),
        "intersection_over_union": intersection_over_union(
            heatmap_a,
            heatmap_b,
            threshold=threshold,
        ),
        "dice_score": dice_score(
            heatmap_a,
            heatmap_b,
            threshold=threshold,
        ),
        "threshold": float(threshold),
        "heatmap_a_mean": float(np.mean(heatmap_a)),
        "heatmap_b_mean": float(np.mean(heatmap_b)),
        "heatmap_a_std": float(np.std(heatmap_a)),
        "heatmap_b_std": float(np.std(heatmap_b)),
        "heatmap_a_max": float(np.max(heatmap_a)),
        "heatmap_b_max": float(np.max(heatmap_b)),
        "is_degenerate_pair": bool(
            np.std(heatmap_a) <= HEATMAP_EPSILON
            or np.std(heatmap_b) <= HEATMAP_EPSILON
        ),
    }


def compare_heatmap_files(
    heatmap_path_a: str | Path,
    heatmap_path_b: str | Path,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Load two heatmap files and compute stability metrics."""
    heatmap_a = load_heatmap(
        heatmap_path_a
    )
    heatmap_b = load_heatmap(
        heatmap_path_b
    )

    return compare_heatmaps(
        heatmap_a=heatmap_a,
        heatmap_b=heatmap_b,
        threshold=threshold,
    )