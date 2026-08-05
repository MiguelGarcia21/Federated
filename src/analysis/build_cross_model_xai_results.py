#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS: tuple[int, ...] = (42, 7, 123)

COMPARISON_ORDER: tuple[str, ...] = (
    "centralized_vs_iid",
    "centralized_vs_non_iid_label_skew",
    "iid_vs_non_iid_label_skew",
)

COMPARISON_LABELS = {
    "centralized_vs_iid": "Centralized vs IID",
    "centralized_vs_non_iid_label_skew": "Centralized vs Non-IID",
    "iid_vs_non_iid_label_skew": "IID vs Non-IID",
}

EXPERIMENT_IDS = {
    seed: f"exp05_cross_model_gradcam_seed{seed}" for seed in SEEDS
}

ALIASES: dict[str, tuple[str, ...]] = {
    "experiment_id": ("experiment_id", "source_experiment_id"),
    "seed": ("seed", "random_seed"),
    "comparison": (
        "comparison",
        "model_comparison",
        "comparison_name",
        "pair",
    ),
    "status": (
        "comparison_status",
        "status",
        "similarity_status",
    ),
    "pearson": (
        "pearson_correlation",
        "pearson",
        "pearson_similarity",
    ),
    "cosine": (
        "cosine_similarity",
        "cosine",
    ),
    "dice": (
        "dice_score",
        "dice",
        "dice_similarity",
    ),
    "iou": (
        "intersection_over_union",
        "iou",
        "jaccard_index",
    ),
    "mae": (
        "mean_absolute_error",
        "mae",
    ),
    "mse": (
        "mean_squared_error",
        "mse",
    ),
    "confidence_difference": (
        "absolute_confidence_difference",
        "mean_absolute_confidence_difference",
        "confidence_difference_absolute",
        "abs_confidence_difference",
        "confidence_difference",
    ),
    "degenerate": (
        "is_degenerate_pair",
        "degenerate_pair",
        "is_degenerate",
    ),
}

EXCLUDED_FILENAME_PARTS = {
    "summary",
    "quantitative_summary",
    "qualitative",
    "candidate",
    "selection",
    "manifest",
    "snapshot",
    "source",
    "artifact",
    "aligned",
    "sanity",
    "metadata",
}


@dataclass(frozen=True)
class SimilaritySource:
    seed: int
    experiment_id: str
    path: Path
    dataframe: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create thesis-ready three-seed Experiment 5 cross-model "
            "Grad-CAM tables and figures."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root of the thesis-fl-xai repository.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Override metrics directory. Default: <project-root>/outputs/metrics",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Override figures directory. Default: <project-root>/outputs/figures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory. Default: "
            "<project-root>/outputs/thesis_results/cross_model_xai"
        ),
    )
    return parser.parse_args()


def read_csv_safely(path: Path, nrows: int | None = None) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        return None


def first_existing_column(
    df: pd.DataFrame,
    aliases: Sequence[str],
    logical_name: str,
    required: bool = True,
) -> str | None:
    for column in aliases:
        if column in df.columns:
            return column
    if required:
        raise KeyError(
            f"Could not resolve {logical_name!r}. Expected one of "
            f"{list(aliases)}. Found: {list(df.columns)}"
        )
    return None


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def normalize_comparison_name(value: object) -> str:
    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("__", "_")

    aliases = {
        "centralized_vs_iid": "centralized_vs_iid",
        "centralised_vs_iid": "centralized_vs_iid",
        "centralized_iid": "centralized_vs_iid",
        "centralized_vs_non_iid": "centralized_vs_non_iid_label_skew",
        "centralized_vs_noniid": "centralized_vs_non_iid_label_skew",
        "centralized_vs_non_iid_label_skew": (
            "centralized_vs_non_iid_label_skew"
        ),
        "centralised_vs_non_iid_label_skew": (
            "centralized_vs_non_iid_label_skew"
        ),
        "iid_vs_non_iid": "iid_vs_non_iid_label_skew",
        "iid_vs_noniid": "iid_vs_non_iid_label_skew",
        "iid_vs_non_iid_label_skew": "iid_vs_non_iid_label_skew",
    }

    if normalized not in aliases:
        raise ValueError(f"Unsupported Experiment 5 comparison name: {value!r}")
    return aliases[normalized]


def discover_similarity_file(
    metrics_dir: Path,
    seed: int,
    experiment_id: str,
) -> Path:
    candidates: list[tuple[int, Path]] = []

    for path in sorted(metrics_dir.rglob("*.csv")):
        lower_name = path.name.lower()

        if "similarity" not in lower_name:
            continue
        if any(part in lower_name for part in EXCLUDED_FILENAME_PARTS):
            continue

        header = read_csv_safely(path, nrows=5)
        if header is None or header.empty:
            continue

        try:
            first_existing_column(
                header,
                ALIASES["comparison"],
                "comparison",
            )
            first_existing_column(
                header,
                ALIASES["status"],
                "comparison status",
            )
            first_existing_column(
                header,
                ALIASES["pearson"],
                "Pearson similarity",
            )
            first_existing_column(
                header,
                ALIASES["dice"],
                "Dice similarity",
            )
        except KeyError:
            continue

        score = 0
        if experiment_id.lower() in lower_name:
            score += 14
        if f"seed{seed}" in lower_name:
            score += 7
        if "cross_model" in lower_name:
            score += 5
        if "gradcam" in lower_name:
            score += 3

        experiment_column = first_existing_column(
            header,
            ALIASES["experiment_id"],
            "experiment_id",
            required=False,
        )
        if experiment_column is not None:
            ids = header[experiment_column].dropna().astype(str)
            if (ids == experiment_id).any():
                score += 16
            elif experiment_id.lower() in " ".join(ids.tolist()).lower():
                score += 6
            elif experiment_id.lower() not in lower_name:
                continue
        elif experiment_id.lower() not in lower_name and f"seed{seed}" not in lower_name:
            continue

        full = read_csv_safely(path)
        if full is None:
            continue

        if len(full) == 600:
            score += 10
        else:
            score -= abs(len(full) - 600)

        candidates.append((score, path))

    if not candidates:
        raise FileNotFoundError(
            f"No official Experiment 5 similarity CSV found for seed {seed} "
            f"under {metrics_dir}."
        )

    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    top_score = candidates[0][0]
    top = [path for score, path in candidates if score == top_score]

    if len(top) != 1:
        rendered = "\n".join(f"  - {path}" for path in top)
        raise RuntimeError(
            f"Multiple equally likely Experiment 5 similarity files found for "
            f"seed {seed}:\n{rendered}"
        )

    return top[0]


def normalize_similarity_source(
    seed: int,
    experiment_id: str,
    path: Path,
) -> SimilaritySource:
    raw = pd.read_csv(path, low_memory=False)

    if len(raw) != 600:
        raise ValueError(
            f"{experiment_id}: expected 600 similarity rows, found {len(raw)}."
        )

    resolved = {
        name: first_existing_column(
            raw,
            aliases,
            name,
            required=name
            not in {
                "experiment_id",
                "seed",
                "confidence_difference",
                "degenerate",
            },
        )
        for name, aliases in ALIASES.items()
    }

    normalized = pd.DataFrame(index=raw.index)
    normalized["seed"] = seed
    normalized["experiment_id"] = experiment_id
    normalized["source_csv"] = str(path)
    normalized["comparison"] = raw[resolved["comparison"]].map(
        normalize_comparison_name
    )
    normalized["comparison_status"] = (
        raw[resolved["status"]].astype(str).str.strip().str.lower()
    )

    for metric in ("pearson", "cosine", "dice", "iou", "mae", "mse"):
        normalized[metric] = pd.to_numeric(
            raw[resolved[metric]],
            errors="coerce",
        )

    if resolved["confidence_difference"] is not None:
        confidence_values = pd.to_numeric(
            raw[resolved["confidence_difference"]],
            errors="coerce",
        )
        normalized["confidence_difference"] = confidence_values.abs()
    else:
        normalized["confidence_difference"] = np.nan

    if resolved["degenerate"] is not None:
        normalized["is_degenerate_pair"] = bool_series(
            raw[resolved["degenerate"]]
        )
    else:
        normalized["is_degenerate_pair"] = False

    if resolved["experiment_id"] is not None:
        unique_ids = set(
            raw[resolved["experiment_id"]]
            .dropna()
            .astype(str)
            .unique()
        )
        if unique_ids != {experiment_id}:
            raise ValueError(
                f"{experiment_id}: unexpected experiment_id values "
                f"{sorted(unique_ids)}."
            )

    if resolved["seed"] is not None:
        unique_seeds = set(
            pd.to_numeric(raw[resolved["seed"]], errors="raise")
            .astype(int)
            .unique()
        )
        if unique_seeds != {seed}:
            raise ValueError(
                f"{experiment_id}: unexpected seed values "
                f"{sorted(unique_seeds)}."
            )

    comparison_counts = normalized["comparison"].value_counts()
    expected_counts = {comparison: 200 for comparison in COMPARISON_ORDER}
    if comparison_counts.to_dict() != expected_counts:
        raise ValueError(
            f"{experiment_id}: expected 200 rows per comparison. Found "
            f"{comparison_counts.to_dict()}."
        )

    allowed_statuses = {"computed", "skipped_target_changed"}
    unexpected_statuses = set(
        normalized["comparison_status"].dropna().unique()
    ) - allowed_statuses
    if unexpected_statuses:
        raise ValueError(
            f"{experiment_id}: unexpected comparison statuses "
            f"{sorted(unexpected_statuses)}."
        )

    if normalized["is_degenerate_pair"].any():
        count = int(normalized["is_degenerate_pair"].sum())
        raise ValueError(
            f"{experiment_id}: found {count} degenerate heatmap pairs."
        )

    computed = normalized["comparison_status"].eq("computed")
    for metric in ("pearson", "cosine", "dice", "iou", "mae", "mse"):
        values = normalized.loc[computed, metric]
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(
                f"{experiment_id}: {metric} contains NaN/inf for computed pairs."
            )

    if resolved["confidence_difference"] is not None:
        confidence_values = normalized.loc[
            computed, "confidence_difference"
        ]
        if confidence_values.isna().any() or not np.isfinite(
            confidence_values
        ).all():
            raise ValueError(
                f"{experiment_id}: confidence difference contains NaN/inf "
                f"for computed pairs."
            )

    return SimilaritySource(
        seed=seed,
        experiment_id=experiment_id,
        path=path,
        dataframe=normalized,
    )


def load_sources(metrics_dir: Path) -> dict[int, SimilaritySource]:
    sources: dict[int, SimilaritySource] = {}

    for seed in SEEDS:
        experiment_id = EXPERIMENT_IDS[seed]
        path = discover_similarity_file(
            metrics_dir=metrics_dir,
            seed=seed,
            experiment_id=experiment_id,
        )
        sources[seed] = normalize_similarity_source(
            seed=seed,
            experiment_id=experiment_id,
            path=path,
        )

    return sources


def build_seed_comparison_summary(
    sources: dict[int, SimilaritySource],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for seed in SEEDS:
        source = sources[seed]
        for comparison in COMPARISON_ORDER:
            group = source.dataframe[
                source.dataframe["comparison"] == comparison
            ]
            computed = group["comparison_status"].eq("computed")
            skipped = group["comparison_status"].eq(
                "skipped_target_changed"
            )
            computed_group = group.loc[computed]

            row: dict[str, object] = {
                "seed": seed,
                "comparison": comparison,
                "comparison_label": COMPARISON_LABELS[comparison],
                "total_pairs": len(group),
                "computed_pairs": int(computed.sum()),
                "skipped_target_changed_pairs": int(skipped.sum()),
                "prediction_agreement_rate": float(computed.mean()),
                "prediction_agreement_percent": float(computed.mean() * 100.0),
                "experiment_id": source.experiment_id,
                "source_csv": str(source.path),
            }

            for metric in (
                "pearson",
                "cosine",
                "dice",
                "iou",
                "mae",
                "mse",
            ):
                row[f"{metric}_mean"] = float(
                    computed_group[metric].mean()
                )
                row[f"{metric}_standard_deviation_within_seed"] = float(
                    computed_group[metric].std(ddof=1)
                )

            if computed_group["confidence_difference"].notna().any():
                row["confidence_difference_mean"] = float(
                    computed_group["confidence_difference"].mean()
                )
                row["confidence_difference_percentage_points"] = float(
                    computed_group["confidence_difference"].mean() * 100.0
                )
            else:
                row["confidence_difference_mean"] = np.nan
                row["confidence_difference_percentage_points"] = np.nan

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary["_seed_order"] = pd.Categorical(
        summary["seed"],
        categories=list(SEEDS),
        ordered=True,
    )
    summary["_comparison_order"] = pd.Categorical(
        summary["comparison"],
        categories=list(COMPARISON_ORDER),
        ordered=True,
    )
    summary = (
        summary.sort_values(["_comparison_order", "_seed_order"])
        .drop(columns=["_comparison_order", "_seed_order"])
        .reset_index(drop=True)
    )
    return summary


def build_three_seed_summary(
    seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    value_columns = [
        "prediction_agreement_percent",
        "computed_pairs",
        "skipped_target_changed_pairs",
        "pearson_mean",
        "cosine_mean",
        "dice_mean",
        "iou_mean",
        "mae_mean",
        "mse_mean",
        "confidence_difference_percentage_points",
    ]

    rows: list[dict[str, object]] = []

    for comparison in COMPARISON_ORDER:
        group = seed_summary[
            seed_summary["comparison"] == comparison
        ]
        for value_column in value_columns:
            values = pd.to_numeric(
                group[value_column],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue

            rows.append(
                {
                    "comparison": comparison,
                    "comparison_label": COMPARISON_LABELS[comparison],
                    "measure": value_column,
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "n_seeds": int(len(values)),
                }
            )

    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.6f")


def plot_comparison_metric(
    ax: plt.Axes,
    seed_summary: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(COMPARISON_ORDER), dtype=float)
    display_labels = [
        COMPARISON_LABELS[comparison] for comparison in COMPARISON_ORDER
    ]
    seed_offsets = {
        42: -0.10,
        7: 0.00,
        123: 0.10,
    }

    for seed in SEEDS:
        seed_data = (
            seed_summary[seed_summary["seed"] == seed]
            .set_index("comparison")
            .reindex(COMPARISON_ORDER)
        )
        ax.scatter(
            x + seed_offsets[seed],
            seed_data[value_column].to_numpy(dtype=float),
            marker="o",
            s=42,
            alpha=0.78,
            label=f"Seed {seed}",
            zorder=3,
        )

    grouped = (
        seed_summary.groupby("comparison", observed=True)[value_column]
        .agg(["mean", "std"])
        .reindex(COMPARISON_ORDER)
    )
    ax.errorbar(
        x,
        grouped["mean"].to_numpy(dtype=float),
        yerr=grouped["std"].to_numpy(dtype=float),
        fmt="D",
        color="tab:red",
        linewidth=2.0,
        capsize=5,
        markersize=6,
        label="Mean ± SD",
        zorder=4,
    )
    ax.set_xticks(x, display_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def save_two_panel_comparison_figure(
    seed_summary: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    ylabel_a: str,
    ylabel_b: str,
    title_a: str,
    title_b: str,
    figure_title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))

    plot_comparison_metric(
        axes[0],
        seed_summary,
        metric_a,
        ylabel_a,
        title_a,
    )
    plot_comparison_metric(
        axes[1],
        seed_summary,
        metric_b,
        ylabel_b,
        title_b,
    )

    fig.suptitle(figure_title, y=0.98)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, bottom=0.25, wspace=0.30)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def discover_optional_qualitative_artifacts(
    metrics_dir: Path,
    figures_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Inventory and copy existing qualitative Experiment 5 artifacts.

    This step is optional. The script does not regenerate qualitative Grad-CAM
    panels because that would require image/heatmap inputs beyond the saved
    similarity CSVs.
    """
    patterns = (
        "exp05_cross_model_gradcam_seed*_qualitative_candidates.csv",
        "exp05_cross_model_gradcam_seed*_qualitative_figure_selection.csv",
    )

    rows: list[dict[str, object]] = []
    qualitative_dir = output_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    for pattern in patterns:
        for source_path in sorted(metrics_dir.rglob(pattern)):
            destination = qualitative_dir / source_path.name
            shutil.copy2(source_path, destination)
            rows.append(
                {
                    "artifact_type": "qualitative_source_table",
                    "source_path": str(source_path),
                    "copied_path": str(destination),
                    "filename": destination.name,
                }
            )

    for source_path in sorted(
        figures_dir.rglob(
            "exp05_cross_model_gradcam_seed*_qualitative_comparison.png"
        )
    ):
        destination = qualitative_dir / source_path.name
        shutil.copy2(source_path, destination)
        rows.append(
            {
                "artifact_type": "qualitative_figure",
                "source_path": str(source_path),
                "copied_path": str(destination),
                "filename": destination.name,
            }
        )

    return pd.DataFrame(rows)


def write_artifact_catalog(path: Path) -> None:
    text = """# Cross-model XAI artifact catalog

## TABLE_07

**Title:** Cross-model Grad-CAM similarity by seed and model pair  
**Filename:** `table07_cross_model_similarity_by_seed.csv`  
**Purpose:** Reports prediction agreement, comparison-status counts, spatial-similarity metrics, error metrics, and confidence differences for each seed and model pair.

## TABLE_08

**Title:** Three-seed cross-model Grad-CAM summary  
**Filename:** `table08_cross_model_similarity_three_seed_summary.csv`  
**Purpose:** Reports mean, sample standard deviation, minimum, and maximum across seeds for every cross-model measure.

## FIGURE_09

**Structured name:** Cross-model Grad-CAM spatial similarity  
**Files:** `fig09_cross_model_gradcam_spatial_similarity.png` and `.pdf`  
**Panels:** Pearson correlation; Dice overlap.  
**Primary conclusion:** IID and non-IID federated explanations were consistently more similar to each other than either federated explanation was to the centralized explanation.

## FIGURE_10

**Structured name:** Cross-model prediction agreement and confidence difference  
**Files:** `fig10_cross_model_prediction_agreement_confidence.png` and `.pdf`  
**Panels:** Prediction agreement rate; mean absolute confidence difference.  
**Primary conclusion:** All model pairs usually predicted the same target, but the two federated models also had the smallest confidence difference.

## QUALITATIVE INVENTORY

**Filename:** `table09_qualitative_artifact_inventory.csv`  
**Purpose:** Records any existing Experiment 5 qualitative candidate tables, figure-selection tables, and qualitative comparison figures that were copied into the thesis-results package. The analysis module does not regenerate heatmaps.
"""
    path.write_text(text, encoding="utf-8")


def write_manifest(path: Path, output_dir: Path) -> None:
    entries = [
        (
            "TABLE_07",
            "table",
            "table07_cross_model_similarity_by_seed.csv",
        ),
        (
            "TABLE_08",
            "table",
            "table08_cross_model_similarity_three_seed_summary.csv",
        ),
        (
            "TABLE_09",
            "table",
            "table09_qualitative_artifact_inventory.csv",
        ),
        (
            "FIGURE_09",
            "figure_png",
            "fig09_cross_model_gradcam_spatial_similarity.png",
        ),
        (
            "FIGURE_09",
            "figure_pdf",
            "fig09_cross_model_gradcam_spatial_similarity.pdf",
        ),
        (
            "FIGURE_09_SOURCE",
            "source_table",
            "fig09_source_cross_model_gradcam_spatial_similarity.csv",
        ),
        (
            "FIGURE_10",
            "figure_png",
            "fig10_cross_model_prediction_agreement_confidence.png",
        ),
        (
            "FIGURE_10",
            "figure_pdf",
            "fig10_cross_model_prediction_agreement_confidence.pdf",
        ),
        (
            "FIGURE_10_SOURCE",
            "source_table",
            "fig10_source_prediction_agreement_confidence.csv",
        ),
        (
            "CATALOG_03",
            "catalog",
            "cross_model_xai_artifact_catalog.md",
        ),
    ]

    rows = []
    for artifact_id, artifact_type, filename in entries:
        artifact_path = output_dir / filename
        rows.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "filename": filename,
                "exists": artifact_path.exists(),
                "size_bytes": (
                    artifact_path.stat().st_size
                    if artifact_path.exists()
                    else None
                ),
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    metrics_dir = (
        args.metrics_dir.expanduser().resolve()
        if args.metrics_dir
        else project_root / "outputs" / "metrics"
    )
    figures_dir = (
        args.figures_dir.expanduser().resolve()
        if args.figures_dir
        else project_root / "outputs" / "figures"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root
        / "outputs"
        / "thesis_results"
        / "cross_model_xai"
    )

    if not metrics_dir.exists():
        raise FileNotFoundError(
            f"Metrics directory does not exist: {metrics_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(metrics_dir)
    seed_summary = build_seed_comparison_summary(sources)
    three_seed_summary = build_three_seed_summary(seed_summary)

    save_table(
        seed_summary,
        output_dir / "table07_cross_model_similarity_by_seed.csv",
    )
    save_table(
        three_seed_summary,
        output_dir
        / "table08_cross_model_similarity_three_seed_summary.csv",
    )

    save_table(
        seed_summary[
            [
                "seed",
                "comparison",
                "comparison_label",
                "pearson_mean",
                "dice_mean",
                "computed_pairs",
                "skipped_target_changed_pairs",
                "experiment_id",
                "source_csv",
            ]
        ],
        output_dir
        / "fig09_source_cross_model_gradcam_spatial_similarity.csv",
    )

    save_table(
        seed_summary[
            [
                "seed",
                "comparison",
                "comparison_label",
                "prediction_agreement_percent",
                "confidence_difference_percentage_points",
                "computed_pairs",
                "skipped_target_changed_pairs",
                "experiment_id",
                "source_csv",
            ]
        ],
        output_dir
        / "fig10_source_prediction_agreement_confidence.csv",
    )

    save_two_panel_comparison_figure(
        seed_summary=seed_summary,
        metric_a="pearson_mean",
        metric_b="dice_mean",
        ylabel_a="Mean Pearson correlation",
        ylabel_b="Mean Dice score",
        title_a="(a) Grad-CAM Pearson correlation",
        title_b="(b) Grad-CAM Dice overlap",
        figure_title="Cross-model Grad-CAM spatial similarity",
        png_path=output_dir
        / "fig09_cross_model_gradcam_spatial_similarity.png",
        pdf_path=output_dir
        / "fig09_cross_model_gradcam_spatial_similarity.pdf",
    )

    if seed_summary[
        "confidence_difference_percentage_points"
    ].isna().all():
        raise ValueError(
            "No confidence-difference column was found in the official "
            "Experiment 5 similarity CSVs. Add the exact column name to "
            "ALIASES['confidence_difference'] before generating Figure 10."
        )

    save_two_panel_comparison_figure(
        seed_summary=seed_summary,
        metric_a="prediction_agreement_percent",
        metric_b="confidence_difference_percentage_points",
        ylabel_a="Prediction agreement (%)",
        ylabel_b="Mean absolute confidence difference (percentage points)",
        title_a="(a) Prediction agreement",
        title_b="(b) Absolute confidence difference",
        figure_title=(
            "Cross-model prediction agreement and confidence difference"
        ),
        png_path=output_dir
        / "fig10_cross_model_prediction_agreement_confidence.png",
        pdf_path=output_dir
        / "fig10_cross_model_prediction_agreement_confidence.pdf",
    )

    qualitative_inventory = discover_optional_qualitative_artifacts(
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        output_dir=output_dir,
    )
    if qualitative_inventory.empty:
        qualitative_inventory = pd.DataFrame(
            columns=[
                "artifact_type",
                "source_path",
                "copied_path",
                "filename",
            ]
        )
    save_table(
        qualitative_inventory,
        output_dir / "table09_qualitative_artifact_inventory.csv",
    )

    write_artifact_catalog(
        output_dir / "cross_model_xai_artifact_catalog.md"
    )
    write_manifest(
        output_dir / "cross_model_xai_artifact_manifest.csv",
        output_dir,
    )

    print("Cross-model XAI results package created:")
    print(output_dir)

    print("\nDiscovered official Experiment 5 similarity files:")
    for seed in SEEDS:
        source = sources[seed]
        print(f"- {source.experiment_id}: {source.path}")

    print("\nGenerated artifacts:")
    for path in sorted(output_dir.iterdir()):
        print(f"- {path.name}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise