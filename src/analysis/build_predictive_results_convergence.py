#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS: tuple[int, ...] = (42, 7, 123)
CONDITION_ORDER: tuple[str, ...] = (
    "Centralized selected",
    "IID Round 50",
    "Non-IID Round 50",
)
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1 score",
    "auc": "ROC AUC",
    "balanced_accuracy": "Balanced accuracy",
    "loss": "Validation loss",
}

EXPERIMENT_IDS = {
    "centralized": "exp01_centralized_vindr_binary_seed{seed}",
    "iid": "exp02_fedavg_iid_5clients_seed{seed}",
    "non_iid": "exp03_fedavg_noniid_5clients_seed{seed}",
}

ALIASES = {
    "accuracy": ("val_accuracy", "accuracy"),
    "f1": ("val_f1", "f1"),
    "auc": ("val_auc", "auc"),
    "balanced_accuracy": ("val_balanced_accuracy", "balanced_accuracy"),
    "loss": ("val_loss", "centralized_loss", "loss"),
}

EXCLUDED_FILENAME_PARTS = {
    "summary",
    "prediction",
    "metadata",
    "similarity",
    "qualitative",
    "manifest",
    "snapshot",
    "aligned",
    "source",
    "artifact",
    "sanity",
    "timing",
    "debug_runpod",
}


@dataclass(frozen=True)
class ExperimentSource:
    condition: str
    seed: int
    experiment_id: str
    path: Path
    dataframe: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create thesis-ready predictive-performance tables, summaries, "
            "and figures for seeds 42, 7, and 123."
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
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory. "
            "Default: <project-root>/outputs/thesis_results/predictive"
        ),
    )
    return parser.parse_args()


def read_csv_safely(path: Path, nrows: int | None = None) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        return None


def discover_metrics_file(
    metrics_dir: Path,
    experiment_id: str,
    expected_progress_column: str,
    expected_rows: int,
) -> Path:
    candidates: list[tuple[int, Path]] = []

    for path in sorted(metrics_dir.rglob("*.csv")):
        lower_name = path.name.lower()
        if any(part in lower_name for part in EXCLUDED_FILENAME_PARTS):
            continue

        header_df = read_csv_safely(path, nrows=5)
        if header_df is None or header_df.empty:
            continue
        if expected_progress_column not in header_df.columns:
            continue

        score = 0
        if experiment_id.lower() in lower_name:
            score += 7

        if "experiment_id" in header_df.columns:
            ids = header_df["experiment_id"].dropna().astype(str)
            if (ids == experiment_id).any():
                score += 12
            elif experiment_id.lower() in " ".join(ids.tolist()).lower():
                score += 4
            else:
                continue
        elif experiment_id.lower() not in lower_name:
            continue

        score += 4
        if "metrics" in lower_name:
            score += 2
        if "flower_fedavg" in lower_name and expected_progress_column == "round":
            score += 2
        if "centralized" in lower_name and expected_progress_column == "epoch":
            score += 2

        full_df = read_csv_safely(path)
        if full_df is None:
            continue
        if len(full_df) == expected_rows:
            score += 5
        else:
            score -= abs(len(full_df) - expected_rows)

        candidates.append((score, path))

    if not candidates:
        raise FileNotFoundError(
            f"No metrics CSV found for experiment_id={experiment_id!r} "
            f"under {metrics_dir}."
        )

    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    top_score = candidates[0][0]
    top = [path for score, path in candidates if score == top_score]

    if len(top) > 1:
        rendered = "\n".join(f"  - {p}" for p in top)
        raise RuntimeError(
            f"Multiple equally likely metrics files found for {experiment_id}:\n"
            f"{rendered}\nArchive ambiguous duplicates or pass a clean metrics directory."
        )

    return top[0]


def first_existing_column(
    df: pd.DataFrame, aliases: Sequence[str], metric: str
) -> str:
    for name in aliases:
        if name in df.columns:
            return name
    raise KeyError(
        f"Could not resolve metric {metric!r}. Expected one of {list(aliases)}. "
        f"Found columns: {list(df.columns)}"
    )


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def validate_experiment_dataframe(
    df: pd.DataFrame,
    experiment_id: str,
    seed: int,
    progress_column: str,
    expected_rows: int,
    expected_progress: Iterable[int],
) -> None:
    if len(df) != expected_rows:
        raise ValueError(
            f"{experiment_id}: expected {expected_rows} metric rows, found {len(df)}."
        )

    if progress_column not in df.columns:
        raise KeyError(f"{experiment_id}: missing {progress_column!r} column.")

    progress = pd.to_numeric(df[progress_column], errors="raise").astype(int).tolist()
    expected = list(expected_progress)
    if progress != expected:
        raise ValueError(
            f"{experiment_id}: invalid {progress_column} sequence.\n"
            f"Expected: {expected}\nFound: {progress}"
        )

    if "experiment_id" in df.columns:
        unique_ids = set(df["experiment_id"].dropna().astype(str).unique())
        if unique_ids != {experiment_id}:
            raise ValueError(
                f"{experiment_id}: unexpected experiment_id values: {sorted(unique_ids)}"
            )

    if "seed" in df.columns:
        unique_seeds = set(
            pd.to_numeric(df["seed"], errors="raise").astype(int).unique()
        )
        if unique_seeds != {seed}:
            raise ValueError(
                f"{experiment_id}: unexpected seed values: {sorted(unique_seeds)}"
            )

    for column in ("val_num_samples", "num_validation_samples"):
        if column in df.columns:
            values = set(
                pd.to_numeric(df[column], errors="raise").astype(int).unique()
            )
            if values != {3000}:
                raise ValueError(
                    f"{experiment_id}: {column} must equal 3000 in every row; "
                    f"found {sorted(values)}."
                )
            break

    for metric, aliases in ALIASES.items():
        column = first_existing_column(df, aliases, metric)
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(
                f"{experiment_id}: metric column {column!r} contains NaN/inf."
            )


def load_sources(metrics_dir: Path) -> dict[tuple[str, int], ExperimentSource]:
    sources: dict[tuple[str, int], ExperimentSource] = {}

    for seed in SEEDS:
        for condition in ("centralized", "iid", "non_iid"):
            experiment_id = EXPERIMENT_IDS[condition].format(seed=seed)
            progress_column = "epoch" if condition == "centralized" else "round"
            expected_rows = 10 if condition == "centralized" else 50
            expected_progress = (
                range(1, 11) if condition == "centralized" else range(1, 51)
            )

            path = discover_metrics_file(
                metrics_dir=metrics_dir,
                experiment_id=experiment_id,
                expected_progress_column=progress_column,
                expected_rows=expected_rows,
            )
            df = pd.read_csv(path, low_memory=False)
            validate_experiment_dataframe(
                df=df,
                experiment_id=experiment_id,
                seed=seed,
                progress_column=progress_column,
                expected_rows=expected_rows,
                expected_progress=expected_progress,
            )
            sources[(condition, seed)] = ExperimentSource(
                condition=condition,
                seed=seed,
                experiment_id=experiment_id,
                path=path,
                dataframe=df,
            )

    return sources


def normalized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for metric, aliases in ALIASES.items():
        column = first_existing_column(df, aliases, metric)
        result[metric] = pd.to_numeric(df[column], errors="raise").astype(float)
    return result


def selected_centralized_row(source: ExperimentSource) -> dict[str, object]:
    df = source.dataframe
    metrics = normalized_metrics(df)

    best_flags = (
        bool_series(df["is_best_checkpoint"])
        if "is_best_checkpoint" in df.columns
        else None
    )
    if best_flags is not None and int(best_flags.sum()) == 1:
        idx = best_flags.idxmax()
    else:
        idx = metrics["f1"].idxmax()

    row = metrics.loc[idx]
    return {
        "seed": source.seed,
        "condition": "Centralized",
        "selection_rule": "Selected best validation F1",
        "selected_epoch_or_round": int(df.loc[idx, "epoch"]),
        "accuracy": row["accuracy"],
        "f1": row["f1"],
        "auc": row["auc"],
        "balanced_accuracy": row["balanced_accuracy"],
        "loss": row["loss"],
        "experiment_id": source.experiment_id,
        "source_csv": str(source.path),
    }


def selected_federated_rows(
    source: ExperimentSource, label: str
) -> list[dict[str, object]]:
    df = source.dataframe
    metrics = normalized_metrics(df)

    best_idx = metrics["f1"].idxmax()
    final_idx = pd.to_numeric(df["round"], errors="raise").idxmax()

    rows: list[dict[str, object]] = []
    for selection_rule, idx in (
        ("Best validation F1", best_idx),
        ("Final Round 50", final_idx),
    ):
        row = metrics.loc[idx]
        rows.append(
            {
                "seed": source.seed,
                "condition": label,
                "selection_rule": selection_rule,
                "selected_epoch_or_round": int(df.loc[idx, "round"]),
                "accuracy": row["accuracy"],
                "f1": row["f1"],
                "auc": row["auc"],
                "balanced_accuracy": row["balanced_accuracy"],
                "loss": row["loss"],
                "experiment_id": source.experiment_id,
                "source_csv": str(source.path),
            }
        )
    return rows


def set_seed_order(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["_seed_order"] = pd.Categorical(
        result["seed"], categories=list(SEEDS), ordered=True
    )
    result = result.sort_values("_seed_order").drop(columns="_seed_order")
    return result.reset_index(drop=True)


def build_tables(
    sources: dict[tuple[str, int], ExperimentSource]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    centralized = pd.DataFrame(
        [selected_centralized_row(sources[("centralized", seed)]) for seed in SEEDS]
    )
    centralized = set_seed_order(centralized)

    federated_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        federated_rows.extend(selected_federated_rows(sources[("iid", seed)], "IID"))
        federated_rows.extend(
            selected_federated_rows(sources[("non_iid", seed)], "Non-IID")
        )
    federated = pd.DataFrame(federated_rows)
    federated["_seed_order"] = pd.Categorical(
        federated["seed"], categories=list(SEEDS), ordered=True
    )
    federated["_condition_order"] = pd.Categorical(
        federated["condition"], categories=["IID", "Non-IID"], ordered=True
    )
    federated["_selection_order"] = pd.Categorical(
        federated["selection_rule"],
        categories=["Best validation F1", "Final Round 50"],
        ordered=True,
    )
    federated = (
        federated.sort_values(
            ["_condition_order", "_selection_order", "_seed_order"]
        )
        .drop(columns=["_seed_order", "_condition_order", "_selection_order"])
        .reset_index(drop=True)
    )

    comparison_rows = []
    for _, row in centralized.iterrows():
        comparison_rows.append(
            {**row.to_dict(), "display_condition": "Centralized selected"}
        )

    for condition, display in (
        ("IID", "IID Round 50"),
        ("Non-IID", "Non-IID Round 50"),
    ):
        subset = federated[
            (federated["condition"] == condition)
            & (federated["selection_rule"] == "Final Round 50")
        ]
        for _, row in subset.iterrows():
            comparison_rows.append({**row.to_dict(), "display_condition": display})

    comparison = pd.DataFrame(comparison_rows)
    comparison["_condition_order"] = pd.Categorical(
        comparison["display_condition"],
        categories=list(CONDITION_ORDER),
        ordered=True,
    )
    comparison["_seed_order"] = pd.Categorical(
        comparison["seed"], categories=list(SEEDS), ordered=True
    )
    comparison = (
        comparison.sort_values(["_condition_order", "_seed_order"])
        .drop(columns=["_condition_order", "_seed_order"])
        .reset_index(drop=True)
    )

    aggregate_rows: list[dict[str, object]] = []
    for condition in CONDITION_ORDER:
        group = comparison[comparison["display_condition"] == condition]
        for metric in ("accuracy", "f1", "auc", "balanced_accuracy", "loss"):
            values = pd.to_numeric(group[metric], errors="raise")
            aggregate_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "n_seeds": len(values),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)

    return centralized, federated, comparison, aggregate



def build_federated_roundwise_source(
    sources: dict[tuple[str, int], ExperimentSource]
) -> pd.DataFrame:
    """Create one validated long-form table for all federated rounds."""
    rows: list[dict[str, object]] = []

    for condition_key, condition_label in (
        ("iid", "IID"),
        ("non_iid", "Non-IID"),
    ):
        for seed in SEEDS:
            source = sources[(condition_key, seed)]
            metrics = normalized_metrics(source.dataframe)
            rounds = pd.to_numeric(
                source.dataframe["round"], errors="raise"
            ).astype(int)

            for idx in source.dataframe.index:
                rows.append(
                    {
                        "condition": condition_label,
                        "seed": seed,
                        "round": int(rounds.loc[idx]),
                        "accuracy": float(metrics.loc[idx, "accuracy"]),
                        "f1": float(metrics.loc[idx, "f1"]),
                        "auc": float(metrics.loc[idx, "auc"]),
                        "balanced_accuracy": float(
                            metrics.loc[idx, "balanced_accuracy"]
                        ),
                        "loss": float(metrics.loc[idx, "loss"]),
                        "experiment_id": source.experiment_id,
                        "source_csv": str(source.path),
                    }
                )

    result = pd.DataFrame(rows)
    result["_condition_order"] = pd.Categorical(
        result["condition"], categories=["IID", "Non-IID"], ordered=True
    )
    result["_seed_order"] = pd.Categorical(
        result["seed"], categories=list(SEEDS), ordered=True
    )
    result = (
        result.sort_values(["_condition_order", "_seed_order", "round"])
        .drop(columns=["_condition_order", "_seed_order"])
        .reset_index(drop=True)
    )

    expected_rows = 2 * len(SEEDS) * 50
    if len(result) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} federated round-wise rows, found {len(result)}."
        )

    return result


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.6f")


def aggregate_lookup(
    aggregate: pd.DataFrame, condition: str, metric: str, field: str = "mean"
) -> float:
    row = aggregate[
        (aggregate["condition"] == condition) & (aggregate["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError(f"Could not uniquely resolve {condition=} {metric=}.")
    return float(row.iloc[0][field])


def plot_condition_metric(
    ax: plt.Axes,
    comparison: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    x = np.arange(len(CONDITION_ORDER), dtype=float)

    for seed in SEEDS:
        seed_df = (
            comparison[comparison["seed"] == seed]
            .set_index("display_condition")
            .reindex(CONDITION_ORDER)
        )
        ax.plot(
            x,
            seed_df[metric].to_numpy(dtype=float),
            marker="o",
            linewidth=1.2,
            alpha=0.65,
            label=f"Seed {seed}",
        )

    grouped = (
        comparison.groupby("display_condition", observed=True)[metric]
        .agg(["mean", "std"])
        .reindex(CONDITION_ORDER)
    )
    ax.errorbar(
        x,
        grouped["mean"].to_numpy(dtype=float),
        yerr=grouped["std"].to_numpy(dtype=float),
        fmt="D",
        linewidth=2.0,
        capsize=5,
        markersize=6,
        label="Mean ± SD",
    )
    ax.set_xticks(x, CONDITION_ORDER)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def save_two_panel_figure(
    comparison: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    title_a: str,
    title_b: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    plot_condition_metric(axes[0], comparison, metric_a, title_a)
    plot_condition_metric(axes[1], comparison, metric_b, title_b)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_round_and_gap(
    federated: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Plot peak-F1 timing and its improvement over the Round-50 checkpoint.

    Lines connect the IID and non-IID observations obtained with the same seed.
    The right panel expresses the F1 difference in percentage points.
    """
    best = federated[federated["selection_rule"] == "Best validation F1"].copy()
    final = federated[federated["selection_rule"] == "Final Round 50"].copy()

    best_round = best.pivot(
        index="seed", columns="condition", values="selected_epoch_or_round"
    )
    best_f1 = best.pivot(index="seed", columns="condition", values="f1")
    final_f1 = final.pivot(index="seed", columns="condition", values="f1")

    # Convert the raw F1 difference into percentage points for readability.
    gap_percentage_points = (best_f1 - final_f1) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    x = np.arange(2, dtype=float)
    conditions = ["IID", "Non-IID"]

    # Panel (a): communication round with the highest validation F1.
    for seed in SEEDS:
        axes[0].plot(
            x,
            best_round.loc[seed, conditions].to_numpy(dtype=float),
            marker="o",
            linewidth=1.2,
            alpha=0.7,
            label=f"Seed {seed}",
        )

    best_round_mean = best_round[conditions].mean(axis=0)
    best_round_std = best_round[conditions].std(axis=0, ddof=1)
    axes[0].errorbar(
        x,
        best_round_mean.to_numpy(dtype=float),
        yerr=best_round_std.to_numpy(dtype=float),
        fmt="D",
        linewidth=2.0,
        capsize=5,
        markersize=6,
        color="tab:red",
        label="Mean ± SD",
    )
    axes[0].axhline(
        50,
        linestyle="--",
        linewidth=1.0,
        color="0.45",
    )
    axes[0].set_xticks(x, conditions)
    axes[0].set_ylabel("Communication round")
    axes[0].set_title("(a) Communication round of peak validation F1")
    axes[0].set_ylim(0, 52)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel (b): peak-F1 improvement over the final Round-50 checkpoint.
    for seed in SEEDS:
        axes[1].plot(
            x,
            gap_percentage_points.loc[seed, conditions].to_numpy(dtype=float),
            marker="o",
            linewidth=1.2,
            alpha=0.7,
        )

    gap_mean = gap_percentage_points[conditions].mean(axis=0)
    gap_std = gap_percentage_points[conditions].std(axis=0, ddof=1)
    axes[1].errorbar(
        x,
        gap_mean.to_numpy(dtype=float),
        yerr=gap_std.to_numpy(dtype=float),
        fmt="D",
        linewidth=2.0,
        capsize=5,
        markersize=6,
        color="tab:red",
        label="Mean ± SD",
    )
    axes[1].axhline(
        0,
        linestyle="--",
        linewidth=1.0,
        color="0.45",
    )
    axes[1].set_xticks(x, conditions)
    axes[1].set_ylabel("Best F1 − Round-50 F1 (percentage points)")
    axes[1].set_title("(b) Improvement of peak F1 over Round 50")
    axes[1].grid(axis="y", alpha=0.25)

    maximum_gap = float(np.nanmax(gap_percentage_points[conditions].to_numpy()))
    axes[1].set_ylim(-0.05, max(2.0, maximum_gap + 0.30))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.34)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_federated_convergence(
    roundwise: pd.DataFrame,
    condition: str,
    figure_number: int,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Plot F1 and ROC AUC trajectories for one federated condition."""
    subset = roundwise[roundwise["condition"] == condition].copy()
    if subset.empty:
        raise ValueError(f"No round-wise rows found for condition {condition!r}.")

    expected_rows = len(SEEDS) * 50
    if len(subset) != expected_rows:
        raise ValueError(
            f"{condition}: expected {expected_rows} rows, found {len(subset)}."
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    metrics = (
        ("f1", "Validation F1"),
        ("auc", "Validation ROC AUC"),
    )

    shared_handles: list[object] = []
    shared_labels: list[str] = []

    for panel_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[panel_index]

        for seed in SEEDS:
            seed_df = subset[subset["seed"] == seed].sort_values("round")
            line = ax.plot(
                seed_df["round"].to_numpy(dtype=int),
                seed_df[metric].to_numpy(dtype=float),
                linewidth=1.0,
                alpha=0.55,
                label=f"Seed {seed}",
            )[0]
            if panel_index == 0:
                shared_handles.append(line)
                shared_labels.append(f"Seed {seed}")

        grouped = (
            subset.groupby("round", observed=True)[metric]
            .agg(["mean", "std"])
            .sort_index()
        )
        rounds = grouped.index.to_numpy(dtype=int)
        means = grouped["mean"].to_numpy(dtype=float)
        stds = grouped["std"].to_numpy(dtype=float)

        mean_line = ax.plot(
            rounds,
            means,
            linewidth=2.4,
            color="0.15",
            label="Three-seed mean",
        )[0]
        band = ax.fill_between(
            rounds,
            means - stds,
            means + stds,
            color="0.6",
            alpha=0.20,
            label="Mean ± SD",
        )

        if panel_index == 0:
            shared_handles.extend([mean_line, band])
            shared_labels.extend(["Three-seed mean", "Mean ± SD"])

        panel_letter = "a" if panel_index == 0 else "b"
        ax.set_title(
            f"({panel_letter}) {ylabel} over communication rounds"
        )
        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 50)
        ax.set_xticks([1, 5, 10, 20, 30, 40, 50])
        ax.grid(axis="both", alpha=0.22)

    fig.legend(
        shared_handles,
        shared_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_results_summary(
    output_path: Path,
    centralized: pd.DataFrame,
    federated: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> None:
    f1_c = aggregate_lookup(aggregate, "Centralized selected", "f1")
    f1_iid = aggregate_lookup(aggregate, "IID Round 50", "f1")
    f1_non = aggregate_lookup(aggregate, "Non-IID Round 50", "f1")

    ba_c = aggregate_lookup(
        aggregate, "Centralized selected", "balanced_accuracy"
    )
    ba_iid = aggregate_lookup(aggregate, "IID Round 50", "balanced_accuracy")
    ba_non = aggregate_lookup(
        aggregate, "Non-IID Round 50", "balanced_accuracy"
    )

    acc_c = aggregate_lookup(aggregate, "Centralized selected", "accuracy")
    acc_iid = aggregate_lookup(aggregate, "IID Round 50", "accuracy")
    acc_non = aggregate_lookup(aggregate, "Non-IID Round 50", "accuracy")

    auc_c = aggregate_lookup(aggregate, "Centralized selected", "auc")
    auc_iid = aggregate_lookup(aggregate, "IID Round 50", "auc")
    auc_non = aggregate_lookup(aggregate, "Non-IID Round 50", "auc")

    best = federated[federated["selection_rule"] == "Best validation F1"]
    final = federated[federated["selection_rule"] == "Final Round 50"]
    merged = best.merge(
        final,
        on=["seed", "condition"],
        suffixes=("_best", "_final"),
        validate="one_to_one",
    )

    mean_best_round_iid = merged.loc[
        merged["condition"] == "IID", "selected_epoch_or_round_best"
    ].mean()
    mean_best_round_non = merged.loc[
        merged["condition"] == "Non-IID", "selected_epoch_or_round_best"
    ].mean()
    mean_gap_iid_percentage_points = 100.0 * (
        merged.loc[merged["condition"] == "IID", "f1_best"]
        - merged.loc[merged["condition"] == "IID", "f1_final"]
    ).mean()
    mean_gap_non_percentage_points = 100.0 * (
        merged.loc[merged["condition"] == "Non-IID", "f1_best"]
        - merged.loc[merged["condition"] == "Non-IID", "f1_final"]
    ).mean()

    selected_epochs = ", ".join(
        f"seed {int(row.seed)}: epoch {int(row.selected_epoch_or_round)}"
        for row in centralized.itertuples()
    )

    text = f"""# SUMMARY_01 — Predictive performance across three seeds

## Suggested Results sequence

Predictive performance was compared across the centralized baseline and the two federated conditions using seeds 42, 7, and 123. The centralized value for each seed corresponds to the checkpoint selected by validation F1. Federated results are reported separately for the best validation-F1 round and the final Round-50 model; these selection rules should not be combined or described as equivalent.

Figure 1 compares the selected centralized checkpoints with the final IID and non-IID federated models using F1 and balanced accuracy. Across the three seeds, mean F1 was {fmt(f1_c)} for centralized training, {fmt(f1_iid)} for IID FedAvg, and {fmt(f1_non)} for non-IID FedAvg. Mean balanced accuracy was {fmt(ba_c)}, {fmt(ba_iid)}, and {fmt(ba_non)}, respectively. The differences were small relative to the overall performance level, and no condition showed a persistent predictive collapse across the tested seeds.

Figure 2 provides complementary evidence using accuracy and ROC AUC. Mean accuracy was {fmt(acc_c)} for the selected centralized checkpoints, {fmt(acc_iid)} for IID Round 50, and {fmt(acc_non)} for non-IID Round 50. Mean ROC AUC was {fmt(auc_c)}, {fmt(auc_iid)}, and {fmt(auc_non)}, respectively. These results show that the three training conditions remained broadly comparable not only in threshold-dependent metrics but also in ranking performance.

Figure 3 examines model-selection timing within the federated experiments. The best validation-F1 round occurred at a mean round of {mean_best_round_iid:.1f} for IID training and {mean_best_round_non:.1f} for non-IID training. The mean improvement of the best checkpoint over Round 50 was {mean_gap_iid_percentage_points:.2f} percentage points for IID and {mean_gap_non_percentage_points:.2f} percentage points for non-IID. Thus, the final communication round was not consistently the strongest validation-F1 model, supporting the decision to report both best-round and final-round results.

For the centralized condition, the selected epochs were {selected_epochs}. The variation in selected epoch shows that checkpoint selection was also relevant for centralized training even though its final predictive values were stable across seeds.

## Suggested Discussion links

The first two figures support the interpretation that five-client FedAvg preserved predictive viability under both IID and label-skew non-IID partitioning. The appropriate claim is comparability rather than superiority: the sample contains only three seeds, and the experiment was designed as a controlled robustness check rather than a population-level statistical comparison.

Figure 3 supports a separate model-selection argument. Continued optimization to Round 50 did not guarantee a higher validation F1. This should be discussed as evidence that convergence of the training procedure and selection of the most useful validation checkpoint are related but distinct decisions.

## Caveats

- The centralized condition uses its selected best-validation-F1 checkpoint, whereas the displayed federated comparison uses Round 50.
- Best-round federated results are reported separately in Tables 2 and 4.
- Mean and standard deviation are descriptive summaries over three seeds.
- No significance test is implied by these figures.
"""
    output_path.write_text(text, encoding="utf-8")


def write_artifact_catalog(output_path: Path) -> None:
    text = """# Predictive-results artifact catalog

## TABLE_01

**Title:** Centralized selected-checkpoint performance by seed  
**Filename:** `table01_centralized_selected_checkpoint_by_seed.csv`  
**Purpose:** Documents the selected epoch and predictive metrics for each centralized replication.  
**Primary conclusion:** Centralized predictive performance was reproducible, although the selected epoch varied by seed.

## TABLE_02

**Title:** Federated best-round and final-round performance by seed  
**Filename:** `table02_federated_best_and_final_by_seed.csv`  
**Purpose:** Keeps best validation-F1 and final Round-50 results explicitly separate.  
**Primary conclusion:** The strongest validation-F1 checkpoint generally occurred before Round 50.

## TABLE_03

**Title:** Selected/final predictive comparison by seed  
**Filename:** `table03_predictive_comparison_by_seed.csv`  
**Purpose:** Supplies the exact per-seed values plotted in Figures 1 and 2.  
**Primary conclusion:** Centralized, IID, and non-IID conditions achieved broadly comparable predictive performance.

## TABLE_04

**Title:** Three-seed descriptive predictive summary  
**Filename:** `table04_predictive_comparison_three_seed_summary.csv`  
**Purpose:** Reports mean, sample standard deviation, minimum, and maximum for each metric and condition.  
**Primary conclusion:** Cross-seed differences between training conditions were modest relative to their overall performance.

## FIGURE_01

**Title:** F1 and balanced accuracy across centralized and federated training  
**Files:** `fig01_final_f1_balanced_accuracy_three_seeds.png` and `.pdf`  
**Panels:** F1; balanced accuracy.  
**Primary conclusion:** Final/selected predictive performance was broadly comparable across conditions.  
**Suggested placement:** Full-width figure followed by two Results paragraphs.  
**Caption:** Selected centralized checkpoints and final Round-50 federated performance across seeds 42, 7, and 123. Lines connect results from the same seed; diamonds show the across-seed mean with sample standard deviation. Centralized checkpoints were selected using validation F1, whereas federated values correspond to Round 50.

## FIGURE_02

**Title:** Accuracy and ROC AUC across centralized and federated training  
**Files:** `fig02_final_accuracy_auc_three_seeds.png` and `.pdf`  
**Panels:** Accuracy; ROC AUC.  
**Primary conclusion:** The absence of a major performance collapse was also visible in overall accuracy and ranking ability.  
**Suggested placement:** After the interpretation of Figure 1, with one introductory and two follow-up paragraphs.  
**Caption:** Accuracy and ROC AUC for the selected centralized checkpoint and final federated models across three seeds. Individual lines represent paired seed results; diamonds show the mean and sample standard deviation.

## FIGURE_03

**Title:** Timing and magnitude of best-versus-final federated performance  
**Files:** `fig03_best_round_and_final_f1_gap.png` and `.pdf`  
**Panels:** Communication round of peak validation F1; peak-F1 improvement over Round 50 in percentage points.  
**Primary conclusion:** The final round was not consistently the best validation-F1 checkpoint, although the performance difference was modest.  
**Suggested placement:** Separate from general performance-comparison figures because it supports a model-selection conclusion.  
**Caption:** Communication round at which validation F1 was highest and the corresponding improvement over final Round-50 F1, expressed in percentage points. Lines connect IID and non-IID results obtained with the same seed. The dashed reference lines indicate Round 50 in panel (a) and no best-versus-final difference in panel (b).

## FIGURE_04

**Title:** IID federated predictive convergence across 50 communication rounds  
**Files:** `fig04_iid_predictive_convergence.png` and `.pdf`  
**Panels:** Validation F1 over rounds; validation ROC AUC over rounds.  
**Primary conclusion:** IID FedAvg improved rapidly and entered a strong performance region before the final communication round, while smaller seed-specific fluctuations remained.  
**Suggested placement:** After Figure 3, with separate Results discussion for threshold-dependent and ranking-oriented convergence.  
**Visual structure:** Three thin seed trajectories, one thicker three-seed mean, and a sample-standard-deviation band in each panel.

## FIGURE_05

**Title:** Non-IID federated predictive convergence across 50 communication rounds  
**Files:** `fig05_noniid_predictive_convergence.png` and `.pdf`  
**Panels:** Validation F1 over rounds; validation ROC AUC over rounds.  
**Primary conclusion:** Label-skew non-IID FedAvg also converged to strong predictive performance without a persistent late-round collapse.  
**Suggested placement:** After Figure 4 so that the non-IID trajectory receives independent discussion space rather than being compressed into one crowded comparison dashboard.  
**Visual structure:** Three thin seed trajectories, one thicker three-seed mean, and a sample-standard-deviation band in each panel.

## SUMMARY_01

**Title:** Predictive performance across three seeds  
**Filename:** `summary01_predictive_performance_results.md`  
**Purpose:** Provides a Results-ready paragraph sequence, Discussion links, and caveats based on the generated tables.
"""
    output_path.write_text(text, encoding="utf-8")


def write_manifest(output_path: Path, output_dir: Path) -> None:
    entries = [
        ("TABLE_01", "table", "table01_centralized_selected_checkpoint_by_seed.csv"),
        ("TABLE_02", "table", "table02_federated_best_and_final_by_seed.csv"),
        ("TABLE_03", "table", "table03_predictive_comparison_by_seed.csv"),
        ("TABLE_04", "table", "table04_predictive_comparison_three_seed_summary.csv"),
        ("FIGURE_01", "figure_png", "fig01_final_f1_balanced_accuracy_three_seeds.png"),
        ("FIGURE_01", "figure_pdf", "fig01_final_f1_balanced_accuracy_three_seeds.pdf"),
        ("FIGURE_01_SOURCE", "source_table", "fig01_source_final_f1_balanced_accuracy.csv"),
        ("FIGURE_02", "figure_png", "fig02_final_accuracy_auc_three_seeds.png"),
        ("FIGURE_02", "figure_pdf", "fig02_final_accuracy_auc_three_seeds.pdf"),
        ("FIGURE_02_SOURCE", "source_table", "fig02_source_final_accuracy_auc.csv"),
        ("FIGURE_03", "figure_png", "fig03_best_round_and_final_f1_gap.png"),
        ("FIGURE_03", "figure_pdf", "fig03_best_round_and_final_f1_gap.pdf"),
        ("FIGURE_03_SOURCE", "source_table", "fig03_source_best_round_and_final_f1.csv"),
        ("FIGURE_04", "figure_png", "fig04_iid_predictive_convergence.png"),
        ("FIGURE_04", "figure_pdf", "fig04_iid_predictive_convergence.pdf"),
        ("FIGURE_04_SOURCE", "source_table", "fig04_source_iid_predictive_convergence.csv"),
        ("FIGURE_05", "figure_png", "fig05_noniid_predictive_convergence.png"),
        ("FIGURE_05", "figure_pdf", "fig05_noniid_predictive_convergence.pdf"),
        ("FIGURE_05_SOURCE", "source_table", "fig05_source_noniid_predictive_convergence.csv"),
        ("SUMMARY_01", "summary", "summary01_predictive_performance_results.md"),
        ("CATALOG_01", "catalog", "predictive_artifact_catalog.md"),
    ]
    rows = []
    for artifact_id, artifact_type, filename in entries:
        path = output_dir / filename
        rows.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "filename": filename,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    metrics_dir = (
        args.metrics_dir.expanduser().resolve()
        if args.metrics_dir
        else project_root / "outputs" / "metrics"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root / "outputs" / "thesis_results" / "predictive"
    )

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory does not exist: {metrics_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(metrics_dir)
    centralized, federated, comparison, aggregate = build_tables(sources)
    roundwise = build_federated_roundwise_source(sources)

    save_table(
        centralized,
        output_dir / "table01_centralized_selected_checkpoint_by_seed.csv",
    )
    save_table(
        federated,
        output_dir / "table02_federated_best_and_final_by_seed.csv",
    )
    save_table(
        comparison,
        output_dir / "table03_predictive_comparison_by_seed.csv",
    )
    save_table(
        aggregate,
        output_dir / "table04_predictive_comparison_three_seed_summary.csv",
    )

    save_table(
        comparison[
            [
                "seed",
                "display_condition",
                "f1",
                "balanced_accuracy",
                "experiment_id",
                "source_csv",
            ]
        ],
        output_dir / "fig01_source_final_f1_balanced_accuracy.csv",
    )
    save_table(
        comparison[
            [
                "seed",
                "display_condition",
                "accuracy",
                "auc",
                "experiment_id",
                "source_csv",
            ]
        ],
        output_dir / "fig02_source_final_accuracy_auc.csv",
    )

    best = federated[federated["selection_rule"] == "Best validation F1"].copy()
    final = federated[federated["selection_rule"] == "Final Round 50"].copy()
    fig03_source = best.merge(
        final,
        on=["seed", "condition"],
        suffixes=("_best", "_final"),
        validate="one_to_one",
    )
    fig03_source["f1_gap_best_minus_final"] = (
        fig03_source["f1_best"] - fig03_source["f1_final"]
    )
    fig03_source["f1_gap_best_minus_final_percentage_points"] = (
        fig03_source["f1_gap_best_minus_final"] * 100.0
    )
    save_table(
        fig03_source,
        output_dir / "fig03_source_best_round_and_final_f1.csv",
    )

    save_table(
        roundwise[roundwise["condition"] == "IID"],
        output_dir / "fig04_source_iid_predictive_convergence.csv",
    )
    save_table(
        roundwise[roundwise["condition"] == "Non-IID"],
        output_dir / "fig05_source_noniid_predictive_convergence.csv",
    )

    save_two_panel_figure(
        comparison=comparison,
        metric_a="f1",
        metric_b="balanced_accuracy",
        title_a="F1 across training conditions",
        title_b="Balanced accuracy across training conditions",
        png_path=output_dir / "fig01_final_f1_balanced_accuracy_three_seeds.png",
        pdf_path=output_dir / "fig01_final_f1_balanced_accuracy_three_seeds.pdf",
    )
    save_two_panel_figure(
        comparison=comparison,
        metric_a="accuracy",
        metric_b="auc",
        title_a="Accuracy across training conditions",
        title_b="ROC AUC across training conditions",
        png_path=output_dir / "fig02_final_accuracy_auc_three_seeds.png",
        pdf_path=output_dir / "fig02_final_accuracy_auc_three_seeds.pdf",
    )
    plot_best_round_and_gap(
        federated=federated,
        png_path=output_dir / "fig03_best_round_and_final_f1_gap.png",
        pdf_path=output_dir / "fig03_best_round_and_final_f1_gap.pdf",
    )

    plot_federated_convergence(
        roundwise=roundwise,
        condition="IID",
        figure_number=4,
        png_path=output_dir / "fig04_iid_predictive_convergence.png",
        pdf_path=output_dir / "fig04_iid_predictive_convergence.pdf",
    )
    plot_federated_convergence(
        roundwise=roundwise,
        condition="Non-IID",
        figure_number=5,
        png_path=output_dir / "fig05_noniid_predictive_convergence.png",
        pdf_path=output_dir / "fig05_noniid_predictive_convergence.pdf",
    )

    write_results_summary(
        output_path=output_dir / "summary01_predictive_performance_results.md",
        centralized=centralized,
        federated=federated,
        aggregate=aggregate,
    )
    write_artifact_catalog(output_dir / "predictive_artifact_catalog.md")
    write_manifest(
        output_path=output_dir / "predictive_artifact_manifest.csv",
        output_dir=output_dir,
    )

    print("Predictive-results package created:")
    print(output_dir)
    print("\nDiscovered official source files:")
    for key in sorted(sources):
        source = sources[key]
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