#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SEEDS: tuple[int, ...] = (42, 7, 123)
CONDITIONS: tuple[str, ...] = ("IID", "Non-IID")
COMMON_ROUNDS: tuple[int, ...] = (0, 5, 10, 20, 30, 50)
EARLY_ADJACENT_ROUNDS: tuple[int, ...] = (5, 10)
EXPECTED_ROUNDS = {
    42: (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
    7: COMMON_ROUNDS,
    123: COMMON_ROUNDS,
}
EXPERIMENT_IDS = {
    "IID": "exp04_roundwise_gradcam_iid_seed{seed}",
    "Non-IID": "exp04_roundwise_gradcam_noniid_seed{seed}",
}
EXPECTED_SPLIT_TYPES = {
    "IID": {"iid"},
    "Non-IID": {"non_iid_label_skew"},
}

ALIASES: dict[str, tuple[str, ...]] = {
    "round": ("round", "checkpoint_round", "current_round"),
    "image_id": ("image_id",),
    "seed": ("seed",),
    "split_type": ("split_type", "source_split_type"),
    "previous_status": (
        "to_previous_comparison_status",
        "comparison_status_to_previous",
        "previous_comparison_status",
        "status_to_previous",
    ),
    "final_status": (
        "to_final_comparison_status",
        "comparison_status_to_final",
        "final_comparison_status",
        "status_to_final",
    ),
    "pearson_previous": (
        "pearson_correlation_to_previous",
        "pearson_to_previous",
        "to_previous_pearson",
        "similarity_to_previous_pearson",
        "pearson_previous",
    ),
    "dice_previous": (
        "dice_score_to_previous",
        "dice_to_previous",
        "to_previous_dice",
        "similarity_to_previous_dice",
        "dice_previous",
    ),
    "pearson_final": (
        "pearson_correlation_to_final",
        "pearson_to_final",
        "to_final_pearson",
        "similarity_to_final_pearson",
        "pearson_final",
    ),
    "dice_final": (
        "dice_score_to_final",
        "dice_to_final",
        "to_final_dice",
        "similarity_to_final_dice",
        "dice_final",
    ),
    "degenerate_previous": (
        "is_degenerate_pair_to_previous",
        "degenerate_pair_to_previous",
        "to_previous_is_degenerate_pair",
    ),
    "degenerate_final": (
        "is_degenerate_pair_to_final",
        "degenerate_pair_to_final",
        "to_final_is_degenerate_pair",
    ),
}

EXCLUDED_FILENAME_PARTS = {
    "sanity",
    "summary",
    "quantitative_summary",
    "qualitative",
    "manifest",
    "snapshot",
    "source",
    "artifact",
    "aligned",
    "candidate",
    "selection",
}


@dataclass(frozen=True)
class XaiSource:
    condition: str
    seed: int
    experiment_id: str
    path: Path
    dataframe: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create thesis-ready three-seed Experiment 4 Grad-CAM "
            "stability tables and figures."
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
            "Override output directory. Default: "
            "<project-root>/outputs/thesis_results/xai_stability"
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


def discover_similarity_file(
    metrics_dir: Path,
    experiment_id: str,
    expected_rows: int,
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
                ALIASES["previous_status"],
                "previous comparison status",
            )
            first_existing_column(
                header,
                ALIASES["final_status"],
                "final comparison status",
            )
        except KeyError:
            continue

        score = 0
        if experiment_id.lower() in lower_name:
            score += 12
        if "roundwise_xai_similarity" in lower_name:
            score += 6
        if "roundwise" in lower_name:
            score += 2

        if "experiment_id" in header.columns:
            ids = header["experiment_id"].dropna().astype(str)
            if (ids == experiment_id).any():
                score += 14
            elif experiment_id.lower() in " ".join(ids.tolist()).lower():
                score += 5
            elif experiment_id.lower() not in lower_name:
                continue
        elif experiment_id.lower() not in lower_name:
            continue

        full = read_csv_safely(path)
        if full is None:
            continue
        if len(full) == expected_rows:
            score += 8
        else:
            score -= abs(len(full) - expected_rows)

        candidates.append((score, path))

    if not candidates:
        raise FileNotFoundError(
            f"No official Experiment 4 similarity CSV found for "
            f"{experiment_id!r} under {metrics_dir}."
        )

    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    top_score = candidates[0][0]
    top = [path for score, path in candidates if score == top_score]

    if len(top) != 1:
        rendered = "\n".join(f"  - {path}" for path in top)
        raise RuntimeError(
            f"Multiple equally likely similarity files found for "
            f"{experiment_id}:\n{rendered}"
        )

    return top[0]


def normalize_source(
    condition: str,
    seed: int,
    experiment_id: str,
    path: Path,
) -> XaiSource:
    raw = pd.read_csv(path, low_memory=False)
    expected_rounds = EXPECTED_ROUNDS[seed]
    expected_rows = len(expected_rounds) * 200

    if len(raw) != expected_rows:
        raise ValueError(
            f"{experiment_id}: expected {expected_rows} rows, found {len(raw)}."
        )

    resolved = {
        name: first_existing_column(
            raw,
            aliases,
            name,
            required=name not in {"seed", "split_type", "degenerate_previous", "degenerate_final"},
        )
        for name, aliases in ALIASES.items()
    }

    normalized = pd.DataFrame()
    normalized["condition"] = condition
    normalized["seed"] = seed
    normalized["experiment_id"] = experiment_id
    normalized["source_csv"] = str(path)
    normalized["image_id"] = raw[resolved["image_id"]].astype(str)
    normalized["round"] = pd.to_numeric(
        raw[resolved["round"]], errors="raise"
    ).astype(int)

    normalized["previous_status"] = (
        raw[resolved["previous_status"]].astype(str).str.strip()
    )
    normalized["final_status"] = (
        raw[resolved["final_status"]].astype(str).str.strip()
    )

    for logical_name in (
        "pearson_previous",
        "dice_previous",
        "pearson_final",
        "dice_final",
    ):
        normalized[logical_name] = pd.to_numeric(
            raw[resolved[logical_name]], errors="coerce"
        )

    if resolved["degenerate_previous"] is not None:
        normalized["degenerate_previous"] = bool_series(
            raw[resolved["degenerate_previous"]]
        )
    else:
        normalized["degenerate_previous"] = False

    if resolved["degenerate_final"] is not None:
        normalized["degenerate_final"] = bool_series(
            raw[resolved["degenerate_final"]]
        )
    else:
        normalized["degenerate_final"] = False

    if resolved["seed"] is not None:
        unique_seeds = set(
            pd.to_numeric(raw[resolved["seed"]], errors="raise")
            .astype(int)
            .unique()
        )
        if unique_seeds != {seed}:
            raise ValueError(
                f"{experiment_id}: unexpected seed values {sorted(unique_seeds)}."
            )

    if resolved["split_type"] is not None:
        split_values = set(
            raw[resolved["split_type"]]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )
        if split_values and not split_values.issubset(
            EXPECTED_SPLIT_TYPES[condition]
        ):
            raise ValueError(
                f"{experiment_id}: unexpected split-type values "
                f"{sorted(split_values)}."
            )

    rounds = tuple(sorted(normalized["round"].unique()))
    if rounds != tuple(expected_rounds):
        raise ValueError(
            f"{experiment_id}: expected rounds {expected_rounds}, found {rounds}."
        )

    counts = normalized.groupby("round", observed=True).size()
    if not (counts == 200).all():
        raise ValueError(
            f"{experiment_id}: every round must contain 200 rows. "
            f"Found: {counts.to_dict()}"
        )

    duplicate_pairs = normalized.duplicated(["image_id", "round"]).sum()
    if duplicate_pairs:
        raise ValueError(
            f"{experiment_id}: found {duplicate_pairs} duplicate image-round pairs."
        )

    if normalized["image_id"].nunique() != 200:
        raise ValueError(
            f"{experiment_id}: expected 200 unique images, found "
            f"{normalized['image_id'].nunique()}."
        )

    if normalized["degenerate_previous"].any():
        raise ValueError(f"{experiment_id}: degenerate previous-pair rows found.")
    if normalized["degenerate_final"].any():
        raise ValueError(f"{experiment_id}: degenerate final-pair rows found.")

    computed_previous = normalized["previous_status"].eq("computed")
    computed_final = normalized["final_status"].isin(
        {"computed", "self_comparison"}
    )

    for column in ("pearson_previous", "dice_previous"):
        values = normalized.loc[computed_previous, column]
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(
                f"{experiment_id}: {column} contains invalid values for "
                f"computed comparisons."
            )

    for column in ("pearson_final", "dice_final"):
        values = normalized.loc[computed_final, column]
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(
                f"{experiment_id}: {column} contains invalid values for "
                f"computed/self comparisons."
            )

    return XaiSource(
        condition=condition,
        seed=seed,
        experiment_id=experiment_id,
        path=path,
        dataframe=normalized,
    )


def load_sources(metrics_dir: Path) -> dict[tuple[str, int], XaiSource]:
    sources: dict[tuple[str, int], XaiSource] = {}

    for condition in CONDITIONS:
        for seed in SEEDS:
            experiment_id = EXPERIMENT_IDS[condition].format(seed=seed)
            expected_rows = len(EXPECTED_ROUNDS[seed]) * 200
            path = discover_similarity_file(
                metrics_dir=metrics_dir,
                experiment_id=experiment_id,
                expected_rows=expected_rows,
            )
            sources[(condition, seed)] = normalize_source(
                condition=condition,
                seed=seed,
                experiment_id=experiment_id,
                path=path,
            )

    return sources


def build_seed_round_summary(
    sources: dict[tuple[str, int], XaiSource]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            source = sources[(condition, seed)]
            for round_number, group in source.dataframe.groupby(
                "round", observed=True
            ):
                previous_computed = group["previous_status"].eq("computed")
                previous_changed = group["previous_status"].eq(
                    "skipped_target_changed"
                )
                previous_not_applicable = group["previous_status"].eq(
                    "not_applicable_first_round"
                )

                final_valid = group["final_status"].isin(
                    {"computed", "self_comparison"}
                )
                final_changed = group["final_status"].eq(
                    "skipped_target_changed"
                )
                final_self = group["final_status"].eq("self_comparison")

                row: dict[str, object] = {
                    "condition": condition,
                    "seed": seed,
                    "round": int(round_number),
                    "n_images": len(group),
                    "previous_computed_count": int(previous_computed.sum()),
                    "previous_target_changed_count": int(previous_changed.sum()),
                    "previous_not_applicable_count": int(
                        previous_not_applicable.sum()
                    ),
                    "previous_target_changed_rate_percent": (
                        float(previous_changed.mean() * 100.0)
                    ),
                    "final_valid_count": int(final_valid.sum()),
                    "final_target_changed_count": int(final_changed.sum()),
                    "final_self_comparison_count": int(final_self.sum()),
                    "final_target_changed_rate_percent": (
                        float(final_changed.mean() * 100.0)
                    ),
                    "experiment_id": source.experiment_id,
                    "source_csv": str(source.path),
                }

                for metric in ("pearson", "dice"):
                    previous_values = group.loc[
                        previous_computed, f"{metric}_previous"
                    ]
                    final_values = group.loc[
                        final_valid, f"{metric}_final"
                    ]
                    row[f"{metric}_previous_mean"] = (
                        float(previous_values.mean())
                        if not previous_values.empty
                        else np.nan
                    )
                    row[f"{metric}_final_mean"] = (
                        float(final_values.mean())
                        if not final_values.empty
                        else np.nan
                    )

                rows.append(row)

    summary = pd.DataFrame(rows)
    summary["_condition_order"] = pd.Categorical(
        summary["condition"], categories=list(CONDITIONS), ordered=True
    )
    summary["_seed_order"] = pd.Categorical(
        summary["seed"], categories=list(SEEDS), ordered=True
    )
    summary = (
        summary.sort_values(["_condition_order", "_seed_order", "round"])
        .drop(columns=["_condition_order", "_seed_order"])
        .reset_index(drop=True)
    )
    return summary


def build_three_seed_summary(seed_round: pd.DataFrame) -> pd.DataFrame:
    common = seed_round[seed_round["round"].isin(COMMON_ROUNDS)].copy()
    value_columns = [
        "pearson_previous_mean",
        "dice_previous_mean",
        "pearson_final_mean",
        "dice_final_mean",
        "previous_computed_count",
        "previous_target_changed_count",
        "previous_target_changed_rate_percent",
        "final_valid_count",
        "final_target_changed_count",
        "final_target_changed_rate_percent",
    ]

    rows: list[dict[str, object]] = []
    for (condition, round_number), group in common.groupby(
        ["condition", "round"], observed=True
    ):
        for value_column in value_columns:
            values = pd.to_numeric(group[value_column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "condition": condition,
                    "round": int(round_number),
                    "measure": value_column,
                    "mean": float(values.mean()),
                    "standard_deviation": (
                        float(values.std(ddof=1)) if len(values) > 1 else np.nan
                    ),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "n_seeds": int(len(values)),
                }
            )

    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.6f")


def get_condition_color(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str):
    line = ax.plot(x, y, linewidth=2.2, marker="D", label=label)[0]
    return line, line.get_color()


def plot_early_adjacent_stability(
    seed_round: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    early = seed_round[
        seed_round["round"].isin(EARLY_ADJACENT_ROUNDS)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    metrics = (
        ("pearson_previous_mean", "Pearson correlation"),
        ("dice_previous_mean", "Dice score"),
    )
    x = np.arange(len(EARLY_ADJACENT_ROUNDS), dtype=float)
    transition_labels = ["Round 0→5", "Round 5→10"]
    offsets = {"IID": -0.10, "Non-IID": 0.10}
    seed_markers = {42: "o", 7: "s", 123: "^"}
    condition_colors: dict[str, str] = {}

    for panel_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[panel_index]

        for condition in CONDITIONS:
            condition_data = early[early["condition"] == condition]
            grouped = (
                condition_data.groupby("round", observed=True)[metric]
                .agg(["mean", "std"])
                .reindex(EARLY_ADJACENT_ROUNDS)
            )
            positions = x + offsets[condition]
            _, condition_color = get_condition_color(
                ax,
                positions,
                grouped["mean"].to_numpy(dtype=float),
                condition,
            )
            condition_colors.setdefault(condition, condition_color)

            ax.errorbar(
                positions,
                grouped["mean"].to_numpy(dtype=float),
                yerr=grouped["std"].to_numpy(dtype=float),
                fmt="none",
                capsize=4,
                color=condition_color,
            )

            for seed in SEEDS:
                seed_data = (
                    condition_data[condition_data["seed"] == seed]
                    .set_index("round")
                    .reindex(EARLY_ADJACENT_ROUNDS)
                )
                ax.scatter(
                    positions,
                    seed_data[metric].to_numpy(dtype=float),
                    marker=seed_markers[seed],
                    color=condition_color,
                    alpha=0.70,
                    zorder=3,
                )

        panel_letter = "a" if panel_index == 0 else "b"
        ax.set_xticks(x, transition_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"({panel_letter}) Early adjacent-checkpoint {ylabel}"
        )
        ax.grid(axis="y", alpha=0.25)

        if metric == "pearson_previous_mean":
            ax.axhline(
                0,
                linestyle="--",
                linewidth=1.0,
                color="0.45",
                zorder=0,
            )

    fig.suptitle(
        "Early adjacent-checkpoint Grad-CAM stability",
        y=0.98,
    )

    condition_handles = [
        Line2D(
            [],
            [],
            color=condition_colors[condition],
            marker="D",
            linewidth=2.2,
            label=condition,
        )
        for condition in CONDITIONS
    ]
    seed_handles = [
        Line2D(
            [],
            [],
            color="0.35",
            marker=seed_markers[seed],
            linestyle="none",
            markersize=7,
            label=f"Seed {seed}",
        )
        for seed in SEEDS
    ]
    summary_handle = Line2D(
        [],
        [],
        color="0.25",
        marker="D",
        linewidth=1.3,
        label="Condition mean ± SD",
    )

    fig.legend(
        condition_handles + seed_handles + [summary_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, bottom=0.23, wspace=0.30)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def plot_similarity_to_final(
    seed_round: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    common = seed_round[
        seed_round["round"].isin(COMMON_ROUNDS)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    metrics = (
        ("pearson_final_mean", "Pearson correlation"),
        ("dice_final_mean", "Dice score"),
    )
    seed_markers = {42: "o", 7: "s", 123: "^"}
    condition_colors: dict[str, str] = {}

    for panel_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[panel_index]

        for condition in CONDITIONS:
            condition_data = common[common["condition"] == condition]
            pivot = (
                condition_data.pivot(
                    index="round", columns="seed", values=metric
                )
                .reindex(COMMON_ROUNDS)
            )
            means = pivot.mean(axis=1)
            stds = pivot.std(axis=1, ddof=1)
            rounds = np.asarray(COMMON_ROUNDS, dtype=float)

            _, condition_color = get_condition_color(
                ax,
                rounds,
                means.to_numpy(dtype=float),
                f"{condition} mean",
            )
            condition_colors.setdefault(condition, condition_color)

            ax.fill_between(
                rounds,
                means.to_numpy(dtype=float) - stds.to_numpy(dtype=float),
                means.to_numpy(dtype=float) + stds.to_numpy(dtype=float),
                color=condition_color,
                alpha=0.15,
            )

            for seed in SEEDS:
                ax.plot(
                    rounds,
                    pivot[seed].to_numpy(dtype=float),
                    linewidth=0.9,
                    marker=seed_markers[seed],
                    markersize=4,
                    color=condition_color,
                    alpha=0.48,
                )

        panel_letter = "a" if panel_index == 0 else "b"
        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"({panel_letter}) {ylabel} relative to Round 50"
        )
        ax.set_xticks(COMMON_ROUNDS)
        ax.set_xlim(0, 50)
        ax.grid(axis="both", alpha=0.22)

        if metric == "pearson_final_mean":
            ax.axhline(
                0,
                linestyle="--",
                linewidth=1.0,
                color="0.45",
                zorder=0,
            )

    fig.suptitle(
        "Grad-CAM similarity to the final Round-50 explanation",
        y=0.98,
    )

    condition_handles = [
        Line2D(
            [],
            [],
            color=condition_colors[condition],
            marker="D",
            linewidth=2.2,
            label=f"{condition} mean",
        )
        for condition in CONDITIONS
    ]
    seed_handles = [
        Line2D(
            [],
            [],
            color="0.35",
            marker=seed_markers[seed],
            linestyle="none",
            markersize=7,
            label=f"Seed {seed}",
        )
        for seed in SEEDS
    ]
    band_handle = Patch(
        facecolor="0.75",
        edgecolor="0.55",
        alpha=0.35,
        label="Condition mean ± SD",
    )

    fig.legend(
        condition_handles + seed_handles + [band_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, bottom=0.23, wspace=0.30)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def plot_target_mismatch_to_final(
    seed_round: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    common = seed_round[
        seed_round["round"].isin(COMMON_ROUNDS)
    ].copy()

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.4))
    seed_markers = {42: "o", 7: "s", 123: "^"}
    condition_colors: dict[str, str] = {}

    for condition in CONDITIONS:
        condition_data = common[common["condition"] == condition]
        pivot = (
            condition_data.pivot(
                index="round",
                columns="seed",
                values="final_target_changed_rate_percent",
            )
            .reindex(COMMON_ROUNDS)
        )
        means = pivot.mean(axis=1)
        stds = pivot.std(axis=1, ddof=1)
        rounds = np.asarray(COMMON_ROUNDS, dtype=float)

        _, condition_color = get_condition_color(
            ax,
            rounds,
            means.to_numpy(dtype=float),
            f"{condition} mean",
        )
        condition_colors.setdefault(condition, condition_color)

        lower = np.clip(
            means.to_numpy(dtype=float) - stds.to_numpy(dtype=float),
            0.0,
            100.0,
        )
        upper = np.clip(
            means.to_numpy(dtype=float) + stds.to_numpy(dtype=float),
            0.0,
            100.0,
        )
        ax.fill_between(
            rounds,
            lower,
            upper,
            color=condition_color,
            alpha=0.15,
        )

        for seed in SEEDS:
            ax.plot(
                rounds,
                pivot[seed].to_numpy(dtype=float),
                linewidth=0.9,
                marker=seed_markers[seed],
                markersize=4,
                color=condition_color,
                alpha=0.48,
            )

    ax.set_xlabel("Communication round")
    ax.set_ylabel(
        "Images with a different predicted target from Round 50 (%)"
    )
    ax.set_title("Prediction-target mismatch relative to Round 50")
    ax.set_xticks(COMMON_ROUNDS)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.grid(axis="both", alpha=0.22)

    condition_handles = [
        Line2D(
            [],
            [],
            color=condition_colors[condition],
            marker="D",
            linewidth=2.2,
            label=f"{condition} mean",
        )
        for condition in CONDITIONS
    ]
    seed_handles = [
        Line2D(
            [],
            [],
            color="0.35",
            marker=seed_markers[seed],
            linestyle="none",
            markersize=7,
            label=f"Seed {seed}",
        )
        for seed in SEEDS
    ]
    band_handle = Patch(
        facecolor="0.75",
        edgecolor="0.55",
        alpha=0.35,
        label="Condition mean ± SD",
    )

    fig.legend(
        condition_handles + seed_handles + [band_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def write_artifact_catalog(path: Path) -> None:
    text = """# XAI-stability artifact catalog

## TABLE_05

**Title:** Experiment 4 Grad-CAM stability by seed and checkpoint  
**Filename:** `table05_xai_stability_by_seed_and_round.csv`  
**Purpose:** Stores same-target comparison counts, target-change rates, adjacent similarity, and similarity to Round 50 for every official seed and condition.

## TABLE_06

**Title:** Three-seed Grad-CAM stability summary at common checkpoints  
**Filename:** `table06_xai_stability_three_seed_summary.csv`  
**Purpose:** Reports mean, sample standard deviation, minimum, and maximum at the checkpoints shared by all seeds.

## FIGURE_06

**Structured name:** Early adjacent-checkpoint Grad-CAM stability  
**Files:** `fig06_early_adjacent_gradcam_stability.png` and `.pdf`  
**Figure-level title:** Early adjacent-checkpoint Grad-CAM stability  
**Panels:** Pearson correlation; Dice score.  
**Scope:** Only the fully comparable transitions 0→5 and 5→10.  
**Primary conclusion:** Explanations changed strongly in the first training interval and became substantially more stable by Round 10.

## FIGURE_07

**Structured name:** Grad-CAM convergence toward the final Round-50 explanation  
**Files:** `fig07_gradcam_similarity_to_final.png` and `.pdf`  
**Figure-level title:** Grad-CAM similarity to the final Round-50 explanation  
**Panels:** Pearson correlation to Round 50; Dice overlap with Round 50.  
**Scope:** Common checkpoints 0, 5, 10, 20, 30, and 50.  
**Primary conclusion:** Both IID and non-IID explanations progressively approached the final explanation pattern.

## FIGURE_08

**Structured name:** Prediction-target mismatch relative to Round 50  
**Files:** `fig08_prediction_target_mismatch_to_final.png` and `.pdf`  
**Panels:** One line-chart panel.  
**Scope:** Common checkpoints 0, 5, 10, 20, 30, and 50.  
**Primary conclusion:** Predicted targets became increasingly consistent with the final model during training.

## Interpretation boundary

Adjacent-checkpoint values after Round 10 are not pooled across all three seeds because the checkpoint spacing differs. Seed 42 used additional intermediate checkpoints, while seeds 7 and 123 used the reduced checkpoint set. Similarity to Round 50 is used for the aligned three-seed later-round analysis.
"""
    path.write_text(text, encoding="utf-8")


def write_manifest(path: Path, output_dir: Path) -> None:
    entries = [
        ("TABLE_05", "table", "table05_xai_stability_by_seed_and_round.csv"),
        ("TABLE_06", "table", "table06_xai_stability_three_seed_summary.csv"),
        ("FIGURE_06", "figure_png", "fig06_early_adjacent_gradcam_stability.png"),
        ("FIGURE_06", "figure_pdf", "fig06_early_adjacent_gradcam_stability.pdf"),
        ("FIGURE_06_SOURCE", "source_table", "fig06_source_early_adjacent_gradcam_stability.csv"),
        ("FIGURE_07", "figure_png", "fig07_gradcam_similarity_to_final.png"),
        ("FIGURE_07", "figure_pdf", "fig07_gradcam_similarity_to_final.pdf"),
        ("FIGURE_07_SOURCE", "source_table", "fig07_source_gradcam_similarity_to_final.csv"),
        ("FIGURE_08", "figure_png", "fig08_prediction_target_mismatch_to_final.png"),
        ("FIGURE_08", "figure_pdf", "fig08_prediction_target_mismatch_to_final.pdf"),
        ("FIGURE_08_SOURCE", "source_table", "fig08_source_prediction_target_mismatch_to_final.csv"),
        ("CATALOG_02", "catalog", "xai_stability_artifact_catalog.md"),
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
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root / "outputs" / "thesis_results" / "xai_stability"
    )

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory does not exist: {metrics_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(metrics_dir)
    seed_round = build_seed_round_summary(sources)
    three_seed = build_three_seed_summary(seed_round)

    save_table(
        seed_round,
        output_dir / "table05_xai_stability_by_seed_and_round.csv",
    )
    save_table(
        three_seed,
        output_dir / "table06_xai_stability_three_seed_summary.csv",
    )

    early_source = seed_round[
        seed_round["round"].isin(EARLY_ADJACENT_ROUNDS)
    ].copy()
    final_source = seed_round[
        seed_round["round"].isin(COMMON_ROUNDS)
    ].copy()

    save_table(
        early_source[
            [
                "condition",
                "seed",
                "round",
                "pearson_previous_mean",
                "dice_previous_mean",
                "previous_computed_count",
                "previous_target_changed_count",
                "previous_target_changed_rate_percent",
                "experiment_id",
                "source_csv",
            ]
        ],
        output_dir / "fig06_source_early_adjacent_gradcam_stability.csv",
    )
    save_table(
        final_source[
            [
                "condition",
                "seed",
                "round",
                "pearson_final_mean",
                "dice_final_mean",
                "final_valid_count",
                "final_target_changed_count",
                "final_target_changed_rate_percent",
                "experiment_id",
                "source_csv",
            ]
        ],
        output_dir / "fig07_source_gradcam_similarity_to_final.csv",
    )
    save_table(
        final_source[
            [
                "condition",
                "seed",
                "round",
                "final_target_changed_rate_percent",
                "final_target_changed_count",
                "final_valid_count",
                "experiment_id",
                "source_csv",
            ]
        ],
        output_dir / "fig08_source_prediction_target_mismatch_to_final.csv",
    )

    plot_early_adjacent_stability(
        seed_round=seed_round,
        png_path=output_dir / "fig06_early_adjacent_gradcam_stability.png",
        pdf_path=output_dir / "fig06_early_adjacent_gradcam_stability.pdf",
    )
    plot_similarity_to_final(
        seed_round=seed_round,
        png_path=output_dir / "fig07_gradcam_similarity_to_final.png",
        pdf_path=output_dir / "fig07_gradcam_similarity_to_final.pdf",
    )
    plot_target_mismatch_to_final(
        seed_round=seed_round,
        png_path=output_dir / "fig08_prediction_target_mismatch_to_final.png",
        pdf_path=output_dir / "fig08_prediction_target_mismatch_to_final.pdf",
    )

    write_artifact_catalog(
        output_dir / "xai_stability_artifact_catalog.md"
    )
    write_manifest(
        output_dir / "xai_stability_artifact_manifest.csv",
        output_dir,
    )

    print("XAI-stability results package created:")
    print(output_dir)
    print("\nDiscovered official Experiment 4 sources:")
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
