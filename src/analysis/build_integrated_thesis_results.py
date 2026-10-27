#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

EXPECTED_ARTIFACTS = (
    ("TABLE_01", "table", "predictive/table01_centralized_selected_checkpoint_by_seed.csv"),
    ("TABLE_02", "table", "predictive/table02_federated_best_and_final_by_seed.csv"),
    ("TABLE_03", "table", "predictive/table03_predictive_comparison_by_seed.csv"),
    ("TABLE_04", "table", "predictive/table04_predictive_comparison_three_seed_summary.csv"),
    ("FIGURE_01", "figure", "predictive/fig01_final_f1_balanced_accuracy_three_seeds.png"),
    ("FIGURE_02", "figure", "predictive/fig02_final_accuracy_auc_three_seeds.png"),
    ("FIGURE_03", "figure", "predictive/fig03_best_round_and_final_f1_gap.png"),
    ("FIGURE_04", "figure", "predictive/fig04_iid_predictive_convergence.png"),
    ("FIGURE_05", "figure", "predictive/fig05_noniid_predictive_convergence.png"),
    ("TABLE_05", "table", "xai_stability/table05_xai_stability_by_seed_and_round.csv"),
    ("TABLE_06", "table", "xai_stability/table06_xai_stability_three_seed_summary.csv"),
    ("FIGURE_06", "figure", "xai_stability/fig06_early_adjacent_gradcam_stability.png"),
    ("FIGURE_07", "figure", "xai_stability/fig07_gradcam_similarity_to_final.png"),
    ("FIGURE_08", "figure", "xai_stability/fig08_prediction_target_mismatch_to_final.png"),
    ("TABLE_07", "table", "cross_model_xai/table07_cross_model_similarity_by_seed.csv"),
    ("TABLE_08", "table", "cross_model_xai/table08_cross_model_similarity_three_seed_summary.csv"),
    ("FIGURE_09", "figure", "cross_model_xai/fig09_cross_model_gradcam_spatial_similarity.png"),
    ("FIGURE_10", "figure", "cross_model_xai/fig10_cross_model_prediction_agreement_confidence.png"),
)

FIGURE_ROWS = (
    ("FIGURE_01", "Predictive performance across training conditions", "predictive", "fig01_final_f1_balanced_accuracy_three_seeds.png", "Validation F1 and balanced accuracy across centralized, IID, and non-IID training."),
    ("FIGURE_02", "Overall accuracy and ranking performance", "predictive", "fig02_final_accuracy_auc_three_seeds.png", "Validation accuracy and ROC AUC across the three training conditions."),
    ("FIGURE_03", "Peak-versus-final federated performance", "predictive", "fig03_best_round_and_final_f1_gap.png", "Peak-F1 timing and improvement over Round 50."),
    ("FIGURE_04", "IID FedAvg predictive convergence", "predictive", "fig04_iid_predictive_convergence.png", "IID validation F1 and ROC-AUC trajectories over 50 rounds."),
    ("FIGURE_05", "Non-IID label-skew FedAvg predictive convergence", "predictive", "fig05_noniid_predictive_convergence.png", "Non-IID validation F1 and ROC-AUC trajectories over 50 rounds."),
    ("FIGURE_06", "Early adjacent-checkpoint Grad-CAM stability", "xai_stability", "fig06_early_adjacent_gradcam_stability.png", "Pearson and Dice stability for the aligned 0→5 and 5→10 transitions."),
    ("FIGURE_07", "Grad-CAM similarity to the final explanation", "xai_stability", "fig07_gradcam_similarity_to_final.png", "Pearson and Dice similarity to Round 50 at common checkpoints."),
    ("FIGURE_08", "Prediction-target mismatch relative to Round 50", "xai_stability", "fig08_prediction_target_mismatch_to_final.png", "Percentage of images with a different predicted target from Round 50."),
    ("FIGURE_09", "Cross-model Grad-CAM spatial similarity", "cross_model_xai", "fig09_cross_model_gradcam_spatial_similarity.png", "Pearson and Dice similarity across centralized and federated model pairs."),
    ("FIGURE_10", "Cross-model prediction agreement and confidence difference", "cross_model_xai", "fig10_cross_model_prediction_agreement_confidence.png", "Prediction agreement and confidence difference across model pairs."),
)

EVIDENCE_ROWS = (
    ("Predictive viability", "Can federated learning preserve predictive performance relative to centralized training?", "Figures 1–5; Tables 1–4", "The three conditions achieved closely grouped predictive performance across three seeds.", "Use comparability rather than superiority language; n=3 seeds."),
    ("Client heterogeneity", "Does label-skew non-IID training cause persistent predictive failure?", "Figures 1–3 and 5; Tables 2–4", "The tested non-IID condition converged to strong performance without persistent late collapse.", "Applies to the tested five-client label-skew configuration."),
    ("Explanation stability", "How do Grad-CAM explanations change across communication rounds?", "Figures 6–8; Tables 5–6", "Explanations changed strongly early and became increasingly stable relative to Round 50.", "Later adjacent intervals are not pooled when checkpoint spacing differs."),
    ("Cross-model explanation consistency", "Do models with similar predictions produce similar spatial explanations?", "Figures 9–10; Tables 7–8", "Prediction agreement was high while centralized-versus-federated Grad-CAM similarity remained moderate.", "Similar predictions do not imply equivalent spatial reasoning."),
    ("Federated explanation similarity", "How similar are IID and non-IID federated explanations?", "Figure 9; Tables 7–8", "IID and non-IID explanations were consistently more similar to each other than either was to centralized training.", "Do not claim centralized is systematically closer to IID or non-IID."),
)

ALIASES = {
    "category": (
        "thesis_display_category",
        "category",
        "selection_category",
        "qualitative_category",
        "case_category",
    ),
    "true_label": ("true_label", "target", "ground_truth", "label"),
    "centralized": (
        "centralized_predicted_label",
        "centralized_prediction",
        "centralized_predicted_class",
        "centralized_pred",
    ),
    "iid": (
        "iid_predicted_label",
        "iid_prediction",
        "iid_predicted_class",
        "iid_pred",
    ),
    "non_iid": (
        "non_iid_predicted_label",
        "non_iid_prediction",
        "non_iid_predicted_class",
        "non_iid_pred",
        "non_iid_label_skew_predicted_label",
    ),
    "centralized_confidence": (
        "centralized_confidence",
        "centralized_prediction_confidence",
        "centralized_conf",
        "centralized_mean_prediction_confidence",
    ),
    "iid_confidence": (
        "iid_confidence",
        "iid_prediction_confidence",
        "iid_conf",
        "iid_mean_prediction_confidence",
    ),
    "non_iid_confidence": (
        "non_iid_confidence",
        "non_iid_prediction_confidence",
        "non_iid_conf",
        "non_iid_mean_prediction_confidence",
        "non_iid_label_skew_confidence",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the integrated thesis-results package.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--thesis-results-dir", type=Path, default=None)
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--figure11-reviewed",
        action="store_true",
        help=(
            "Mark the regenerated Figure 11 as visually accepted. Use only "
            "after inspecting the generated PNG."
        ),
    )
    return parser.parse_args()


def first_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    return next((name for name in aliases if name in df.columns), None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unavailable"


def validate_core(thesis_results_dir: Path) -> pd.DataFrame:
    rows = []
    for artifact_id, artifact_type, relative_path in EXPECTED_ARTIFACTS:
        path = thesis_results_dir / relative_path
        exists = path.exists()
        size = path.stat().st_size if exists else None
        rows.append({
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "relative_path": relative_path,
            "exists": exists,
            "nonempty": bool(exists and size and size > 0),
            "size_bytes": size,
            "status": "passed" if exists and size and size > 0 else "missing_required",
        })
    return pd.DataFrame(rows)


def discover_first(roots: Sequence[Path], patterns: Sequence[str]) -> Path | None:
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            for pattern in patterns:
                matches.extend(root.rglob(pattern))
    return sorted({path.resolve() for path in matches})[0] if matches else None


def normalize_label(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip()
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text.lower()


def correct_selection(source: Path, destination: Path) -> str:
    df = pd.read_csv(source, low_memory=False)
    resolved = {name: first_column(df, aliases) for name, aliases in ALIASES.items()}
    category = resolved["category"]
    if category is None:
        df["original_category"] = pd.NA
        df["thesis_display_category"] = pd.NA
        status = "category_column_not_found"
    else:
        df["original_category"] = df[category].astype(str)
        df["thesis_display_category"] = df[category].astype(str)
        needed = (resolved["true_label"], resolved["centralized"], resolved["iid"], resolved["non_iid"])
        corrections = 0
        if all(column is not None for column in needed):
            for idx, row in df.iterrows():
                true_label = normalize_label(row[resolved["true_label"]])
                preds = [normalize_label(row[resolved[key]]) for key in ("centralized", "iid", "non_iid")]
                if all(not pd.isna(value) for value in preds) and preds[0] == preds[1] == preds[2] and not pd.isna(true_label) and preds[0] != true_label:
                    df.at[idx, "thesis_display_category"] = "Shared incorrect prediction and similar explanations"
                    corrections += 1
            status = f"prediction_based_correction_applied_to_{corrections}_rows"
        else:
            status = "prediction_columns_not_found"
    df.to_csv(destination, index=False)
    return status



CATEGORY_LABELS = {
    "all_correct_but_explanations_differ": (
        "All models correct, but explanations differ"
    ),
    "all_models_correct_but_explanations_differ": (
        "All models correct, but explanations differ"
    ),
    "predictions_and_explanations_strongly_agree": (
        "Predictions and explanations strongly agree"
    ),
    "all_models_agree_high_similarity": (
        "Predictions and explanations strongly agree"
    ),
    "federated_models_similar_centralized_differs": (
        "Federated models similar, centralized differs"
    ),
    "federated_explanations_agree_centralized_differs": (
        "Federated models similar, centralized differs"
    ),
    "prediction_target_disagreement": "Prediction-target disagreement",
    "shared_incorrect_prediction_and_similar_explanations": (
        "Shared incorrect prediction and similar explanations"
    ),
}


def humanize_category(value: object) -> str:
    text = str(value).strip()
    normalized = (
        text.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )
    if normalized in CATEGORY_LABELS:
        return CATEGORY_LABELS[normalized]
    if text:
        return text.replace("_", " ").strip().capitalize()
    return "Representative Grad-CAM case"


def contiguous_intervals(
    active: np.ndarray,
    minimum_length: int,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None

    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            if index - start >= minimum_length:
                intervals.append((start, index - 1))
            start = None

    if start is not None and len(active) - start >= minimum_length:
        intervals.append((start, len(active) - 1))

    return intervals


def extract_gradcam_panels(
    source_figure: Path,
    expected_rows: int = 4,
    expected_columns: int = 3,
) -> list[list[np.ndarray]]:
    """Extract the 4×3 heatmap panels from the original composite figure.

    The original figure has large non-white square image regions separated by
    white text/spacing. Detection is based on normalized non-white density,
    making it independent of the exact pixel dimensions.
    """
    image = Image.open(source_figure).convert("RGB")
    array = np.asarray(image)
    grayscale = array.mean(axis=2)
    non_white = grayscale < 245

    height, width = non_white.shape
    row_density = non_white.mean(axis=1)
    row_intervals = contiguous_intervals(
        row_density > 0.20,
        minimum_length=max(30, int(height * 0.10)),
    )

    row_intervals = sorted(
        sorted(
            row_intervals,
            key=lambda item: item[1] - item[0],
            reverse=True,
        )[:expected_rows]
    )

    if len(row_intervals) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} Grad-CAM image rows in "
            f"{source_figure}, detected {len(row_intervals)}: "
            f"{row_intervals}"
        )

    panels: list[list[np.ndarray]] = []

    for row_start, row_end in row_intervals:
        row_mask = non_white[row_start : row_end + 1]
        column_density = row_mask.mean(axis=0)
        column_intervals = contiguous_intervals(
            column_density > 0.20,
            minimum_length=max(30, int(width * 0.15)),
        )
        column_intervals = sorted(
            sorted(
                column_intervals,
                key=lambda item: item[1] - item[0],
                reverse=True,
            )[:expected_columns]
        )

        if len(column_intervals) != expected_columns:
            raise ValueError(
                f"Expected {expected_columns} Grad-CAM columns in image row "
                f"{row_start}:{row_end}, detected {len(column_intervals)}: "
                f"{column_intervals}"
            )

        row_panels: list[np.ndarray] = []
        for column_start, column_end in column_intervals:
            crop = array[
                row_start : row_end + 1,
                column_start : column_end + 1,
            ]
            row_panels.append(crop)
        panels.append(row_panels)

    return panels


def format_prediction(
    row: pd.Series,
    prediction_column: str | None,
    confidence_column: str | None,
) -> str:
    if prediction_column is None or pd.isna(row[prediction_column]):
        prediction_text = "Pred: unavailable"
    else:
        prediction = normalize_label(row[prediction_column])
        prediction_text = f"Pred: {prediction}"

    if confidence_column is None or pd.isna(row[confidence_column]):
        return prediction_text

    try:
        confidence = float(row[confidence_column])
    except (TypeError, ValueError):
        return prediction_text

    return f"{prediction_text} | Confidence: {confidence:.5f}"


def rebuild_qualitative_figure(
    original_figure: Path,
    corrected_selection_csv: Path,
    png_path: Path,
    pdf_path: Path,
) -> None:
    selection = pd.read_csv(corrected_selection_csv, low_memory=False)
    panels = extract_gradcam_panels(original_figure)

    if len(selection) != len(panels):
        raise ValueError(
            f"Qualitative selection contains {len(selection)} rows, while "
            f"the source figure contains {len(panels)} image rows."
        )

    resolved = {
        name: first_column(selection, aliases)
        for name, aliases in ALIASES.items()
    }

    model_columns = (
        (
            "Centralized",
            resolved["centralized"],
            resolved["centralized_confidence"],
        ),
        (
            "IID FedAvg",
            resolved["iid"],
            resolved["iid_confidence"],
        ),
        (
            "Non-IID FedAvg",
            resolved["non_iid"],
            resolved["non_iid_confidence"],
        ),
    )

    fig, axes = plt.subplots(
        nrows=len(panels),
        ncols=3,
        figsize=(12.8, 16.0),
    )
    if len(panels) == 1:
        axes = np.asarray([axes])

    for row_index, (panel_row, (_, row)) in enumerate(
        zip(panels, selection.iterrows())
    ):
        for column_index, (
            model_name,
            prediction_column,
            confidence_column,
        ) in enumerate(model_columns):
            ax = axes[row_index, column_index]
            ax.imshow(panel_row[column_index])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                format_prediction(
                    row,
                    prediction_column,
                    confidence_column,
                ),
                fontsize=10,
                pad=6,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.7)
                spine.set_edgecolor("0.35")

    fig.suptitle(
        "Representative Grad-CAM comparisons across training conditions",
        fontsize=17,
        y=0.992,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.835,
        bottom=0.025,
        hspace=0.40,
        wspace=0.08,
    )
    fig.canvas.draw()

    for column_index, (model_name, _, _) in enumerate(model_columns):
        position = axes[0, column_index].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.925,
            model_name,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    row_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    category_column = resolved["category"]
    true_label_column = resolved["true_label"]

    for row_index, (_, row) in enumerate(selection.iterrows()):
        if category_column is not None:
            category = humanize_category(row[category_column])
        else:
            category = "Representative Grad-CAM case"

        true_label_text = ""
        if (
            true_label_column is not None
            and not pd.isna(row[true_label_column])
        ):
            true_label = normalize_label(row[true_label_column])
            true_label_text = f" — True label: {true_label}"

        position = axes[row_index, 0].get_position()
        fig.text(
            0.055,
            position.y1 + 0.024,
            f"{row_letters[row_index]}. {category}{true_label_text}",
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_figure11_visual_specification(path: Path) -> None:
    path.write_text(
        "# Figure 11 visual specification\n\n"
        "Overall layout: four qualitative case rows by three model columns.\n\n"
        "Shared columns:\n"
        "- Centralized\n"
        "- IID FedAvg\n"
        "- Non-IID FedAvg\n\n"
        "Each row contains:\n"
        "- one row-level qualitative category;\n"
        "- the shared true label once;\n"
        "- one Grad-CAM overlay per model;\n"
        "- concise prediction and confidence text per model.\n\n"
        "The main figure intentionally omits image identifiers, internal "
        "experiment paths, and repeated model headings. Those details remain "
        "in the corrected source CSV and original-composite provenance file.\n",
        encoding="utf-8",
    )


def prepare_qualitative(
    project_root: Path,
    thesis_results_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    output_dir: Path,
    figure11_reviewed: bool,
) -> pd.DataFrame:
    qualitative_dir = output_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    figure_roots = (
        figures_dir,
        thesis_results_dir / "cross_model_xai" / "qualitative",
    )
    table_roots = (
        metrics_dir,
        thesis_results_dir / "cross_model_xai" / "qualitative",
    )

    figure = discover_first(
        figure_roots,
        (
            "exp05_cross_model_gradcam_seed42_qualitative_comparison.png",
            "*seed42*qualitative*comparison*.png",
        ),
    )
    selection = discover_first(
        table_roots,
        (
            "exp05_cross_model_gradcam_seed42_qualitative_figure_selection.csv",
            "*seed42*qualitative*selection*.csv",
        ),
    )
    candidates = discover_first(
        table_roots,
        (
            "exp05_cross_model_gradcam_seed42_qualitative_candidates.csv",
            "*seed42*qualitative*candidates*.csv",
        ),
    )

    rows = []

    corrected_selection: Path | None = None
    if selection:
        corrected_selection = (
            qualitative_dir
            / "fig11_source_seed42_qualitative_selection_corrected.csv"
        )
        correction_status = correct_selection(
            selection,
            corrected_selection,
        )
        rows.append(
            (
                "FIGURE_11_SELECTION",
                "qualitative_source_table",
                str(selection),
                str(corrected_selection),
                "copied_and_checked",
                correction_status,
            )
        )
    else:
        rows.append(
            (
                "FIGURE_11_SELECTION",
                "qualitative_source_table",
                None,
                None,
                "not_found",
                "No qualitative selection table found.",
            )
        )

    if candidates:
        destination = (
            qualitative_dir
            / "fig11_source_seed42_qualitative_candidates.csv"
        )
        shutil.copy2(candidates, destination)
        rows.append(
            (
                "FIGURE_11_CANDIDATES",
                "qualitative_source_table",
                str(candidates),
                str(destination),
                "copied",
                "Candidate table preserved without modification.",
            )
        )
    else:
        rows.append(
            (
                "FIGURE_11_CANDIDATES",
                "qualitative_source_table",
                None,
                None,
                "not_found",
                "No qualitative candidate table found.",
            )
        )

    if figure and corrected_selection is not None:
        original_copy = (
            qualitative_dir
            / "fig11_source_original_composite.png"
        )
        shutil.copy2(figure, original_copy)

        png_destination = (
            qualitative_dir
            / "fig11_seed42_qualitative_gradcam_comparison.png"
        )
        pdf_destination = (
            qualitative_dir
            / "fig11_seed42_qualitative_gradcam_comparison.pdf"
        )

        rebuild_qualitative_figure(
            original_figure=figure,
            corrected_selection_csv=corrected_selection,
            png_path=png_destination,
            pdf_path=pdf_destination,
        )
        review_status = (
            "visually_accepted"
            if figure11_reviewed
            else "generated_requires_visual_review"
        )
        rows.append(
            (
                "FIGURE_11",
                "qualitative_figure",
                str(figure),
                str(png_destination),
                review_status,
                (
                    "Polished 4×3 figure regenerated from the official "
                    "composite; original composite preserved separately."
                ),
            )
        )
        rows.append(
            (
                "FIGURE_11_PDF",
                "qualitative_figure_pdf",
                str(figure),
                str(pdf_destination),
                review_status,
                "PDF export of the polished qualitative figure.",
            )
        )
        rows.append(
            (
                "FIGURE_11_ORIGINAL",
                "qualitative_source_figure",
                str(figure),
                str(original_copy),
                "copied_for_provenance",
                "Unmodified original composite retained for traceability.",
            )
        )
    elif figure:
        destination = (
            qualitative_dir
            / "fig11_seed42_qualitative_gradcam_comparison.png"
        )
        shutil.copy2(figure, destination)
        rows.append(
            (
                "FIGURE_11",
                "qualitative_figure",
                str(figure),
                str(destination),
                "copied_selection_missing",
                (
                    "Original figure copied because the qualitative "
                    "selection table was unavailable."
                ),
            )
        )
    else:
        rows.append(
            (
                "FIGURE_11",
                "qualitative_figure",
                None,
                None,
                "not_found",
                "No qualitative figure found.",
            )
        )

    write_figure11_visual_specification(
        qualitative_dir / "fig11_visual_specification.md"
    )

    return pd.DataFrame(
        rows,
        columns=[
            "artifact_id",
            "artifact_type",
            "source_path",
            "output_path",
            "status",
            "notes",
        ],
    )


def build_figure_catalog(thesis_results_dir: Path, qualitative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for artifact_id, title, module, filename, visual_scope in FIGURE_ROWS:
        path = thesis_results_dir / module / filename
        rows.append((artifact_id, title, module, filename, str(path), path.exists(), visual_scope, "visually_accepted"))
    q = qualitative[qualitative["artifact_id"] == "FIGURE_11"]
    if not q.empty:
        item = q.iloc[0]
        rows.append(("FIGURE_11", "Representative cross-model Grad-CAM examples", "integrated_qualitative", Path(item["output_path"]).name if pd.notna(item["output_path"]) else None, item["output_path"], item["status"] != "not_found", "Representative seed-42 cases covering agreement and explanation differences.", item["status"]))
    return pd.DataFrame(rows, columns=["artifact_id", "title", "module", "filename", "path", "exists", "visual_scope", "review_status"])


def write_evidence_catalog(path: Path) -> None:
    path.write_text(
        "# Integrated thesis-results evidence catalog\n\n"
        "Stable findings:\n\n"
        "1. Centralized, IID FedAvg, and non-IID label-skew FedAvg achieved closely grouped predictive performance across three seeds.\n"
        "2. Peak federated validation F1 occurred before Round 50 in every run, with modest best-versus-final gaps.\n"
        "3. Grad-CAM explanations changed strongly early and then became increasingly stable.\n"
        "4. Prediction-target mismatch relative to Round 50 declined rapidly.\n"
        "5. IID and non-IID federated explanations were consistently more similar to each other than either was to centralized training.\n"
        "6. High prediction agreement did not guarantee high spatial explanation agreement.\n\n"
        "Interpretation boundaries:\n\n"
        "- Descriptive summaries use three seeds; no significance test is implied.\n"
        "- Centralized selected checkpoints and federated Round-50 checkpoints use different selection rules.\n"
        "- Later adjacent-checkpoint XAI values are not pooled across unequal checkpoint intervals.\n"
        "- Qualitative agreement does not establish correctness.\n",
        encoding="utf-8",
    )


def build_manifest(thesis_results_dir: Path, output_dir: Path, project_root: Path) -> pd.DataFrame:
    rows = []
    branch = git_value(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(project_root, "rev-parse", "HEAD")
    for path in sorted(thesis_results_dir.rglob("*")):
        if not path.is_file() or output_dir.resolve() in path.resolve().parents:
            continue
        rows.append({
            "relative_path": str(path.relative_to(project_root)),
            "filename": path.name,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_branch": branch,
            "git_commit": commit,
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    thesis_results_dir = args.thesis_results_dir.expanduser().resolve() if args.thesis_results_dir else project_root / "outputs" / "thesis_results"
    metrics_dir = args.metrics_dir.expanduser().resolve() if args.metrics_dir else project_root / "outputs" / "metrics"
    figures_dir = args.figures_dir.expanduser().resolve() if args.figures_dir else project_root / "outputs" / "figures"
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else thesis_results_dir / "integrated"

    if not thesis_results_dir.exists():
        raise FileNotFoundError(f"Thesis-results directory does not exist: {thesis_results_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_core(thesis_results_dir)
    validation.to_csv(output_dir / "table10_artifact_validation.csv", index=False)
    missing = validation[validation["status"] == "missing_required"]
    if not missing.empty:
        rendered = "\n".join(f"- {row.artifact_id}: {row.relative_path}" for row in missing.itertuples())
        raise FileNotFoundError(f"Required analysis artifacts are missing or empty:\n{rendered}")

    qualitative = prepare_qualitative(
        project_root,
        thesis_results_dir,
        metrics_dir,
        figures_dir,
        output_dir,
        figure11_reviewed=args.figure11_reviewed,
    )
    qualitative.to_csv(output_dir / "table11_qualitative_artifact_inventory.csv", index=False)

    build_figure_catalog(thesis_results_dir, qualitative).to_csv(output_dir / "table12_final_figure_catalog.csv", index=False)
    pd.DataFrame(EVIDENCE_ROWS, columns=["evidence_theme", "guiding_question", "supporting_artifacts", "supported_conclusion", "interpretation_boundary"]).to_csv(output_dir / "table13_cross_experiment_evidence_map.csv", index=False)
    write_evidence_catalog(output_dir / "integrated_evidence_catalog.md")
    build_manifest(thesis_results_dir, output_dir, project_root).to_csv(output_dir / "final_master_artifact_manifest.csv", index=False)

    branch = git_value(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(project_root, "rev-parse", "HEAD")
    status = git_value(project_root, "status", "--short") or "clean"
    snapshot = [
        "Integrated thesis-results provenance snapshot",
        "",
        f"Project root: {project_root}",
        f"Git branch: {branch}",
        f"Git commit: {commit}",
        f"Git status: {status}",
        "",
        f"Validated core artifacts: {(validation['status'] == 'passed').sum()}",
        "Qualitative artifact statuses:",
    ]
    snapshot.extend(f"- {row.artifact_id}: {row.status} ({row.notes})" for row in qualitative.itertuples())
    (output_dir / "final_provenance_snapshot.txt").write_text("\n".join(snapshot) + "\n", encoding="utf-8")

    print("Integrated thesis-results package created:")
    print(output_dir)
    print("\nCore artifact validation:")
    print(validation["status"].value_counts().to_string())
    print("\nQualitative artifacts:")
    for row in qualitative.itertuples():
        print(f"- {row.artifact_id}: {row.status}")
    print("\nGenerated integrated artifacts:")
    for path in sorted(output_dir.iterdir()):
        print(f"- {path.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise