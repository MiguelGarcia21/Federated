#!/usr/bin/env python3
"""Build a thesis-writing support package from validated analysis artifacts.

Run from the thesis-fl-xai repository root:

    python -m src.analysis.build_thesis_writing_package --project-root .

Outputs are written to:

    outputs/thesis_results/writing/

This module does not regenerate figures, retrain models, or alter experimental
results. It reads the validated tables and catalogs produced by Modules 1–4
and creates structured writing aids for the Results, Discussion, captions,
research-question answers, and limitations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the thesis Results and Discussion writing package."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root of the thesis-fl-xai repository.",
    )
    parser.add_argument(
        "--thesis-results-dir",
        type=Path,
        default=None,
        help=(
            "Override thesis-results directory. Default: "
            "<project-root>/outputs/thesis_results"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override writing output directory. Default: "
            "<thesis-results-dir>/writing"
        ),
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required writing source does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required writing source is empty: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path), low_memory=False)


def aggregate_value(
    df: pd.DataFrame,
    condition: str,
    metric: str,
    field: str = "mean",
) -> float:
    row = df[
        (df["condition"] == condition)
        & (df["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError(
            f"Could not uniquely resolve condition={condition!r}, "
            f"metric={metric!r}."
        )
    return float(row.iloc[0][field])


def long_summary_value(
    df: pd.DataFrame,
    *,
    comparison: str | None = None,
    condition: str | None = None,
    round_number: int | None = None,
    measure: str,
    field: str = "mean",
) -> float:
    mask = df["measure"] == measure

    if comparison is not None:
        mask &= df["comparison"] == comparison
    if condition is not None:
        mask &= df["condition"] == condition
    if round_number is not None:
        mask &= pd.to_numeric(df["round"], errors="raise").astype(int) == round_number

    row = df[mask]
    if len(row) != 1:
        criteria = {
            "comparison": comparison,
            "condition": condition,
            "round": round_number,
            "measure": measure,
        }
        raise ValueError(
            f"Could not uniquely resolve summary row with criteria {criteria}."
        )
    return float(row.iloc[0][field])


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_caption_workbook(
    output_path: Path,
    figure_catalog: pd.DataFrame,
) -> None:
    review_by_id = {
        str(row.artifact_id): str(row.review_status)
        for row in figure_catalog.itertuples()
    }

    text = f"""# Figure caption workbook

These captions are structured drafts for later sentence-level editing. They
preserve the validated selection rules, plotted units, and interpretation
boundaries.

## Figure 1 — Predictive performance across training conditions

**Status:** {review_by_id.get("FIGURE_01", "unknown")}

Selected centralized checkpoints and final Round-50 federated performance
across seeds 42, 7, and 123. Panel (a) reports validation F1 and panel (b)
reports balanced accuracy. Individual points represent seeds; diamonds and
error bars represent the three-seed mean and sample standard deviation.
Centralized checkpoints were selected by validation F1, whereas federated
values correspond to Round 50.

## Figure 2 — Overall accuracy and ranking performance

**Status:** {review_by_id.get("FIGURE_02", "unknown")}

Validation accuracy and ROC AUC for the selected centralized checkpoints and
final IID and non-IID federated models. Individual points represent seeds 42,
7, and 123; diamonds and error bars show the mean and sample standard
deviation. Narrow y-axis ranges are used to display small differences between
otherwise closely grouped results.

## Figure 3 — Peak-versus-final federated performance

**Status:** {review_by_id.get("FIGURE_03", "unknown")}

Panel (a) shows the communication round at which each IID and non-IID run
reached its highest validation F1. Panel (b) shows the improvement of the peak
validation F1 over the corresponding Round-50 F1 in percentage points. Lines
connect IID and non-IID results obtained with the same seed. Dashed reference
lines indicate Round 50 in panel (a) and no peak-versus-final difference in
panel (b).

## Figure 4 — IID FedAvg predictive convergence

**Status:** {review_by_id.get("FIGURE_04", "unknown")}

Validation F1 and ROC AUC trajectories for IID FedAvg across 50 communication
rounds. Thin lines represent individual seeds, the thicker line represents
the three-seed mean, and the shaded region represents mean ± sample standard
deviation. The y-axis limits match Figure 5 to support direct visual
comparison.

## Figure 5 — Non-IID label-skew FedAvg predictive convergence

**Status:** {review_by_id.get("FIGURE_05", "unknown")}

Validation F1 and ROC AUC trajectories for non-IID label-skew FedAvg across
50 communication rounds. Thin lines represent individual seeds, the thicker
line represents the three-seed mean, and the shaded region represents mean ±
sample standard deviation. The y-axis limits match Figure 4.

## Figure 6 — Early adjacent-checkpoint Grad-CAM stability

**Status:** {review_by_id.get("FIGURE_06", "unknown")}

Grad-CAM Pearson correlation and Dice overlap for the aligned Round 0→5 and
Round 5→10 checkpoint transitions. Color identifies IID and non-IID
conditions, marker shape identifies the seed, and diamonds with error bars
represent condition means and sample standard deviations. The Pearson panel
includes a zero-correlation reference line.

## Figure 7 — Grad-CAM similarity to the final explanation

**Status:** {review_by_id.get("FIGURE_07", "unknown")}

Pearson correlation and Dice overlap between each common checkpoint and the
corresponding Round-50 Grad-CAM explanation. Lines show IID and non-IID
condition means together with individual seed trajectories and sample
standard-deviation bands. Round 50 reaches one by definition because it is a
self-comparison.

## Figure 8 — Prediction-target mismatch relative to Round 50

**Status:** {review_by_id.get("FIGURE_08", "unknown")}

Percentage of images whose predicted Grad-CAM target differs from the target
predicted at Round 50. Lines show IID and non-IID means, individual seed
trajectories, and sample-standard-deviation bands at the common checkpoints.

## Figure 9 — Cross-model Grad-CAM spatial similarity

**Status:** {review_by_id.get("FIGURE_09", "unknown")}

Mean Grad-CAM Pearson correlation and Dice overlap for centralized-versus-IID,
centralized-versus-non-IID, and IID-versus-non-IID comparisons. Similarity
metrics are calculated only for computed same-target pairs. Individual points
represent seeds; diamonds and error bars show the three-seed mean and sample
standard deviation.

## Figure 10 — Cross-model prediction agreement and confidence difference

**Status:** {review_by_id.get("FIGURE_10", "unknown")}

Pairwise prediction agreement and mean absolute confidence difference for the
centralized, IID federated, and non-IID federated models. Agreement is
expressed as a percentage of the 200-image XAI subset, while confidence
difference is expressed in percentage points.

## Figure 11 — Representative cross-model Grad-CAM examples

**Status:** {review_by_id.get("FIGURE_11", "not yet visually accepted")}

Representative seed-42 Grad-CAM overlays comparing centralized, IID FedAvg,
and non-IID FedAvg models. Rows illustrate cases of explanation disagreement,
shared incorrect prediction, federated explanation similarity, and
prediction-target disagreement. The final caption should state whether each
heatmap was independently normalized; qualitative spatial agreement does not
establish prediction correctness.
"""
    output_path.write_text(text, encoding="utf-8")


def build_results_outline(
    output_path: Path,
    predictive_summary: pd.DataFrame,
    xai_summary: pd.DataFrame,
    cross_model_summary: pd.DataFrame,
) -> None:
    f1_c = aggregate_value(
        predictive_summary,
        "Centralized selected",
        "f1",
    )
    f1_iid = aggregate_value(
        predictive_summary,
        "IID Round 50",
        "f1",
    )
    f1_non = aggregate_value(
        predictive_summary,
        "Non-IID Round 50",
        "f1",
    )

    auc_c = aggregate_value(
        predictive_summary,
        "Centralized selected",
        "auc",
    )
    auc_iid = aggregate_value(
        predictive_summary,
        "IID Round 50",
        "auc",
    )
    auc_non = aggregate_value(
        predictive_summary,
        "Non-IID Round 50",
        "auc",
    )

    pearson_ci = long_summary_value(
        cross_model_summary,
        comparison="centralized_vs_iid",
        measure="pearson_mean",
    )
    pearson_cn = long_summary_value(
        cross_model_summary,
        comparison="centralized_vs_non_iid_label_skew",
        measure="pearson_mean",
    )
    pearson_in = long_summary_value(
        cross_model_summary,
        comparison="iid_vs_non_iid_label_skew",
        measure="pearson_mean",
    )

    dice_ci = long_summary_value(
        cross_model_summary,
        comparison="centralized_vs_iid",
        measure="dice_mean",
    )
    dice_cn = long_summary_value(
        cross_model_summary,
        comparison="centralized_vs_non_iid_label_skew",
        measure="dice_mean",
    )
    dice_in = long_summary_value(
        cross_model_summary,
        comparison="iid_vs_non_iid_label_skew",
        measure="dice_mean",
    )

    mismatch_iid_5 = long_summary_value(
        xai_summary,
        condition="IID",
        round_number=5,
        measure="final_target_changed_rate_percent",
    )
    mismatch_non_5 = long_summary_value(
        xai_summary,
        condition="Non-IID",
        round_number=5,
        measure="final_target_changed_rate_percent",
    )

    mismatch_iid_30 = long_summary_value(
        xai_summary,
        condition="IID",
        round_number=30,
        measure="final_target_changed_rate_percent",
    )
    mismatch_non_30 = long_summary_value(
        xai_summary,
        condition="Non-IID",
        round_number=30,
        measure="final_target_changed_rate_percent",
    )

    text = f"""# Results chapter evidence outline

## Section 1 — Predictive performance across training conditions

Opening paragraph:
- State that the comparison includes seeds 42, 7, and 123.
- State the checkpoint rules before reporting values.
- Centralized values use the selected validation-F1 checkpoint.
- Federated comparison values use Round 50.

Figure 1 evidence:
- Mean validation F1: centralized {fmt(f1_c)}, IID {fmt(f1_iid)}, non-IID {fmt(f1_non)}.
- Balanced accuracy follows the same broad comparability pattern.
- Appropriate claim: broadly comparable performance.
- Avoid claiming superiority from three descriptive replications.

Figure 2 evidence:
- Mean ROC AUC: centralized {fmt(auc_c)}, IID {fmt(auc_iid)}, non-IID {fmt(auc_non)}.
- Accuracy and ranking performance remain closely grouped.
- Seed-specific ordering varies between conditions.

## Section 2 — Federated convergence and checkpoint selection

Figure 3 evidence:
- Every federated run reaches its highest validation F1 before Round 50.
- Peak-versus-final differences are positive but modest.
- Convergence of training and choice of validation checkpoint are related but
  distinct decisions.

Figures 4 and 5 evidence:
- Both IID and non-IID performance improve rapidly during early rounds.
- Later rounds remain strong but include smaller seed-level fluctuations.
- Non-IID label skew does not produce a persistent late-round collapse in the
  tested configuration.

## Section 3 — Explanation stability during federated training

Figure 6 evidence:
- The Round 0→5 interval has low and variable Pearson and Dice similarity.
- The Round 5→10 interval shows a sharp increase in spatial stability.
- Do not pool later adjacent intervals across unequal checkpoint spacing.

Figure 7 evidence:
- Similarity to Round 50 increases through the shared checkpoints.
- Pearson similarity rises more quickly than thresholded Dice overlap.
- Both IID and non-IID explanations approach their final patterns.

Figure 8 evidence:
- Mean target mismatch at Round 5 is approximately {mismatch_iid_5:.1f}% for IID
  and {mismatch_non_5:.1f}% for non-IID.
- By Round 30, mismatch falls to approximately {mismatch_iid_30:.1f}% and
  {mismatch_non_30:.1f}%, respectively.
- This supports increasing comparability of later Grad-CAM maps.

## Section 4 — Cross-model prediction and explanation comparison

Figure 9 evidence:
- Mean Pearson: centralized–IID {fmt(pearson_ci)}, centralized–non-IID
  {fmt(pearson_cn)}, IID–non-IID {fmt(pearson_in)}.
- Mean Dice: centralized–IID {fmt(dice_ci)}, centralized–non-IID
  {fmt(dice_cn)}, IID–non-IID {fmt(dice_in)}.
- The federated models are consistently more similar to each other than either
  is to the centralized model.
- Do not claim that centralized is systematically closer to IID or non-IID.

Figure 10 evidence:
- Pairwise prediction agreement remains high.
- IID–non-IID confidence differences are smaller than centralized–federated
  differences.
- High target agreement does not imply high spatial explanation agreement.

## Section 5 — Representative qualitative examples

Figure 11 evidence:
- Use the figure only after its final row labels and metadata hierarchy are
  visually accepted.
- Discuss each row by category rather than describing every color region.
- Explicitly separate correctness, prediction agreement, and explanation
  agreement.
- State the normalization boundary for Grad-CAM intensity.
"""
    output_path.write_text(text, encoding="utf-8")


def build_discussion_outline(output_path: Path) -> None:
    text = """# Discussion chapter evidence outline

## Theme 1 — Federated learning preserved predictive viability

Synthesize Figures 1, 2, 4, and 5.

Core interpretation:
- Five-client FedAvg preserved strong predictive performance under both IID
  and the tested label-skew non-IID partition.
- The correct claim is comparability, not statistical equivalence or
  superiority.
- Non-IID heterogeneity altered local trajectories but did not cause a
  persistent final collapse.

Connection to methodology:
- The shared canonical split and validation population reduce dataset-change
  confounding.
- The three-seed design checks robustness but remains too small for strong
  population-level inference.

## Theme 2 — Final-round completion is not the same as optimal checkpoint selection

Synthesize Figure 3 with Figures 4 and 5.

Core interpretation:
- The strongest validation-F1 checkpoints consistently occurred before the
  predetermined final round.
- Round 50 remained competitive, but validation-based selection retained a
  measurable advantage.
- Training convergence, operational stopping, and checkpoint selection should
  be discussed as separate decisions.

## Theme 3 — Predictive convergence preceded full explanatory stabilization

Synthesize Figures 4–8.

Core interpretation:
- Predictive metrics reached a strong region early.
- Grad-CAM similarity changed strongly during the earliest interval and
  continued approaching the final explanation afterward.
- Predictive convergence and explanation convergence were related but not
  identical processes.

Methodological importance:
- Explanation comparisons must account for target changes.
- High heatmap similarity calculated on a shrinking or selected subset can be
  misleading unless target-change counts are reported alongside it.

## Theme 4 — Federated training produced a shared explanation regime

Synthesize Figures 9 and 10.

Core interpretation:
- IID and non-IID federated explanations were more similar to each other than
  either was to centralized training.
- This suggests that the federated optimization process itself may shape the
  learned spatial explanation pattern, beyond the exact client partition.
- This is an interpretation supported by the controlled experiment, not a
  claim of universal causal mechanism.

## Theme 5 — Similar predictions do not guarantee similar explanations

Synthesize Figures 9–11.

Core interpretation:
- Prediction agreement remained high across model pairs.
- Centralized-versus-federated explanation similarity remained only moderate.
- The qualitative examples illustrate both agreement and divergence cases.
- Shared explanation patterns can accompany shared incorrect predictions.

## Final synthesis toward the research questions

The thesis can conclude that:
- Federated learning remained predictively viable in the tested medical-image
  classification setting.
- Label-skew non-IID partitioning did not cause persistent predictive or
  explanatory collapse.
- Grad-CAM explanations were unstable early and progressively stabilized.
- Federated models converged to more similar spatial explanations to each
  other than to the centralized model.
- Prediction similarity alone was insufficient to establish explanation
  similarity or correctness.
"""
    output_path.write_text(text, encoding="utf-8")


def build_rq_matrix(output_path: Path) -> None:
    rows = [
        {
            "research_question": "RQ1",
            "question": (
                "How does federated training affect predictive performance "
                "relative to centralized training?"
            ),
            "primary_artifacts": "Figures 1–5; Tables 1–4",
            "supported_answer": (
                "IID and label-skew non-IID FedAvg achieved predictive "
                "performance broadly comparable to the centralized baseline "
                "across the three tested seeds."
            ),
            "strength_of_claim": "Descriptive controlled three-seed evidence",
            "key_boundary": (
                "Do not claim statistical equivalence or universal superiority."
            ),
        },
        {
            "research_question": "RQ2",
            "question": (
                "How stable are Grad-CAM explanations during federated "
                "training?"
            ),
            "primary_artifacts": "Figures 6–8; Tables 5–6",
            "supported_answer": (
                "Explanations changed strongly during early training, then "
                "became increasingly stable and consistent with the final "
                "model target and explanation."
            ),
            "strength_of_claim": "Aligned checkpoint evidence across three seeds",
            "key_boundary": (
                "Later adjacent intervals cannot be pooled when checkpoint "
                "spacing differs."
            ),
        },
        {
            "research_question": "RQ3",
            "question": (
                "How similar are centralized, IID federated, and non-IID "
                "federated Grad-CAM explanations?"
            ),
            "primary_artifacts": "Figures 9–11; Tables 7–8",
            "supported_answer": (
                "The two federated models produced explanations that were "
                "consistently more similar to each other than either was to "
                "the centralized model."
            ),
            "strength_of_claim": "Consistent three-seed pairwise evidence",
            "key_boundary": (
                "Centralized was not consistently closer to either federated "
                "condition across seeds."
            ),
        },
        {
            "research_question": "RQ4",
            "question": (
                "Does prediction agreement imply explanation agreement or "
                "correctness?"
            ),
            "primary_artifacts": "Figures 9–11",
            "supported_answer": (
                "No. Model pairs often agreed on the predicted target while "
                "showing substantially different spatial explanations, and "
                "shared explanations could accompany shared incorrect "
                "predictions."
            ),
            "strength_of_claim": "Quantitative and representative qualitative evidence",
            "key_boundary": (
                "Qualitative examples illustrate mechanisms but do not estimate "
                "population prevalence."
            ),
        },
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_limitations_register(output_path: Path) -> None:
    rows = [
        {
            "limitation_id": "L01",
            "limitation": "Only three random seeds were evaluated.",
            "affected_claims": "All aggregate comparisons",
            "required_wording": (
                "Describe means and standard deviations as descriptive rather "
                "than inferential evidence."
            ),
        },
        {
            "limitation_id": "L02",
            "limitation": (
                "The experiment uses one dataset, one binary task, one model "
                "architecture, and five clients."
            ),
            "affected_claims": "Generalizability",
            "required_wording": (
                "Restrict conclusions to the tested VinDr-CXR binary "
                "classification configuration."
            ),
        },
        {
            "limitation_id": "L03",
            "limitation": (
                "The non-IID evaluation uses one controlled label-skew setup."
            ),
            "affected_claims": "Client heterogeneity",
            "required_wording": (
                "Do not generalize to all forms or strengths of non-IID data."
            ),
        },
        {
            "limitation_id": "L04",
            "limitation": (
                "Centralized and federated comparison checkpoints follow "
                "different selection rules."
            ),
            "affected_claims": "Figures 1–2",
            "required_wording": (
                "State that centralized uses selected validation-F1 checkpoints "
                "while federated figures use Round 50."
            ),
        },
        {
            "limitation_id": "L05",
            "limitation": (
                "Grad-CAM is one post-hoc explanation method and does not prove "
                "causal model reasoning."
            ),
            "affected_claims": "All XAI interpretations",
            "required_wording": (
                "Use explanation similarity and spatial focus language rather "
                "than causal reasoning claims."
            ),
        },
        {
            "limitation_id": "L06",
            "limitation": (
                "Heatmap comparisons are conditional on compatible predicted "
                "targets."
            ),
            "affected_claims": "Figures 6–9",
            "required_wording": (
                "Report target-change counts or mismatch rates alongside "
                "similarity metrics."
            ),
        },
        {
            "limitation_id": "L07",
            "limitation": (
                "Qualitative cases are selected examples rather than a random "
                "population sample."
            ),
            "affected_claims": "Figure 11",
            "required_wording": (
                "Use the figure to illustrate patterns, not estimate their "
                "frequency."
            ),
        },
        {
            "limitation_id": "L08",
            "limitation": (
                "Independently normalized Grad-CAM overlays may not support "
                "absolute activation-strength comparisons."
            ),
            "affected_claims": "Figure 11",
            "required_wording": (
                "State the normalization procedure and restrict interpretation "
                "to spatial location and shape when appropriate."
            ),
        },
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_package_readme(output_path: Path) -> None:
    text = """# Thesis writing package

This directory contains structured aids for writing the Results, Discussion,
captions, research-question answers, and limitations.

Files:

- `figure_caption_workbook.md`
- `results_chapter_evidence_outline.md`
- `discussion_chapter_evidence_outline.md`
- `research_question_answer_matrix.csv`
- `limitations_register.csv`

The files are evidence-organizing drafts. Final prose should be edited after
the full figure order and Figure 11 presentation are finalized.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    thesis_results_dir = (
        args.thesis_results_dir.expanduser().resolve()
        if args.thesis_results_dir
        else project_root / "outputs" / "thesis_results"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else thesis_results_dir / "writing"
    )

    predictive_summary = read_csv(
        thesis_results_dir
        / "predictive"
        / "table04_predictive_comparison_three_seed_summary.csv"
    )
    xai_summary = read_csv(
        thesis_results_dir
        / "xai_stability"
        / "table06_xai_stability_three_seed_summary.csv"
    )
    cross_model_summary = read_csv(
        thesis_results_dir
        / "cross_model_xai"
        / "table08_cross_model_similarity_three_seed_summary.csv"
    )
    figure_catalog = read_csv(
        thesis_results_dir
        / "integrated"
        / "table12_final_figure_catalog.csv"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    build_caption_workbook(
        output_dir / "figure_caption_workbook.md",
        figure_catalog,
    )
    build_results_outline(
        output_dir / "results_chapter_evidence_outline.md",
        predictive_summary,
        xai_summary,
        cross_model_summary,
    )
    build_discussion_outline(
        output_dir / "discussion_chapter_evidence_outline.md"
    )
    build_rq_matrix(
        output_dir / "research_question_answer_matrix.csv"
    )
    build_limitations_register(
        output_dir / "limitations_register.csv"
    )
    write_package_readme(
        output_dir / "README.md"
    )

    print("Thesis writing package created:")
    print(output_dir)
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
