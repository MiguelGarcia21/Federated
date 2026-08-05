# Figure caption workbook

These captions are structured drafts for later sentence-level editing. They
preserve the validated selection rules, plotted units, and interpretation
boundaries.

## Figure 1 — Predictive performance across training conditions

**Status:** visually_accepted

Selected centralized checkpoints and final Round-50 federated performance
across seeds 42, 7, and 123. Panel (a) reports validation F1 and panel (b)
reports balanced accuracy. Individual points represent seeds; diamonds and
error bars represent the three-seed mean and sample standard deviation.
Centralized checkpoints were selected by validation F1, whereas federated
values correspond to Round 50.

## Figure 2 — Overall accuracy and ranking performance

**Status:** visually_accepted

Validation accuracy and ROC AUC for the selected centralized checkpoints and
final IID and non-IID federated models. Individual points represent seeds 42,
7, and 123; diamonds and error bars show the mean and sample standard
deviation. Narrow y-axis ranges are used to display small differences between
otherwise closely grouped results.

## Figure 3 — Peak-versus-final federated performance

**Status:** visually_accepted

Panel (a) shows the communication round at which each IID and non-IID run
reached its highest validation F1. Panel (b) shows the improvement of the peak
validation F1 over the corresponding Round-50 F1 in percentage points. Lines
connect IID and non-IID results obtained with the same seed. Dashed reference
lines indicate Round 50 in panel (a) and no peak-versus-final difference in
panel (b).

## Figure 4 — IID FedAvg predictive convergence

**Status:** visually_accepted

Validation F1 and ROC AUC trajectories for IID FedAvg across 50 communication
rounds. Thin lines represent individual seeds, the thicker line represents
the three-seed mean, and the shaded region represents mean ± sample standard
deviation. The y-axis limits match Figure 5 to support direct visual
comparison.

## Figure 5 — Non-IID label-skew FedAvg predictive convergence

**Status:** visually_accepted

Validation F1 and ROC AUC trajectories for non-IID label-skew FedAvg across
50 communication rounds. Thin lines represent individual seeds, the thicker
line represents the three-seed mean, and the shaded region represents mean ±
sample standard deviation. The y-axis limits match Figure 4.

## Figure 6 — Early adjacent-checkpoint Grad-CAM stability

**Status:** visually_accepted

Grad-CAM Pearson correlation and Dice overlap for the aligned Round 0→5 and
Round 5→10 checkpoint transitions. Color identifies IID and non-IID
conditions, marker shape identifies the seed, and diamonds with error bars
represent condition means and sample standard deviations. The Pearson panel
includes a zero-correlation reference line.

## Figure 7 — Grad-CAM similarity to the final explanation

**Status:** visually_accepted

Pearson correlation and Dice overlap between each common checkpoint and the
corresponding Round-50 Grad-CAM explanation. Lines show IID and non-IID
condition means together with individual seed trajectories and sample
standard-deviation bands. Round 50 reaches one by definition because it is a
self-comparison.

## Figure 8 — Prediction-target mismatch relative to Round 50

**Status:** visually_accepted

Percentage of images whose predicted Grad-CAM target differs from the target
predicted at Round 50. Lines show IID and non-IID means, individual seed
trajectories, and sample-standard-deviation bands at the common checkpoints.

## Figure 9 — Cross-model Grad-CAM spatial similarity

**Status:** visually_accepted

Mean Grad-CAM Pearson correlation and Dice overlap for centralized-versus-IID,
centralized-versus-non-IID, and IID-versus-non-IID comparisons. Similarity
metrics are calculated only for computed same-target pairs. Individual points
represent seeds; diamonds and error bars show the three-seed mean and sample
standard deviation.

## Figure 10 — Cross-model prediction agreement and confidence difference

**Status:** visually_accepted

Pairwise prediction agreement and mean absolute confidence difference for the
centralized, IID federated, and non-IID federated models. Agreement is
expressed as a percentage of the 200-image XAI subset, while confidence
difference is expressed in percentage points.

## Figure 11 — Representative cross-model Grad-CAM examples

**Status:** visually_accepted

Representative seed-42 Grad-CAM overlays comparing centralized, IID FedAvg,
and non-IID FedAvg models. Rows illustrate cases of explanation disagreement,
shared incorrect prediction, federated explanation similarity, and
prediction-target disagreement. The final caption should state whether each
heatmap was independently normalized; qualitative spatial agreement does not
establish prediction correctness.
