# Predictive-results artifact catalog

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
**Visual structure:** Individual seed points are horizontally offset within each condition; diamonds and error bars show the across-seed mean and sample standard deviation. No lines connect the categorical training conditions.  
**Caption:** Selected centralized checkpoints and final Round-50 federated performance across seeds 42, 7, and 123. Centralized checkpoints were selected using validation F1, whereas federated values correspond to Round 50.

## FIGURE_02

**Title:** Accuracy and ROC AUC across centralized and federated training  
**Files:** `fig02_final_accuracy_auc_three_seeds.png` and `.pdf`  
**Panels:** Accuracy; ROC AUC.  
**Primary conclusion:** The absence of a major performance collapse was also visible in overall accuracy and ranking ability.  
**Suggested placement:** After the interpretation of Figure 1, with one introductory and two follow-up paragraphs.  
**Visual structure:** Individual seed points are horizontally offset within each condition; diamonds and error bars show the mean and sample standard deviation. No lines connect the categorical training conditions.  
**Caption:** Accuracy and ROC AUC for the selected centralized checkpoint and final federated models across three seeds.

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
**Visual structure:** Three thin seed trajectories, one thicker three-seed mean, and a sample-standard-deviation band in each panel. The F1 and ROC-AUC limits are fixed across Figures 4 and 5 for direct visual comparison. The F1 and ROC-AUC limits are fixed across Figures 4 and 5 for direct visual comparison.

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
