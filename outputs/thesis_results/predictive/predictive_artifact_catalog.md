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
**Panels:** Best-F1 communication round; best-F1 minus Round-50 F1.  
**Primary conclusion:** The final round was not consistently the best validation-F1 checkpoint.  
**Suggested placement:** Separate from general performance-comparison figures because it supports a model-selection conclusion.  
**Caption:** Communication round at which validation F1 was highest and the corresponding difference between best-round and final Round-50 F1. Positive differences indicate that the best validation-F1 checkpoint outperformed the final model.

## SUMMARY_01

**Title:** Predictive performance across three seeds  
**Filename:** `summary01_predictive_performance_results.md`  
**Purpose:** Provides a Results-ready paragraph sequence, Discussion links, and caveats based on the generated tables.
