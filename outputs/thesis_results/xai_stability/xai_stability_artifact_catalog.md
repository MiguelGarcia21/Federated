# XAI-stability artifact catalog

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
