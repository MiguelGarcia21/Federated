# Cross-model XAI artifact catalog

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
