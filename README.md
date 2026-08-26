# [FINAL THESIS TITLE]

**Bachelor Thesis — Bachelor of Science (B.Sc.) in Software Engineering**

**Author:** Miguel Angel Garcia Higuita  
**University:** University of Europe for Applied Science
**Year:** 2026  
**Status:** Final thesis preparation and repository freeze in progress

---

## Overview

This repository contains the implementation, experimental configuration, analysis workflow, and thesis-result artifacts for a Bachelor of Science thesis in Software Engineering investigating **Federated Learning (FL)**, **Explainable Artificial Intelligence (XAI)**, and **medical image classification**.

The study examines whether models trained centrally and through federated learning exhibit similar predictive behaviour and similar **Grad-CAM spatial attribution patterns**. Particular attention is given to the effect of client-data heterogeneity and to the temporal evolution of model explanations during federated optimization.

The experimental application is binary **normal-versus-abnormal chest-radiograph classification** using the **VinDr-CXR** dataset.

> **Thesis title:** [FINAL THESIS TITLE]

---

## Research Focus

The thesis studies four closely related aspects of model behaviour:

- predictive performance during federated training;
- the evolution and stability of Grad-CAM spatial attribution maps across communication rounds;
- differences between IID and controlled label-skew non-IID federated training;
- the relationship between prediction agreement and explanation-map agreement across centralized and federated models.

Predictive performance and explanation behaviour are treated as related but distinct experimental quantities. Grad-CAM similarity is used to characterize **spatial attribution agreement** and is not interpreted as evidence of identical causal reasoning or clinical correctness.

---

## Experimental Design

The thesis contains five experiment families.

| Experiment | Description |
|---|---|
| **Experiment 1** | Centralized ResNet-18 baseline |
| **Experiment 2** | Five-client FedAvg with an IID client partition |
| **Experiment 3** | Five-client FedAvg with controlled non-IID label skew |
| **Experiment 4** | Round-wise Grad-CAM stability analysis across federated checkpoints |
| **Experiment 5** | Cross-model Grad-CAM comparison between centralized, IID, and non-IID models |

The principal quantitative analyses are repeated using three random seeds:

```text
42
7
123
```

These runs are treated as replications of the same experimental design rather than as additional experiments.

---

## Dataset and Task

The experiments use the **VinDr-CXR** chest-radiograph dataset.

The annotation information is transformed into a binary classification task:

- **Class 0 — Normal:** all annotation `class_id` values associated with the image are `14`.
- **Class 1 — Abnormal:** at least one annotation `class_id` associated with the image differs from `14`.

The canonical experimental population is:

| Population | Images |
|---|---:|
| Training | 12,000 |
| Validation | 3,000 |
| Fixed XAI subset | 200 |

The same canonical training population, validation population, and XAI subset are retained across the principal experimental conditions and multi-seed replications.

The raw medical-image dataset is not intended to be redistributed through this repository. Dataset access and usage remain subject to the terms of the original VinDr-CXR source.

---

## Core Experimental Configuration

| Component | Configuration |
|---|---|
| Architecture | ResNet-18 |
| Classification task | Binary normal vs. abnormal |
| Input resolution | 224 × 224 |
| Pretrained weights | Disabled |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Batch size | `16` |
| Centralized training | 10 epochs |
| Federated clients | 5 |
| Federated aggregation | FedAvg |
| Federated training | 50 communication rounds |
| Local training | 1 epoch per client per round |
| XAI method | Grad-CAM |
| Grad-CAM target layer | `model.layer4[-1]` |
| XAI subset | 200 validation images |
| Seeds | 42, 7, 123 |

The centralized comparison model is selected using the highest validation F1 observed during its training run. Federated endpoint comparisons use the terminal **Round-50 global model**.

---

## Predictive Evaluation

The primary predictive metrics used in the thesis are:

- F1 score;
- ROC-AUC;
- balanced accuracy;
- accuracy.

The evaluation pipeline additionally records supporting metrics such as:

- precision;
- recall;
- specificity;
- average precision;
- negative predictive value;
- false-positive and false-negative rates;
- Matthews correlation coefficient;
- Brier score;
- confusion-matrix counts.

Federated predictive evaluation is performed on the same complete canonical validation population after each communication round.

---

## Grad-CAM Analysis

Grad-CAM is generated using the final residual block of the fourth ResNet stage:

```text
model.layer4[-1]
```

The analysis uses the model-predicted class as the explanation target.

Numerical Grad-CAM maps are independently normalized to `[0, 1]` and compared using complementary similarity measures:

- Pearson correlation;
- cosine similarity;
- Dice coefficient;
- Intersection over Union (IoU);
- mean absolute error (MAE);
- mean squared error (MSE).

Dice and IoU use a threshold of `0.5`.

Spatial similarity is calculated only when both compared model states predict the same target class for the corresponding image. Prediction-target changes are recorded separately instead of being interpreted as ordinary low-similarity Grad-CAM comparisons.

---

## Repository Structure

The project is organized into modular implementation, configuration, data-definition, analysis, and output directories.

```text
thesis-fl-xai/
├── configs/
│
├── data/
│   ├── interim/
│   └── processed/
│
├── src/
│   ├── analysis/
│   ├── data/
│   ├── experiments/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── xai/
│
└── outputs/
    ├── checkpoints/
    ├── figures/
    ├── heatmaps/
    ├── metrics/
    └── thesis_results/
        ├── predictive/
        ├── xai_stability/
        ├── cross_model_xai/
        ├── integrated/
        └── writing/
```

### `configs/`

Experiment-specific configuration files and configuration snapshots.

### `data/interim/`

Canonical train/validation definitions, the fixed XAI subset, and realized federated client partitions.

### `src/experiments/`

Entry points for the centralized, federated, and Grad-CAM experiments.

### `src/analysis/`

Post-training analysis used to generate thesis-ready predictive, XAI-stability, cross-model, integrated, and writing-support artifacts.

### `outputs/metrics/`

Raw numerical outputs generated by the experiments.

### `outputs/thesis_results/`

Curated tables, figures, source CSVs, manifests, qualitative XAI artifacts, and writing-support outputs used to construct and verify the thesis results.

---

## Reproducibility

The project uses explicit random seeds for Python, NumPy, PyTorch, and CUDA operations.

The experimental runs are **seed-controlled**, but the GPU configuration is not claimed to be bitwise deterministic. The reported multi-seed results should therefore be interpreted as repeated stochastic realizations of the same experimental design.

The main training environment uses:

- Python 3.12.3;
- PyTorch and Torchvision;
- Flower;
- Ray;
- CUDA-enabled GPU execution.

The final thesis-result generation is performed from saved numerical artifacts rather than by manually transcribing values from training logs.

---

## Artifact Provenance and Integrity

The thesis-result workflow includes explicit provenance tracking.

The master artifact manifest records information including:

- relative artifact path;
- filename and file type;
- file size;
- SHA-256 digest;
- Git branch;
- Git commit associated with the generated artifact package.

The repository also contains a provenance snapshot describing the state from which the integrated thesis-result package was generated.

These files are intended to make the reported tables and figures traceable to their underlying numerical artifacts and repository state.

Before the final university submission, the provenance records should be updated and checked against the repository state used for the submitted thesis.

---

## Main Findings

The thesis results support four principal observations:

1. Centralized, IID federated, and controlled label-skew non-IID federated training achieved closely grouped predictive performance under the evaluated configuration.
2. Grad-CAM spatial attribution patterns changed strongly during the earliest federated rounds and became substantially more stable later in training.
3. Predictive development and Grad-CAM explanation stabilization did not behave as a single synchronized process.
4. High prediction agreement between models did not necessarily imply high Grad-CAM spatial attribution agreement. Across the completed replications, the IID and non-IID federated models exhibited stronger cross-model Grad-CAM similarity to each other than either federated model exhibited relative to the selected centralized model.

These conclusions are specific to the evaluated dataset, architecture, federated configuration, heterogeneity mechanism, and Grad-CAM implementation.

---

## Reproducing the Experiments

The repository separates experiment execution from post-training thesis analysis.

A typical workflow is:

1. Prepare the VinDr-CXR data according to the project preprocessing pipeline.
2. Create or verify the canonical train/validation split and fixed XAI subset.
3. Run the centralized baseline.
4. Run IID FedAvg.
5. Run non-IID label-skew FedAvg.
6. Generate round-wise Grad-CAM outputs.
7. Generate cross-model Grad-CAM outputs.
8. Run the analysis modules under `src/analysis/`.
9. Validate the generated artifacts and provenance records.

Exact experiment settings should be taken from the corresponding configuration snapshots and saved artifact metadata rather than inferred from filenames alone.

---

## Academic Use

This repository accompanies a Bachelor of Science thesis in Software Engineering and is maintained primarily for:

- academic review;
- reproducibility;
- traceability of reported results;
- preservation of the experimental implementation and thesis evidence.

If material from this repository is reused, please cite the final thesis once its bibliographic information is available.

---

## Citation

The final thesis citation will be added after the thesis title and university submission metadata are finalized.

```text
Miguel Angel Garcia Higuita.
"[FINAL THESIS TITLE]."
Bachelor Thesis, Bachelor of Science in Software Engineering,
University of Europe for Applied Sciences, 2026.
```

---

## Author

**Miguel Angel Garcia Higuita**  
Bachelor of Science in Software Engineering  
Bachelor Thesis, 2026