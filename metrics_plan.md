# Metrics Plan

**Thesis title:** Evaluating the Evolution of Explainability in Federated Learning for Medical Image Classification: A Comparative Study of Faithfulness, Stability, and Privacy

**Purpose of this file:**  
This document defines the evaluation metrics that will be used throughout the thesis. It should be treated as the reference document for the methodology, experiment implementation, and results reporting.

---

## 1. Research Questions Covered by the Metrics

### Primary Research Question

**RQ1:** How do explanation faithfulness and stability evolve during federated learning for medical image classification, and how does their convergence relate to model performance?

### Secondary Research Question 1

**RQ2:** How do Grad-CAM and LRP compare in terms of faithfulness, stability, and localization quality within federated medical image classification?

### Optional Research Question 2

**RQ3:** How does Differential Privacy influence explanation faithfulness, stability, and predictive performance in federated medical image classification?

---

## 2. Evaluation Overview

The thesis will evaluate four dimensions:

1. **Predictive performance**
2. **Explainability faithfulness**
3. **Explanation stability over federated learning rounds**
4. **Localization quality against medical annotations**

If Differential Privacy is included, a fifth dimension will be added:

5. **Privacy-performance-explainability trade-off**

---

## 3. Predictive Performance Metrics

These metrics evaluate the classification model itself.

### 3.1 Accuracy

**Purpose:** Measures the proportion of correctly classified samples.

**Use:**  
Useful for binary classification, but should not be used alone if the dataset is imbalanced.

**Formula:**

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Report for:**

- Centralized baseline
- Federated IID setting
- Federated non-IID setting
- Client-count experiments
- Optional DP experiments

---

### 3.2 Precision

**Purpose:** Measures how many predicted positive cases are actually positive.

**Formula:**

```text
Precision = TP / (TP + FP)
```

**Why relevant:**  
Important in medical diagnosis because false positives can lead to unnecessary follow-up procedures.

---

### 3.3 Recall / Sensitivity

**Purpose:** Measures how many true positive cases are correctly detected.

**Formula:**

```text
Recall = TP / (TP + FN)
```

**Why relevant:**  
Especially important in medical imaging because missed positive cases can be clinically serious.

---

### 3.4 F1-score

**Purpose:** Harmonic mean of precision and recall.

**Formula:**

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Why relevant:**  
Useful when the dataset is imbalanced.

---

### 3.5 AUC-ROC

**Purpose:** Measures the model's ability to distinguish between positive and negative cases across thresholds.

**Why relevant:**  
AUC is widely used in medical imaging classification and is better than accuracy when the dataset is imbalanced.

**Report as:**

```text
AUC ± standard deviation
```

across multiple seeds.

---

### 3.6 Loss

**Purpose:** Tracks model optimization progress.

**Use in thesis:**  
Loss will be logged per communication round and compared with explanation stability.

**Important for RQ1:**  
Loss convergence can be compared with explanation convergence.

---

## 4. Federated Learning Metrics

These metrics evaluate the training process in the federated setting.

### 4.1 Global Model Performance per Round

For each communication round, log:

```text
round,
global_loss,
global_accuracy,
global_auc,
global_f1
```

**Purpose:**  
Used to compare model convergence with explanation convergence.

---

### 4.2 Number of Communication Rounds to Convergence

**Definition:**  
The round at which the model performance reaches a stable plateau.

A practical definition:

```text
Performance is considered converged if AUC changes by less than 1% over 5 consecutive evaluation rounds.
```

This threshold can be adjusted after pilot experiments.

---

### 4.3 IID vs Non-IID Performance Gap

**Definition:**

```text
Gap = Metric_IID - Metric_nonIID
```

**Purpose:**  
Measures the effect of client heterogeneity.

**Report for:**

- AUC
- F1
- explanation faithfulness
- explanation stability

---

### 4.4 Client Count Effect

Compare:

```text
3 clients
5 clients
10 clients
```

**Purpose:**  
Determines whether explanation stability changes when the number of simulated clients increases.

---

## 5. Explainability Methods

The thesis will compare:

1. **Grad-CAM**
2. **Layer-wise Relevance Propagation (LRP)**

For each method, explanations should be generated on a fixed validation subset to allow fair comparison across rounds.

Recommended validation subset:

```text
200–500 fixed validation images
```

The same image IDs must be used across:

- centralized model
- FL IID model
- FL non-IID model
- all communication rounds
- Grad-CAM
- LRP

---

## 6. Explanation Faithfulness Metrics

Faithfulness measures whether an explanation reflects the actual decision-making behavior of the model.

---

### 6.1 Infidelity

**Purpose:**  
Measures how much the model output changes when input perturbations are applied, compared to what the explanation predicts.

**Interpretation:**

```text
Lower infidelity = more faithful explanation
Higher infidelity = less faithful explanation
```

**Use for:**

- Grad-CAM
- LRP
- centralized vs federated comparison
- round-wise analysis
- optional DP analysis

---

### 6.2 Sensitivity

**Purpose:**  
Measures how much the explanation changes when small perturbations are applied to the input.

**Interpretation:**

```text
Lower sensitivity = more stable/robust explanation
Higher sensitivity = less stable explanation
```

**Note:**  
Sensitivity can also be interpreted as a robustness metric, so it should be discussed carefully.

---

### 6.3 AOPC / Deletion Score

**AOPC:** Area Over the Perturbation Curve.

**Purpose:**  
Measures whether removing the most important pixels according to the explanation significantly reduces model confidence.

**Interpretation:**

```text
A stronger confidence drop after removing highlighted pixels suggests higher faithfulness.
```

**Use:**  
This is useful for comparing Grad-CAM and LRP.

---

### 6.4 Insertion Score

**Purpose:**  
Measures whether adding the most important pixels first helps the model recover its prediction.

**Interpretation:**

```text
Higher insertion performance = explanation highlights useful regions.
```

**Optional:**  
Use only if implementation time allows.

---

## 7. Explanation Stability Metrics

Stability is the core of the thesis.

The goal is to measure how explanations change during federated learning.

---

### 7.1 Consecutive-Round Heatmap Similarity

**Definition:**  
Similarity between the explanation at round `t` and round `t+1`.

For image `i`:

```text
Similarity_t = sim(H_i,t, H_i,t+1)
```

Where:

```text
H_i,t = heatmap for image i at round t
```

Possible similarity functions:

- Pearson correlation
- Spearman correlation
- Cosine similarity
- Structural Similarity Index Measure (SSIM)

**Recommended default:**

```text
Pearson correlation + SSIM
```

**Interpretation:**

```text
Higher similarity = more stable explanations across rounds.
```

---

### 7.2 Similarity to Final-Round Explanation

**Definition:**  
Similarity between the explanation at round `t` and the explanation at the final round.

```text
Similarity_to_final_t = sim(H_i,t, H_i,final)
```

**Purpose:**  
Measures how quickly explanations approach their final stable form.

**Important for RQ1:**  
This helps determine whether explanations converge before, after, or alongside model accuracy.

---

### 7.3 Explanation Convergence Round

**Definition:**  
The communication round at which explanation stability reaches a plateau.

A practical definition:

```text
Explanation convergence occurs when heatmap similarity to the final round changes by less than 1% over 5 consecutive evaluated checkpoints.
```

Because explanations may be evaluated every 5 or 10 communication rounds, convergence should be computed over evaluated checkpoints.

---

### 7.4 Accuracy-Convergence vs Explanation-Convergence Gap

**Definition:**

```text
Convergence Gap = Explanation Convergence Round - Accuracy Convergence Round
```

**Interpretation:**

```text
Gap > 0:
Explanations converge after model performance.

Gap = 0:
Explanations and performance converge together.

Gap < 0:
Explanations converge before model performance.
```

This is one of the most important measurements in the thesis.

---

### 7.5 Explanation Stability Across Clients

Optional, if local client models are saved.

**Definition:**  
Compare explanations generated by local client models before aggregation.

**Purpose:**  
Measures whether different clients learn different visual decision patterns.

**Use only if time allows.**

---

## 8. Localization Metrics

Localization metrics compare explanations with available medical annotations.

This is especially relevant for VinDr-CXR because it includes radiologist bounding boxes.

---

### 8.1 Pointing Game

**Definition:**  
Checks whether the maximum heatmap activation falls inside a ground-truth bounding box.

```text
Pointing Game = number of hits / number of evaluated images
```

**Interpretation:**

```text
Higher score = better localization.
```

**Use for:**

- Grad-CAM
- LRP
- centralized model
- federated model
- selected rounds

---

### 8.2 Bounding Box Overlap

**Purpose:**  
Measures how much the explanation overlaps with annotated abnormal regions.

Possible metric:

```text
Overlap = highlighted area inside bounding box / total highlighted area
```

**Practical implementation:**

1. Normalize heatmap to [0, 1].
2. Threshold top-k% of heatmap pixels.
3. Compute overlap with bounding box region.

Recommended thresholds:

```text
Top 5%
Top 10%
Top 20%
```

---

### 8.3 IoU-like Localization Score

If heatmaps are converted into binary masks, compute:

```text
IoU = intersection(heatmap_mask, bbox_mask) / union(heatmap_mask, bbox_mask)
```

**Warning:**  
IoU may be harsh because heatmaps are soft explanations and bounding boxes are approximate annotations.

Use this as a supplementary metric, not the only localization metric.

---

## 9. Differential Privacy Metrics

This section is optional and should only be used if DP-SGD experiments are included.

---

### 9.1 Privacy Budget ε

**Purpose:**  
Measures the strength of Differential Privacy.

Example settings:

```text
No DP
ε ≈ 10
ε ≈ 5
ε ≈ 2
```

**Interpretation:**

```text
Lower ε = stronger privacy
Higher ε = weaker privacy
No DP = baseline
```

---

### 9.2 Privacy-Performance Trade-off

For each privacy setting, report:

```text
AUC
F1
Accuracy
Loss
```

Then compute degradation compared with no DP:

```text
Performance Degradation = Metric_noDP - Metric_DP
```

---

### 9.3 Privacy-Explainability Trade-off

For each privacy setting, report:

```text
Infidelity
Sensitivity
AOPC
Heatmap stability
Localization score
```

Then compute explanation degradation compared with no DP:

```text
Explanation Degradation = XAI_metric_noDP - XAI_metric_DP
```

The exact direction of the subtraction depends on whether higher or lower is better for the metric.

---

## 10. Experimental Conditions

The recommended experimental conditions are:

---

### Condition A: Centralized Baseline

```text
Dataset: VinDr-CXR
Training: centralized
Model: ResNet-18 / DenseNet-121 / EfficientNet-B0
XAI: Grad-CAM, LRP
```

---

### Condition B: Federated IID

```text
Dataset: VinDr-CXR
Training: FedAvg
Clients: 5
Split: IID
XAI: Grad-CAM, LRP
```

---

### Condition C: Federated Non-IID

```text
Dataset: VinDr-CXR
Training: FedAvg
Clients: 5
Split: non-IID
XAI: Grad-CAM, LRP
```

---

### Condition D: Client Count Analysis

```text
Dataset: VinDr-CXR
Training: FedAvg
Clients: 3, 5, 10
Split: non-IID
XAI: Grad-CAM, LRP
```

---

### Condition E: Secondary Dataset Validation

```text
Dataset: CheXpert subset
Training: centralized + 5-client FedAvg
Split: non-IID
XAI: Grad-CAM, LRP
```

---

### Condition F: Optional Differential Privacy

```text
Dataset: VinDr-CXR
Training: FedAvg + DP-SGD
Clients: 5
Split: non-IID
Privacy: ε ≈ 10, ε ≈ 5, ε ≈ 2
XAI: Grad-CAM, LRP
```

---

## 11. Recommended Checkpoints for XAI Evaluation

Do not generate explanations for every image at every round.

Recommended strategy:

```text
Evaluate XAI every 5 or 10 communication rounds.
Use a fixed validation subset of 200–500 images.
```

Example for 50 rounds:

```text
Rounds: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
```

Example for 100 rounds:

```text
Rounds: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
```

---

## 12. Statistical Repetition

To avoid relying on one random run, repeat core experiments.

Minimum:

```text
2 seeds
```

Preferred:

```text
3 seeds
```

Report:

```text
mean ± standard deviation
```

for key metrics.

Core repeated experiments:

- centralized baseline
- 5-client IID FedAvg
- 5-client non-IID FedAvg
- main round-wise explanation tracking

Optional repeated experiments:

- client count analysis
- CheXpert validation
- DP-SGD

---

## 13. Main Result Tables

### Table 1: Predictive Performance

```text
Experiment | Dataset | Training | Clients | Split | AUC | F1 | Precision | Recall | Accuracy
```

---

### Table 2: XAI Faithfulness

```text
Experiment | Dataset | Training | Split | XAI Method | Infidelity | Sensitivity | AOPC
```

---

### Table 3: XAI Stability

```text
Experiment | Dataset | Training | Split | XAI Method | Similarity Prev Round | Similarity Final Round | Explanation Convergence Round
```

---

### Table 4: Localization

```text
Experiment | Dataset | Training | XAI Method | Pointing Game | Top-10% Overlap | IoU-like Score
```

---

### Table 5: Convergence Comparison

```text
Experiment | Split | XAI Method | Accuracy Convergence Round | Explanation Convergence Round | Convergence Gap
```

---

### Table 6: Optional DP Trade-off

```text
ε | AUC | F1 | Infidelity | Sensitivity | Stability | Localization
```

---

## 14. Required Figures

The thesis should include the following figures:

1. Label distribution of the dataset.
2. Example images with bounding boxes.
3. Accuracy/loss over communication rounds.
4. Faithfulness over communication rounds.
5. Stability over communication rounds.
6. Similarity to final explanation over communication rounds.
7. Accuracy convergence vs explanation convergence.
8. Grad-CAM vs LRP examples.
9. Localization examples with bounding boxes.
10. Client count effect on explanation stability.
11. Optional DP privacy-performance-explainability trade-off.

---

## 15. Threats to Validity

The thesis should explicitly discuss the following limitations:

1. Simulated federated clients are not real hospitals.
2. Non-IID splits are artificial approximations of clinical heterogeneity.
3. XAI metrics are proxy measures and do not fully represent clinical trust.
4. Bounding boxes are imperfect approximations of clinically relevant regions.
5. Results may depend on model architecture.
6. Results may depend on selected disease labels.
7. Explanations are evaluated on a subset, not the full validation set.
8. DP-SGD experiments may be sensitive to hyperparameters.
9. CheXpert validation may not be directly comparable due to uncertainty labels and dataset differences.
10. No clinician user study is included.

---

## 16. Reproducibility Requirements

Every experiment should log:

```text
dataset version
preprocessing settings
model architecture
optimizer
learning rate
batch size
number of clients
client split method
number of rounds
local epochs
random seed
checkpoint path
XAI method
XAI evaluation subset image IDs
metric values
runtime
GPU type
```

Save:

```text
configs/
outputs/metrics/
outputs/checkpoints/
outputs/figures/
outputs/heatmaps/
```

The validation subset used for XAI must be saved as:

```text
outputs/metrics/xai_validation_image_ids.csv
```

---

## 17. Final Metric Priority

If time is limited, prioritize metrics in this order:

1. AUC
2. F1
3. Accuracy/loss over rounds
4. Heatmap similarity to final round
5. Consecutive-round heatmap similarity
6. Infidelity
7. AOPC / deletion score
8. Pointing game
9. Top-k bounding-box overlap
10. DP-related metrics

The thesis can still be strong without DP if the explanation-dynamics analysis is complete.

---

## 18. Final Notes

The main scientific contribution is not simply comparing Grad-CAM and LRP.

The main contribution is measuring whether explanations stabilize during federated learning and whether explanation convergence follows, precedes, or lags behind model performance convergence.

Therefore, the most important metrics are:

```text
Accuracy convergence round
Explanation convergence round
Convergence gap
Faithfulness over communication rounds
Stability over communication rounds
```

These should be treated as the central results of the thesis.
