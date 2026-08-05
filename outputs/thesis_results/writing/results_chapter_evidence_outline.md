# Results chapter evidence outline

## Section 1 — Predictive performance across training conditions

Opening paragraph:
- State that the comparison includes seeds 42, 7, and 123.
- State the checkpoint rules before reporting values.
- Centralized values use the selected validation-F1 checkpoint.
- Federated comparison values use Round 50.

Figure 1 evidence:
- Mean validation F1: centralized 0.860, IID 0.867, non-IID 0.860.
- Balanced accuracy follows the same broad comparability pattern.
- Appropriate claim: broadly comparable performance.
- Avoid claiming superiority from three descriptive replications.

Figure 2 evidence:
- Mean ROC AUC: centralized 0.967, IID 0.968, non-IID 0.967.
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
- Mean target mismatch at Round 5 is approximately 17.8% for IID
  and 14.8% for non-IID.
- By Round 30, mismatch falls to approximately 2.5% and
  4.3%, respectively.
- This supports increasing comparability of later Grad-CAM maps.

## Section 4 — Cross-model prediction and explanation comparison

Figure 9 evidence:
- Mean Pearson: centralized–IID 0.574, centralized–non-IID
  0.554, IID–non-IID 0.844.
- Mean Dice: centralized–IID 0.549, centralized–non-IID
  0.534, IID–non-IID 0.734.
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
