# Discussion chapter evidence outline

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
