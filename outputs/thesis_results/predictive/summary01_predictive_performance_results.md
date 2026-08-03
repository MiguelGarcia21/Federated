# SUMMARY_01 — Predictive performance across three seeds

## Suggested Results sequence

Predictive performance was compared across the centralized baseline and the two federated conditions using seeds 42, 7, and 123. The centralized value for each seed corresponds to the checkpoint selected by validation F1. Federated results are reported separately for the best validation-F1 round and the final Round-50 model; these selection rules should not be combined or described as equivalent.

Figure 1 compares the selected centralized checkpoints with the final IID and non-IID federated models using F1 and balanced accuracy. Across the three seeds, mean F1 was 0.860 for centralized training, 0.867 for IID FedAvg, and 0.860 for non-IID FedAvg. Mean balanced accuracy was 0.906, 0.912, and 0.908, respectively. The differences were small relative to the overall performance level, and no condition showed a persistent predictive collapse across the tested seeds.

Figure 2 provides complementary evidence using accuracy and ROC AUC. Mean accuracy was 0.916 for the selected centralized checkpoints, 0.920 for IID Round 50, and 0.915 for non-IID Round 50. Mean ROC AUC was 0.967, 0.968, and 0.967, respectively. These results show that the three training conditions remained broadly comparable not only in threshold-dependent metrics but also in ranking performance.

Figure 3 examines model-selection timing within the federated experiments. The best validation-F1 round occurred at a mean round of 28.0 for IID training and 27.0 for non-IID training. The mean difference between the best F1 and Round-50 F1 was 0.0136 for IID and 0.0127 for non-IID. Thus, the final communication round was not consistently the strongest validation-F1 model, supporting the decision to report both best-round and final-round results.

For the centralized condition, the selected epochs were seed 42: epoch 2, seed 7: epoch 10, seed 123: epoch 9. The variation in selected epoch shows that checkpoint selection was also relevant for centralized training even though its final predictive values were stable across seeds.

## Suggested Discussion links

The first two figures support the interpretation that five-client FedAvg preserved predictive viability under both IID and label-skew non-IID partitioning. The appropriate claim is comparability rather than superiority: the sample contains only three seeds, and the experiment was designed as a controlled robustness check rather than a population-level statistical comparison.

Figure 3 supports a separate model-selection argument. Continued optimization to Round 50 did not guarantee a higher validation F1. This should be discussed as evidence that convergence of the training procedure and selection of the most useful validation checkpoint are related but distinct decisions.

## Caveats

- The centralized condition uses its selected best-validation-F1 checkpoint, whereas the displayed federated comparison uses Round 50.
- Best-round federated results are reported separately in Tables 2 and 4.
- Mean and standard deviation are descriptive summaries over three seeds.
- No significance test is implied by these figures.
