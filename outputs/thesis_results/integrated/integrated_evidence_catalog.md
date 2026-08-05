# Integrated thesis-results evidence catalog

Stable findings:

1. Centralized, IID FedAvg, and non-IID label-skew FedAvg achieved closely grouped predictive performance across three seeds.
2. Peak federated validation F1 occurred before Round 50 in every run, with modest best-versus-final gaps.
3. Grad-CAM explanations changed strongly early and then became increasingly stable.
4. Prediction-target mismatch relative to Round 50 declined rapidly.
5. IID and non-IID federated explanations were consistently more similar to each other than either was to centralized training.
6. High prediction agreement did not guarantee high spatial explanation agreement.

Interpretation boundaries:

- Descriptive summaries use three seeds; no significance test is implied.
- Centralized selected checkpoints and federated Round-50 checkpoints use different selection rules.
- Later adjacent-checkpoint XAI values are not pooled across unequal checkpoint intervals.
- Qualitative agreement does not establish correctness.
