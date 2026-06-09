# Model comparison notes (report)

Generated UTC: 2026-06-08T13:36:44.611474+00:00

## Models compared

- **DINO-full**: heads = knn_k100, linear_probe, patient_attention_mil, patient_mean_embed, patient_mean_from_probe
- **Partner DINO+iBOT cross-slice**: heads = cross_slice_transformer
- **Partner DINO-only (May 2026)**: heads = linear_probe, patient_attention_mil, patient_mean_from_probe

## Metrics used

Downstream clinical metrics: ROC-AUC, balanced accuracy, accuracy, precision/recall/F1 (slice linear probe only).
Representation-only metrics (PCA, silhouette, etc.) are excluded.

## Split and level

- Split file: `/mnt/md0/Liron/dbt_simclr_project/artifacts/splits/patient_split_v1.json` (seed 42)
- Each row is labeled with `split` (val/test) and `input_level` (slice/patient).
- kNN rows are **val-only** by protocol.

## Best performers (test)

- **slice-level AUROC**: DINO-full (linear_probe) = 0.6360
- **patient-level AUROC**: DINO-full (patient_attention_mil) = 0.6978
- **Balanced accuracy (test, primary threshold policy)**: DINO-full (patient_attention_mil, patient) = 0.6559

## Caveats

- Partner DINO-only (May 2026) and DINO-full (Jun 2026) use **different SSL checkpoints**.
- Partner linear probe early-stops on `val_acc`; DINO-full probe on `val_roc_auc`.
- Do not compare kNN (val) with probe test AUROC directly.
- ImageNet baseline uses ImageNet normalization; not feature-comparable to DBT-SSL embeddings.
- Cross-slice transformer and attention MIL use different aggregation heads.
- Threshold: patient-level test rows use threshold tuned on val balanced accuracy unless noted `@0.5`.
