# Model comparison

Selection rule: max validation macro-F1, then min validation log-loss, then fewest parameters. Selected: **efficientnet_b0**.

| model | val_macro_f1 | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_roc_auc_ovr | test_original_photos_macro_f1 | test_strict_macro_f1 | test_one_per_leaf_acc_95ci_low | test_background_neutralized_accuracy | test_errors | test_n | params_millions | latency_ms_cpu | train_minutes | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9920 | 0.9994 | 0 | 1798 | 2.2290 | 39.3000 | 17.9333 | False |
| resnet50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9920 | 1.0000 | 0 | 1798 | 23.5162 | 54.5400 | 43.8917 | False |
| efficientnet_b0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9920 | 0.9989 | 0 | 1798 | 4.0127 | 165.0200 | 34.6350 | True |

Full table: `model_comparison.csv`.
