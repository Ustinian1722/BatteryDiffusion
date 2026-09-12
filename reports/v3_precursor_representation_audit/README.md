# V3 precursor representation sanity audit

Fixed Random Forest under the same 6-train/1-validation/1-test TTV protocol. This is a representation diagnostic, not a model-selection result. Validation experiments are excluded from RF fitting even though RF hyperparameters are fixed.

| representation | mean MAE | worst MAE | mean RMSE | mean R2 |
|---|---:|---:|---:|---:|
| fusion_absolute | 40.42 | 62.90 | 44.59 | 0.7826 |
| temp_absP_relP_dP | 40.88 | 66.95 | 45.47 | 0.7700 |
| temp_relP | 46.85 | 79.16 | 52.41 | 0.7005 |
| temperature | 47.59 | 77.54 | 51.05 | 0.6982 |
| temp_relP_dP | 47.62 | 79.80 | 53.01 | 0.6947 |
| temp_relP_dP_shape | 48.57 | 81.80 | 54.45 | 0.6806 |

Interpretation: if relative/differential pressure variants outperform or stabilize `fusion_absolute`, the precursor-invariant representation has independent support. If they do not, v3 can still succeed through learned lag fusion, but removal of absolute pressure must not be claimed intrinsically beneficial.
