#!/usr/bin/env python3
"""Numerical-metric wrapper for frozen LATH-Net v2.

The model/protocol are unchanged. Float32 cumulative sums can produce values
like 1.000000119, which scikit-learn correctly rejects as probabilities.
This wrapper clips CDF probabilities to [0,1] only at metric/output time.
"""
from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import evaluate_lath_v2_fold as m


def horizon_metrics_clipped(y, cdf):
    out = {}
    for h in m.HORIZONS_S:
        n = int(math.ceil(h / m.BIN_WIDTH_S))
        prob = np.clip(np.asarray(cdf[:, n - 1], dtype=float), 0.0, 1.0)
        lab = (y <= h).astype(int)
        out[f'brier_{h}'] = float(brier_score_loss(lab, prob))
        if len(np.unique(lab)) == 2:
            out[f'auroc_{h}'] = float(roc_auc_score(lab, prob))
            out[f'auprc_{h}'] = float(average_precision_score(lab, prob))
        else:
            out[f'auroc_{h}'] = np.nan
            out[f'auprc_{h}'] = np.nan
    return out


m.horizon_metrics = horizon_metrics_clipped
m.main()
