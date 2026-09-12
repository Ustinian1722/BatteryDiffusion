#!/usr/bin/env python3
"""Prepare a compact, physically structured module-level cohort.

The raw module temperature export contains a repeating millisecond field and 15
thermocouple channels. The accompanying paper maps TC1-6 to cell 1, TC7-9 to
cell 2, TC10-11 to cell 3, and TC12-15 to cell 4. We reconstruct the 5 Hz time
base from the repeated millisecond field, aggregate each cell's thermocouples by
per-second maximum temperature (hot-spot preserving), and align those four cell
signals with the 1 Hz module force trace.

This yields five channels per second:
    [T_cell1_max, T_cell2_max, T_cell3_max, T_cell4_max, force]

The module cohort has only one fresh and one aged experiment, so synthetic
windows are local augmentation variants only and must never be interpreted as
new module experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def reconstruct_time(df: pd.DataFrame) -> np.ndarray:
    ms = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(ms).mean() < 0.99:
        raise ValueError("Module millisecond column is not numeric")
    resets = np.zeros(len(ms), dtype=int)
    resets[1:] = (np.diff(ms) < 0).astype(int)
    sec = np.cumsum(resets)
    if not (np.nanmin(ms) >= 0 and np.nanmax(ms) < 1000 and resets.sum() > 10):
        raise ValueError("First module temperature field does not behave like milliseconds-within-second")
    return sec.astype(float) + ms / 1000.0


def read_force(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None).iloc[:, :2].copy()
    df.columns = ["time", "force"]
    df["time"] = pd.to_numeric(df.time, errors="coerce")
    df["force"] = pd.to_numeric(df.force, errors="coerce")
    return df.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)


def stage_anchors(temp: np.ndarray) -> tuple[int, int]:
    k = min(11, (len(temp) // 2) * 2 + 1)
    smooth = np.convolve(temp, np.ones(k) / k, mode="same")
    peak = int(np.argmax(smooth))
    grad = np.gradient(smooth)
    rapid = int(np.argmax(grad[: peak + 1])) if peak > 2 else peak
    return rapid, peak


def assign_stage(center_idx: int, rapid_idx: int, peak_idx: int, window: int) -> int:
    margin = max(window // 4, 1)
    if center_idx < rapid_idx - margin:
        return 0
    if center_idx <= peak_idx + margin:
        return 1
    return 2


def quantile_scale(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    flat = np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)
    q01 = np.nanpercentile(flat, 1, axis=1)
    q99 = np.nanpercentile(flat, 99, axis=1)
    center = 0.5 * (q01 + q99)
    scale = np.maximum(0.5 * (q99 - q01), eps)
    return center.astype(np.float32), scale.astype(np.float32)


def aggregate_module(temp_path: Path, force_path: Path) -> tuple[pd.DataFrame, dict]:
    tdf = pd.read_csv(temp_path)
    t = reconstruct_time(tdf)
    temp_cols = list(tdf.columns[1:])
    if len(temp_cols) != 15:
        raise ValueError(f"Expected 15 thermocouple channels, got {len(temp_cols)}")
    vals = tdf[temp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Column order corresponds to TC1..TC15 in the repository export.
    groups = {
        "T_cell1_max": list(range(0, 6)),
        "T_cell2_max": list(range(6, 9)),
        "T_cell3_max": list(range(9, 11)),
        "T_cell4_max": list(range(11, 15)),
    }
    sec = np.floor(t + 1e-9).astype(int)
    frame = pd.DataFrame({"second": sec})
    for name, idx in groups.items():
        frame[name] = np.nanmax(vals[:, idx], axis=1)
    # Preserve the hottest observation in each second; rapid TR peaks can be
    # substantially sharper than a one-second mean.
    per_sec = frame.groupby("second", as_index=False).max()

    force = read_force(force_path)
    force["second"] = np.rint(force.time).astype(int)
    force = force.groupby("second", as_index=False).force.mean()
    merged = per_sec.merge(force, on="second", how="inner").sort_values("second").reset_index(drop=True)

    # Only contiguous one-second runs are eligible for windowing. We report run
    # structure and later generate windows within each run independently.
    gap = np.diff(merged.second.to_numpy())
    run_id = np.zeros(len(merged), dtype=int)
    if len(merged) > 1:
        run_id[1:] = np.cumsum(gap != 1)
    merged["run_id"] = run_id
    run_lengths = merged.groupby("run_id").size().sort_values(ascending=False).tolist()
    meta = {
        "temperature_file": str(temp_path),
        "force_file": str(force_path),
        "temperature_reconstructed_t_start": float(t[0]),
        "temperature_reconstructed_t_end": float(t[-1]),
        "aligned_t_start": int(merged.second.min()),
        "aligned_t_end": int(merged.second.max()),
        "aligned_rows": int(len(merged)),
        "contiguous_runs": int(merged.run_id.nunique()),
        "run_lengths_desc": [int(x) for x in run_lengths],
    }
    return merged, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/osf_c2hnq/Dataset_TR/Dataset_TR/Module/NMC pouch"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/module_aligned_v1"))
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = [
        ("M1", "fresh", args.root / "Fresh/M1_Temperature.csv", args.root / "Fresh/M1_Force.csv"),
        ("M2", "aged", args.root / "Aged/M2_Temperature.csv", args.root / "Aged/M2_Force.csv"),
    ]

    windows = []
    rows = []
    align_meta = []
    channels = ["T_cell1_max", "T_cell2_max", "T_cell3_max", "T_cell4_max", "force"]

    for eid, aging, tp, fp in specs:
        df, meta = aggregate_module(tp, fp)
        meta.update(experiment_id=eid, aging=aging)
        align_meta.append(meta)
        # Stage anchors use the full aligned trigger-cell temperature trace.
        rapid_global, peak_global = stage_anchors(df.T_cell1_max.to_numpy(dtype=float))
        rapid_second = int(df.second.iloc[rapid_global])
        peak_second = int(df.second.iloc[peak_global])

        for rid, g in df.groupby("run_id"):
            g = g.sort_values("second").reset_index(drop=True)
            if len(g) < args.window:
                continue
            arr = g[channels].to_numpy(dtype=np.float32).T
            for start in range(0, len(g) - args.window + 1, args.stride):
                end = start + args.window
                center_second = int(g.second.iloc[start + args.window // 2])
                # Assign stage in global test-time coordinates so runs share the
                # same physical phase definition.
                margin = max(args.window // 4, 1)
                if center_second < rapid_second - margin:
                    stage = 0
                elif center_second <= peak_second + margin:
                    stage = 1
                else:
                    stage = 2
                windows.append(arr[:, start:end])
                rows.append({
                    "experiment_id": eid, "aging": aging, "run_id": int(rid),
                    "stage": stage, "stage_name": ["pre_rapid", "rapid_peak", "post_peak"][stage],
                    "t_start_s": int(g.second.iloc[start]), "t_end_s": int(g.second.iloc[end - 1]),
                    "rapid_anchor_s": rapid_second, "peak_anchor_s": peak_second,
                })

    x = np.stack(windows).astype(np.float32)
    meta_df = pd.DataFrame(rows)
    center, scale = quantile_scale(x)
    x_norm = (x - center[None, :, None]) / scale[None, :, None]
    age = (meta_df.aging == "aged").astype(np.int64).to_numpy()
    stage = meta_df.stage.astype(np.int64).to_numpy()

    np.savez_compressed(
        args.out / "module_temp4_force.npz", x=x_norm.astype(np.float32), age=age, stage=stage,
        experiment_id=meta_df.experiment_id.to_numpy(dtype="U16"), center=center, scale=scale,
        channel_names=np.asarray(channels, dtype="U32"),
    )
    meta_df.to_csv(args.out / "module_windows.csv", index=False)
    (args.out / "alignment_audit.json").write_text(json.dumps(align_meta, indent=2), encoding="utf-8")
    counts = meta_df.groupby(["aging", "stage_name"]).size().to_dict()
    summary = {
        "windows": int(len(meta_df)), "experiments": sorted(meta_df.experiment_id.unique().tolist()),
        "channels": channels,
        "stage_counts": {f"{k[0]}:{k[1]}": int(v) for k, v in counts.items()},
        "guardrail": "Only two independent module experiments exist; generated windows are local augmentation, not new modules.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
