#!/usr/bin/env python3
"""Prepare a conservative pouch-cell temperature + mechanical-force cohort.

This script is intentionally separate from the frozen 2170 preprocessing.
It avoids two failure modes identified in the raw archive audit:

* A2 force has one long missing suffix. We only use the contiguous measured
  prefix and never interpolate across that gap.
* D1's file is named `Pressure`, although the source paper describes pouch-cell
  mechanics as expansion force. Its tail also contains a long exact +1 ramp
  that is inconsistent with a physical force trace. The ramp and everything
  after its onset are quarantined rather than repaired.

No synthetic data are produced here. The output is a clean training cohort for
RA-CDiff. Shape-stage labels remain coarse temperature-shape labels, not venting
annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_xlsx_two_col(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Expected two columns in {path}")
    df = df.iloc[:, :2].copy()
    df.columns = ["time", "value"]
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.loc[df.time.notna()].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def first_long_unit_ramp_start(v: np.ndarray, min_samples: int = 100, tol: float = 1e-8) -> int | None:
    if len(v) < min_samples + 1:
        return None
    finite = np.isfinite(v)
    d = np.diff(v)
    good = finite[:-1] & finite[1:] & (np.abs(d - 1.0) <= tol)
    i = 0
    while i < len(good):
        if not good[i]:
            i += 1
            continue
        j = i + 1
        while j < len(good) and good[j]:
            j += 1
        # good[i:j] describes samples i..j. Quarantine from sample i+1,
        # because the first transition into the ramp may be a large jump.
        if (j - i + 1) >= min_samples:
            return i + 1
        i = j
    return None


def conservative_valid_prefix(df: pd.DataFrame, detect_ramp: bool) -> tuple[pd.DataFrame, dict]:
    v = df.value.to_numpy(dtype=float)
    cut = len(df)
    reasons = []

    nan_idx = np.flatnonzero(~np.isfinite(v))
    if len(nan_idx):
        cut = min(cut, int(nan_idx[0]))
        reasons.append(f"first_missing_index={int(nan_idx[0])}")

    if detect_ramp:
        ramp_start = first_long_unit_ramp_start(v)
        if ramp_start is not None:
            cut = min(cut, int(ramp_start))
            reasons.append(f"unit_ramp_start_index={int(ramp_start)}")

    clean = df.iloc[:cut].copy()
    clean = clean.loc[np.isfinite(clean.value)].reset_index(drop=True)
    meta = {
        "raw_rows": int(len(df)),
        "clean_rows": int(len(clean)),
        "cut_index": int(cut),
        "reasons": reasons,
        "t_start": float(clean.time.min()),
        "t_end": float(clean.time.max()),
    }
    return clean, meta


def fill_small_temperature_gaps(df: pd.DataFrame, max_gap_samples: int = 2) -> pd.DataFrame:
    out = df.copy().set_index("time")
    # Only small interior gaps are allowed. Long gaps remain NaN and later split
    # the trajectory; this avoids fabricating long thermal segments.
    out["value"] = out["value"].interpolate(method="linear", limit=max_gap_samples, limit_area="inside")
    return out.reset_index()


def stage_anchors(temp: np.ndarray) -> tuple[int, int]:
    if len(temp) < 9:
        p = int(np.nanargmax(temp)); return p, p
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


def common_integer_grid(temp: pd.DataFrame, mech: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0 = int(np.ceil(max(temp.time.min(), mech.time.min())))
    t1 = int(np.floor(min(temp.time.max(), mech.time.max())))
    grid = np.arange(t0, t1 + 1, dtype=float)
    if len(grid) == 0:
        raise RuntimeError("No common time range")

    # Reindex on actual measured integer-second timestamps. Mechanical values are
    # not interpolated across missing/corrupted regions because those regions were
    # already truncated from the valid prefix.
    tv = np.interp(grid, temp.time.to_numpy(dtype=float), temp.value.to_numpy(dtype=float))
    mv = np.interp(grid, mech.time.to_numpy(dtype=float), mech.value.to_numpy(dtype=float))
    return grid, tv.astype(np.float32), mv.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/osf_c2hnq/Dataset_TR/Dataset_TR/Cell/NMC pouch"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/pouch_clean_v1"))
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "experiment_id": "A2", "aging": "fresh",
            "temperature": args.root / "Fresh/A2_Temperature.xlsx",
            "mechanical": args.root / "Fresh/A2_Force.xlsx",
            "mechanical_filename_label": "force", "detect_ramp": False,
        },
        {
            "experiment_id": "D1", "aging": "aged",
            "temperature": args.root / "Aged/D1_Temperature.xlsx",
            "mechanical": args.root / "Aged/D1_Pressure.xlsx",
            "mechanical_filename_label": "pressure", "detect_ramp": True,
        },
    ]

    windows: list[np.ndarray] = []
    meta_rows: list[dict] = []
    cleaning = []

    for spec in specs:
        temp_raw = read_xlsx_two_col(spec["temperature"])
        mech_raw = read_xlsx_two_col(spec["mechanical"])
        temp = fill_small_temperature_gaps(temp_raw)
        mech, clean_meta = conservative_valid_prefix(mech_raw, detect_ramp=bool(spec["detect_ramp"]))
        clean_meta.update({
            "experiment_id": spec["experiment_id"],
            "raw_mechanical_file": str(spec["mechanical"]),
            "raw_filename_label": spec["mechanical_filename_label"],
            "interpreted_physical_modality": "force",
        })
        cleaning.append(clean_meta)

        # If any long temperature gap survived, split by simply limiting to rows
        # with finite values. Here the audited files contain at most isolated gaps.
        temp = temp.loc[np.isfinite(temp.value)].reset_index(drop=True)
        grid, tv, mv = common_integer_grid(temp, mech)
        if len(grid) < args.window:
            raise RuntimeError(f"Too short after cleaning: {spec['experiment_id']}")
        arr = np.stack([tv, mv], axis=0)
        rapid, peak = stage_anchors(tv)

        for start in range(0, len(grid) - args.window + 1, args.stride):
            end = start + args.window
            center_i = start + args.window // 2
            stage = assign_stage(center_i, rapid, peak, args.window)
            windows.append(arr[:, start:end])
            meta_rows.append({
                "experiment_id": spec["experiment_id"],
                "aging": spec["aging"],
                "stage": stage,
                "stage_name": ["pre_rapid", "rapid_peak", "post_peak"][stage],
                "t_start_s": float(grid[start]),
                "t_end_s": float(grid[end - 1]),
                "rapid_anchor_s": float(grid[rapid]),
                "peak_anchor_s": float(grid[peak]),
                "mechanical_modality": "force",
                "raw_mechanical_filename_label": spec["mechanical_filename_label"],
            })

    x = np.stack(windows).astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    center, scale = quantile_scale(x)
    x_norm = (x - center[None, :, None]) / scale[None, :, None]
    age = (meta.aging == "aged").astype(np.int64).to_numpy()
    stage = meta.stage.astype(np.int64).to_numpy()

    np.savez_compressed(
        args.out / "cell_pouch_temp_force_clean.npz",
        x=x_norm.astype(np.float32), age=age, stage=stage,
        experiment_id=meta.experiment_id.to_numpy(dtype="U16"),
        center=center, scale=scale,
        channel_names=np.array(["temperature", "force"], dtype="U32"),
    )
    meta.to_csv(args.out / "cell_pouch_temp_force_clean_windows.csv", index=False)
    (args.out / "cleaning_audit.json").write_text(json.dumps(cleaning, indent=2), encoding="utf-8")

    counts = meta.groupby(["aging", "stage_name"]).size().to_dict()
    summary = {
        "windows": int(len(meta)),
        "experiments": sorted(meta.experiment_id.unique().tolist()),
        "fresh_windows": int((meta.aging == "fresh").sum()),
        "aged_windows": int((meta.aging == "aged").sum()),
        "stage_counts": {f"{k[0]}:{k[1]}": int(v) for k, v in counts.items()},
        "channel_names": ["temperature", "force"],
        "guardrail": "D1 filename says Pressure, but pouch mechanics are treated as force; suspicious tail is quarantined, not repaired.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
