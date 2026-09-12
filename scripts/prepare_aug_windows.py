#!/usr/bin/env python3
"""Prepare leakage-aware fixed-length windows for generative augmentation.

V1 intentionally starts with single-cell paired experiments whose two-column
exports contain an explicit elapsed-time axis. Module data are preserved in the
repository but are not forced into this first pilot because the module
thermocouple export uses a different multi-channel time representation.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _looks_numeric(x: object) -> bool:
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


def read_two_col(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if len(df.columns) and all(_looks_numeric(c) for c in df.columns):
            df = pd.read_csv(path, header=None)
    else:
        df = pd.read_excel(path)
        if len(df.columns) and all(_looks_numeric(c) for c in df.columns):
            df = pd.read_excel(path, header=None)
    if df.shape[1] != 2:
        raise ValueError(f"Expected two-column signal file, got {df.shape[1]} columns: {path}")
    df = df.iloc[:, :2].copy()
    df.columns = ["time", "value"]
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates("time")
    return df


def exp_id(path: Path) -> str:
    m = re.match(r"([A-Za-z]+\d+)_", path.name)
    if not m:
        raise ValueError(f"Cannot infer experiment ID from {path.name}")
    return m.group(1).upper()


def context_from_path(path: Path) -> tuple[str, str, str]:
    parts = [p.lower() for p in path.parts]
    level = "module" if "module" in parts else "cell"
    form = "2170" if any("2170" in p for p in parts) else "pouch"
    aging = "aged" if "aged" in parts else "fresh"
    return level, form, aging


def find_signal_files(root: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        try:
            e = exp_id(p)
        except ValueError:
            continue
        sig = None
        low = p.stem.lower()
        if "temperature" in low:
            sig = "temperature"
        elif "pressure" in low:
            sig = "pressure"
        elif "force" in low:
            sig = "force"
        if sig:
            out.setdefault(e, {})[sig] = p
    return out


def quantile_scale(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Scale most observed values to approximately [-1, 1].

    Thermal-runaway signals are extremely heavy-tailed. Median/IQR scaling can
    make the rare but physically important peak region numerically huge, so the
    pilot uses the 1st/99th-percentile midpoint and half-range instead.
    """
    flat = np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)
    q01 = np.nanpercentile(flat, 1, axis=1)
    q99 = np.nanpercentile(flat, 99, axis=1)
    center = 0.5 * (q01 + q99)
    scale = np.maximum(0.5 * (q99 - q01), eps)
    return center.astype(np.float32), scale.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/osf_c2hnq"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/augmentation_v1"))
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--exclude-experiment", action="append", default=[])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    excluded = {x.upper() for x in args.exclude_experiment}
    files = find_signal_files(args.root)

    cohort_windows: dict[str, list[np.ndarray]] = {}
    cohort_meta: dict[str, list[dict]] = {}

    for eid, signals in sorted(files.items()):
        if eid in excluded or "temperature" not in signals:
            continue
        temp_path = signals["temperature"]
        level, form, aging = context_from_path(temp_path)
        if level != "cell":
            continue
        mech_key = "pressure" if "pressure" in signals else "force" if "force" in signals else None
        if mech_key is None:
            continue
        try:
            temp = read_two_col(temp_path)
            mech = read_two_col(signals[mech_key])
        except ValueError:
            continue

        t0 = max(float(temp.time.min()), float(mech.time.min()))
        t1 = min(float(temp.time.max()), float(mech.time.max()))
        if not np.isfinite(t0 + t1) or t1 <= t0:
            continue
        dt_t = np.median(np.diff(temp.time.to_numpy()))
        dt_m = np.median(np.diff(mech.time.to_numpy()))
        dt = float(max(dt_t, dt_m))
        if not np.isfinite(dt) or dt <= 0:
            continue
        grid = np.arange(t0, t1 + 0.5 * dt, dt, dtype=np.float64)
        if len(grid) < args.window:
            continue
        tv = np.interp(grid, temp.time.to_numpy(), temp.value.to_numpy())
        mv = np.interp(grid, mech.time.to_numpy(), mech.value.to_numpy())
        arr = np.stack([tv, mv], axis=0).astype(np.float32)

        cohort = "cell_2170_temp_pressure" if form == "2170" else "cell_pouch_temp_mechanical"
        cohort_windows.setdefault(cohort, [])
        cohort_meta.setdefault(cohort, [])
        for start in range(0, arr.shape[1] - args.window + 1, args.stride):
            end = start + args.window
            cohort_windows[cohort].append(arr[:, start:end])
            cohort_meta[cohort].append({
                "experiment_id": eid,
                "aging": aging,
                "level": level,
                "form_factor": form,
                "mechanical_source_name": mech_key,
                "dt_s": dt,
                "t_start_s": float(grid[start]),
                "t_end_s": float(grid[end - 1]),
                "window_start_index": start,
            })

    summary = []
    for cohort, wins in cohort_windows.items():
        if not wins:
            continue
        x = np.stack(wins, axis=0).astype(np.float32)
        meta = pd.DataFrame(cohort_meta[cohort])
        center, scale = quantile_scale(x)
        x_norm = (x - center[None, :, None]) / scale[None, :, None]
        age = (meta["aging"].str.lower() == "aged").astype(np.int64).to_numpy()
        np.savez_compressed(
            args.out / f"{cohort}.npz",
            x=x_norm.astype(np.float32),
            age=age,
            experiment_id=meta["experiment_id"].to_numpy(dtype="U16"),
            center=center,
            scale=scale,
            channel_names=np.array(["temperature", "mechanical"], dtype="U32"),
        )
        meta.to_csv(args.out / f"{cohort}_windows.csv", index=False)
        (args.out / f"{cohort}_scaler.json").write_text(json.dumps({
            "center": center.tolist(),
            "scale": scale.tolist(),
            "normalization": "global 1st/99th-percentile midpoint and half-range on selected training experiments",
            "excluded_experiments": sorted(excluded),
        }, indent=2), encoding="utf-8")
        summary.append({
            "cohort": cohort,
            "windows": int(len(x)),
            "experiments": int(meta.experiment_id.nunique()),
            "experiment_ids": "+".join(sorted(meta.experiment_id.unique())),
            "fresh_windows": int((meta.aging == "fresh").sum()),
            "aged_windows": int((meta.aging == "aged").sum()),
        })

    pd.DataFrame(summary).to_csv(args.out / "cohort_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
