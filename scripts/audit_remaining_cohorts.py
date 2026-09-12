#!/usr/bin/env python3
"""Focused audit for pouch-cell and module cohorts not yet used by RA-CDiff.

Goals
-----
1. Reconstruct the module temperature time base from the repeated `ms` field
   without silently assuming that row index is elapsed seconds.
2. Quantify missing-value runs in the A2/D1 mechanical files so future windowing
   never interpolates across long sensor gaps.
3. Detect suspicious near-unit-ramp segments in D1's mechanical export and keep
   them quarantined until their provenance is understood.

The script only audits; it does not mutate raw data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_two_col_raw(path: Path) -> pd.DataFrame:
    """Read the raw two-column workbook preserving missing cells."""
    df = pd.read_excel(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Expected >=2 columns: {path}")
    out = df.iloc[:, :2].copy()
    out.columns = ["time", "value"]
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    # Drop an optional textual header, but preserve all data-row NaNs afterwards.
    out = out.loc[out["time"].notna()].reset_index(drop=True)
    return out


def nan_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        runs.append((i, j - 1, j - i))
        i = j
    return runs


def unit_ramp_runs(values: np.ndarray, tol: float = 1e-8) -> list[tuple[int, int, int]]:
    finite = np.isfinite(values)
    good = np.zeros(max(len(values) - 1, 0), dtype=bool)
    if len(values) >= 2:
        good = finite[:-1] & finite[1:] & (np.abs(np.diff(values) - 1.0) <= tol)
    # A run of k good diffs corresponds to k+1 samples.
    runs = []
    for s, e, n in nan_runs(good):
        runs.append((s, e + 1, n + 1))
    return runs


def audit_mechanical(path: Path) -> dict:
    df = read_two_col_raw(path)
    v = df["value"].to_numpy(dtype=float)
    missing = ~np.isfinite(v)
    mruns = nan_runs(missing)
    ramps = unit_ramp_runs(v)
    longest_missing = max((r[2] for r in mruns), default=0)
    longest_ramp = max(ramps, key=lambda r: r[2], default=None)

    rec = {
        "file": str(path),
        "rows": int(len(df)),
        "missing_values": int(missing.sum()),
        "missing_run_count": int(len(mruns)),
        "longest_missing_run": int(longest_missing),
        "missing_runs_top10": sorted(mruns, key=lambda r: r[2], reverse=True)[:10],
        "longest_unit_ramp": longest_ramp,
    }
    finite_v = v[np.isfinite(v)]
    if len(finite_v):
        rec.update({
            "q01": float(np.quantile(finite_v, 0.01)),
            "median": float(np.median(finite_v)),
            "q99": float(np.quantile(finite_v, 0.99)),
            "min": float(np.min(finite_v)),
            "max": float(np.max(finite_v)),
        })

    # Save a context slice around the longest suspicious unit ramp.
    if longest_ramp is not None:
        s, e, n = longest_ramp
        lo, hi = max(0, s - 10), min(len(df), e + 11)
        rec["longest_unit_ramp_context"] = df.iloc[lo:hi].to_dict(orient="records")
    return rec


def reconstruct_module_time(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    first = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(first)
    if valid.mean() < 0.99:
        raise ValueError("Module first column is not sufficiently numeric")

    rounded_unique = np.unique(np.round(first, 9))
    resets = np.zeros(len(first), dtype=int)
    if len(first) > 1:
        resets[1:] = (np.diff(first) < 0).astype(int)
    sec_index = np.cumsum(resets)

    # Candidate interpretation: first column is milliseconds within each second.
    # This is accepted only if it is bounded within [0, 1000) and repeatedly resets.
    ms_like = bool(np.nanmin(first) >= 0 and np.nanmax(first) < 1000 and resets.sum() > 10)
    if ms_like:
        t = sec_index.astype(float) + first / 1000.0
    else:
        t = np.arange(len(first), dtype=float)

    dt = np.diff(t)
    pos = dt[np.isfinite(dt) & (dt > 0)]
    meta = {
        "first_column_name": str(df.columns[0]),
        "first_column_min": float(np.nanmin(first)),
        "first_column_max": float(np.nanmax(first)),
        "unique_first_values": [float(x) for x in rounded_unique[:50]],
        "n_unique_first_values": int(len(rounded_unique)),
        "reset_count": int(resets.sum()),
        "ms_like": ms_like,
        "reconstructed_duration_s": float(t[-1] - t[0]) if len(t) else np.nan,
        "median_positive_dt_s": float(np.median(pos)) if len(pos) else np.nan,
        "sampling_hz": float(1.0 / np.median(pos)) if len(pos) and np.median(pos) > 0 else np.nan,
        "nonpositive_dt_count": int((dt <= 0).sum()),
    }
    return t, meta


def audit_module(temp_path: Path, force_path: Path) -> tuple[dict, pd.DataFrame]:
    tdf = pd.read_csv(temp_path)
    fdf = pd.read_csv(force_path, header=None)
    # Force files in this archive are two-column numeric exports, occasionally
    # parsed with the first row as a header by default pandas behavior.
    if fdf.shape[1] < 2:
        raise ValueError(f"Bad force file: {force_path}")
    fdf = fdf.iloc[:, :2].copy()
    fdf.columns = ["time", "force"]
    fdf["time"] = pd.to_numeric(fdf["time"], errors="coerce")
    fdf["force"] = pd.to_numeric(fdf["force"], errors="coerce")
    fdf = fdf.dropna().reset_index(drop=True)

    t, meta = reconstruct_module_time(tdf)
    value_cols = list(tdf.columns[1:])
    values = tdf[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    channel_rows = []
    for j, name in enumerate(value_cols):
        y = values[:, j]
        finite = np.isfinite(y)
        if finite.sum() < 3:
            continue
        yf = y.copy()
        # Local derivative on reconstructed uniform-ish time base; NaNs preserved.
        dy = np.full_like(yf, np.nan, dtype=float)
        ok = finite[:-1] & finite[1:] & (np.diff(t) > 0)
        d = np.diff(yf)
        dt = np.diff(t)
        idx = np.where(ok)[0]
        dy[idx + 1] = d[idx] / dt[idx]
        peak_i = int(np.nanargmax(yf))
        rapid_i = int(np.nanargmax(dy)) if np.isfinite(dy).any() else -1
        channel_rows.append({
            "channel": str(name),
            "peak_temp_c": float(yf[peak_i]),
            "peak_time_s_reconstructed": float(t[peak_i]),
            "max_dTdt_c_per_s": float(dy[rapid_i]) if rapid_i >= 0 else np.nan,
            "max_dTdt_time_s_reconstructed": float(t[rapid_i]) if rapid_i >= 0 else np.nan,
        })

    meta.update({
        "temperature_file": str(temp_path),
        "temperature_rows": int(len(tdf)),
        "temperature_channels": int(len(value_cols)),
        "force_file": str(force_path),
        "force_rows": int(len(fdf)),
        "force_t_start": float(fdf.time.min()),
        "force_t_end": float(fdf.time.max()),
        "force_duration_s": float(fdf.time.max() - fdf.time.min()),
        "temperature_force_duration_ratio": float((t[-1] - t[0]) / (fdf.time.max() - fdf.time.min())) if len(t) > 1 else np.nan,
    })
    return meta, pd.DataFrame(channel_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/osf_c2hnq/Dataset_TR/Dataset_TR"))
    ap.add_argument("--out", type=Path, default=Path("reports/remaining_cohort_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mech_paths = [
        args.root / "Cell/NMC pouch/Fresh/A2_Force.xlsx",
        args.root / "Cell/NMC pouch/Aged/D1_Pressure.xlsx",
    ]
    mech = [audit_mechanical(p) for p in mech_paths]
    (args.out / "pouch_mechanical_audit.json").write_text(json.dumps(mech, indent=2), encoding="utf-8")

    module_specs = [
        ("M1", args.root / "Module/NMC pouch/Fresh/M1_Temperature.csv", args.root / "Module/NMC pouch/Fresh/M1_Force.csv"),
        ("M2", args.root / "Module/NMC pouch/Aged/M2_Temperature.csv", args.root / "Module/NMC pouch/Aged/M2_Force.csv"),
    ]
    module_meta = []
    channel_tables = []
    for eid, tp, fp in module_specs:
        meta, ch = audit_module(tp, fp)
        meta["experiment_id"] = eid
        module_meta.append(meta)
        ch.insert(0, "experiment_id", eid)
        channel_tables.append(ch)
    pd.DataFrame(module_meta).to_csv(args.out / "module_timebase_audit.csv", index=False)
    pd.concat(channel_tables, ignore_index=True).to_csv(args.out / "module_temperature_events.csv", index=False)

    lines = ["# Remaining-cohort audit", "", "## Pouch-cell mechanical channels", ""]
    for r in mech:
        lines += [
            f"### `{Path(r['file']).name}`",
            f"- rows: **{r['rows']}**; missing: **{r['missing_values']}**; missing runs: **{r['missing_run_count']}**; longest gap: **{r['longest_missing_run']} samples**",
            f"- q01 / median / q99: **{r.get('q01', np.nan):.4f} / {r.get('median', np.nan):.4f} / {r.get('q99', np.nan):.4f}**",
            f"- longest exact +1-per-sample ramp: **{r['longest_unit_ramp']}**",
            "",
        ]
    lines += ["## Module temperature time base", ""]
    for r in module_meta:
        lines += [
            f"### {r['experiment_id']}",
            f"- first field `{r['first_column_name']}`: {r['n_unique_first_values']} unique values, {r['reset_count']} resets, ms-like={r['ms_like']}",
            f"- reconstructed duration: **{r['reconstructed_duration_s']:.2f} s**; median dt: **{r['median_positive_dt_s']:.4f} s**; inferred rate: **{r['sampling_hz']:.2f} Hz**",
            f"- force duration: **{r['force_duration_s']:.2f} s**; duration ratio T/F: **{r['temperature_force_duration_ratio']:.3f}**",
            "",
        ]
    lines += [
        "## Guardrail",
        "",
        "This report does not repair or interpolate raw signals. Long gaps and suspicious ramps remain quarantined until a defensible cleaning/alignment rule is established.",
    ]
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
