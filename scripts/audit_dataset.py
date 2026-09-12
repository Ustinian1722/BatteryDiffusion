#!/usr/bin/env python3
"""Audit the extracted OSF thermal-runaway dataset.

The script treats each experiment ID (A1, A2, B1, B2, C1, C2, D1, M1, M2)
as the independent experimental unit. It reports file-level signal statistics and
experiment-level modality availability without treating time samples as
independent experiments.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def experiment_id(path: Path) -> str:
    m = re.match(r"([A-Za-z]+\d+)_", path.name)
    return m.group(1).upper() if m else path.stem.split("_")[0].upper()


def signal_name(path: Path) -> str:
    stem = path.stem.lower()
    if "temperature" in stem:
        return "temperature"
    if "pressure" in stem:
        return "pressure"
    if "force" in stem:
        return "force"
    return "unknown"


def infer_context(path: Path) -> tuple[str, str, str]:
    parts = [p.lower() for p in path.parts]
    level = "module" if "module" in parts else "cell" if "cell" in parts else "unknown"
    form = "2170" if any("2170" in p for p in parts) else "pouch" if any("pouch" in p for p in parts) else "unknown"
    aging = "aged" if "aged" in parts else "fresh" if "fresh" in parts else "unknown"
    return level, form, aging


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def infer_time_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "time" in str(c).lower():
            return str(c)
    # fallback: first mostly numeric, monotonic column
    for c in df.columns:
        s = numeric_series(df, c).dropna()
        if len(s) >= 3 and s.is_monotonic_increasing:
            return str(c)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/osf_c2hnq"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in args.root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
    )
    file_rows: list[dict] = []

    for path in files:
        rec: dict = {
            "experiment_id": experiment_id(path),
            "signal": signal_name(path),
            "path": str(path),
        }
        level, form, aging = infer_context(path)
        rec.update(level=level, form_factor=form, aging=aging)
        try:
            df = read_table(path)
            rec["rows"] = len(df)
            rec["columns"] = json.dumps([str(c) for c in df.columns], ensure_ascii=False)
            rec["missing_cells"] = int(df.isna().sum().sum())
            time_col = infer_time_column(df)
            rec["time_column"] = time_col or ""
            if time_col:
                t = numeric_series(df, time_col).dropna().to_numpy(dtype=float)
                if len(t) >= 2:
                    dt = np.diff(t)
                    dt = dt[np.isfinite(dt) & (dt > 0)]
                    rec["t_start"] = float(t[0])
                    rec["t_end"] = float(t[-1])
                    rec["duration_s"] = float(t[-1] - t[0])
                    if len(dt):
                        rec["median_dt_s"] = float(np.median(dt))
                        rec["sampling_hz"] = float(1.0 / np.median(dt))
                        rec["dt_cv"] = float(np.std(dt) / np.mean(dt)) if np.mean(dt) else np.nan
            value_cols = [str(c) for c in df.columns if str(c) != time_col]
            # Store simple ranges for all numeric signal columns.
            ranges = {}
            for c in value_cols:
                s = numeric_series(df, c)
                finite = s[np.isfinite(s)]
                if len(finite):
                    ranges[c] = {
                        "min": float(finite.min()),
                        "max": float(finite.max()),
                        "mean": float(finite.mean()),
                        "std": float(finite.std(ddof=0)),
                    }
            rec["numeric_ranges"] = json.dumps(ranges, ensure_ascii=False)
            rec["error"] = ""
        except Exception as exc:  # keep audit running across malformed files
            rec["error"] = repr(exc)
        file_rows.append(rec)

    files_df = pd.DataFrame(file_rows)
    files_df.to_csv(args.out / "signal_file_audit.csv", index=False)

    exp_rows: list[dict] = []
    if not files_df.empty:
        for exp, g in files_df.groupby("experiment_id", sort=True):
            signals = sorted(set(g["signal"].dropna().astype(str)))
            row = {
                "experiment_id": exp,
                "level": g["level"].iloc[0],
                "form_factor": g["form_factor"].iloc[0],
                "aging": g["aging"].iloc[0],
                "signals": "+".join(signals),
                "n_signal_files": len(g),
                "has_temperature": int("temperature" in signals),
                "has_pressure": int("pressure" in signals),
                "has_force": int("force" in signals),
                "multimodal": int("temperature" in signals and ("pressure" in signals or "force" in signals)),
            }
            # Sampling/duration summaries across available signals.
            for col in ("rows", "duration_s", "median_dt_s", "sampling_hz"):
                if col in g.columns:
                    vals = pd.to_numeric(g[col], errors="coerce").dropna()
                    row[f"{col}_min"] = float(vals.min()) if len(vals) else np.nan
                    row[f"{col}_max"] = float(vals.max()) if len(vals) else np.nan
            exp_rows.append(row)

    exp_df = pd.DataFrame(exp_rows)
    exp_df.to_csv(args.out / "experiment_inventory.csv", index=False)

    n_exp = len(exp_df)
    n_multi = int(exp_df["multimodal"].sum()) if n_exp else 0
    summary = [
        "# Detailed dataset audit",
        "",
        f"- Independent experiment IDs: **{n_exp}**",
        f"- Multimodal experiments (temperature + mechanical): **{n_multi}**",
        f"- Signal files audited: **{len(files_df)}**",
        "",
        "## Independent-unit warning",
        "",
        "Time samples or sliding windows from the same experiment are not independent experimental units. ",
        "Any downstream evaluation must split by `experiment_id` before windowing or generative-model fitting.",
        "",
        "## Files",
        "",
        "- `experiment_inventory.csv`: one row per independent experiment.",
        "- `signal_file_audit.csv`: one row per recorded signal file.",
    ]
    (args.out / "detailed_dataset_audit.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(exp_df.to_string(index=False))


if __name__ == "__main__":
    main()
