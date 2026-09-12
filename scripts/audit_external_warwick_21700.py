#!/usr/bin/env python3
"""Audit the public Warwick/Mendeley 21700 TR dataset without redistributing raw files.

Dataset: 10.17632/rgfhdhcd9k.1 (same dataset later cited as .2 in the
Data in Brief article). The script discovers public file metadata, downloads
files only into a temporary directory, inspects MAT/CSV/XLSX structure, and
commits schema/statistical summaries only.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat

DATASET_ID = "rgfhdhcd9k"
VERSION = 1
PUBLIC_API = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/versions/{VERSION}/files"
DATASET_PAGE = f"https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}"


def get_json(url: str):
    r = requests.get(url, timeout=60, headers={"User-Agent": "BatteryDiffusion-research-audit/1.0"})
    if r.ok:
        try:
            return r.json()
        except Exception:
            return None
    return None


def discover_files() -> list[dict]:
    data = get_json(PUBLIC_API)
    out: list[dict] = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("files") or data.get("results") or data.get("data") or []
    else:
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("filename") or row.get("file_name") or row.get("name") or "unknown"
        url = row.get("download_url") or row.get("content_details", {}).get("download_url") or row.get("url")
        fid = row.get("id") or row.get("file_id")
        if not url and fid:
            url = f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}/files/{fid}/file_downloaded"
        if url:
            out.append({"name": str(name), "url": str(url), "id": str(fid or ""), "size": row.get("size")})
    if out:
        return out

    # Fallback: public page source sometimes embeds public-files URLs.
    r = requests.get(DATASET_PAGE, timeout=60, headers={"User-Agent": "BatteryDiffusion-research-audit/1.0"})
    r.raise_for_status()
    urls = sorted(set(re.findall(r'https://data\.mendeley\.com/public-files/datasets/[^"\\]+?/file_downloaded', r.text)))
    for i, url in enumerate(urls):
        out.append({"name": f"discovered_file_{i+1}", "url": url.replace("\\u002F", "/"), "id": "", "size": None})
    return out


def numeric_summary(label: str, key: str, arr: np.ndarray) -> dict | None:
    a = np.asarray(arr)
    if a.dtype.kind not in "iufb" or a.size == 0:
        return None
    flat = a.astype(float, copy=False).ravel()
    finite = np.isfinite(flat)
    vals = flat[finite]
    rec = {
        "file": label,
        "key": key,
        "shape": json.dumps(list(a.shape)),
        "dtype": str(a.dtype),
        "size": int(a.size),
        "finite": int(finite.sum()),
    }
    if len(vals):
        rec.update({"min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean()), "std": float(vals.std())})
    return rec


def inspect_file(path: Path, display_name: str) -> list[dict]:
    ext = path.suffix.lower()
    rows: list[dict] = []
    if ext == ".mat":
        try:
            mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            return [{"file": display_name, "key": "__mat_v7_3__", "shape": "[]", "dtype": "hdf5", "size": 0, "finite": 0}]
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            rec = numeric_summary(display_name, k, np.asarray(v))
            if rec:
                rows.append(rec)
    elif ext == ".csv":
        df = pd.read_csv(path)
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").to_numpy()
            rec = numeric_summary(display_name, str(c), s)
            if rec:
                rows.append(rec)
    elif ext in {".xlsx", ".xls"}:
        book = pd.ExcelFile(path)
        for sheet in book.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            for c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").to_numpy()
                rec = numeric_summary(display_name, f"{sheet}:{c}", s)
                if rec:
                    rows.append(rec)
    return rows


def infer_extension(name: str, content_type: str | None) -> str:
    ext = Path(name).suffix.lower()
    if ext:
        return ext
    ct = (content_type or "").lower()
    if "matlab" in ct or "mat" in ct:
        return ".mat"
    if "csv" in ct:
        return ".csv"
    if "excel" in ct or "spreadsheet" in ct:
        return ".xlsx"
    return ".bin"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/external_warwick_21700"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = discover_files()
    if not files:
        raise RuntimeError("No public files discovered from Mendeley dataset")

    manifest = []
    schema_rows: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, meta in enumerate(files):
            r = requests.get(meta["url"], timeout=180, headers={"User-Agent": "BatteryDiffusion-research-audit/1.0"})
            r.raise_for_status()
            name = meta["name"]
            ext = infer_extension(name, r.headers.get("content-type"))
            if Path(name).suffix == "":
                name = f"file_{i+1}{ext}"
            local = td / name
            local.write_bytes(r.content)
            manifest.append({
                "file_name": name,
                "bytes": len(r.content),
                "content_type": r.headers.get("content-type", ""),
                "source_file_id": meta.get("id", ""),
            })
            schema_rows.extend(inspect_file(local, name))

    pd.DataFrame(manifest).to_csv(args.out / "file_manifest.csv", index=False)
    schema = pd.DataFrame(schema_rows)
    schema.to_csv(args.out / "numeric_schema.csv", index=False)

    if len(schema):
        key_lower = schema["key"].astype(str).str.lower()
        candidate = schema[key_lower.str.contains("press|temp|volt|time|vent|gas", regex=True)].copy()
    else:
        candidate = schema.copy()
    candidate.to_csv(args.out / "candidate_signal_arrays.csv", index=False)

    lines = [
        "# Warwick/Mendeley 21700 external dataset audit",
        "",
        "Dataset DOI: **10.17632/rgfhdhcd9k.1** (the 2025 Data in Brief article cites the same dataset family and later version).",
        "",
        f"- Public files discovered: **{len(manifest)}**",
        f"- Numeric arrays/columns inspected: **{len(schema)}**",
        f"- Candidate thermo-pressure/time/voltage arrays: **{len(candidate)}**",
        "- Raw external files were downloaded only to the temporary CI runner and are not redistributed in this repository.",
        "",
        "## Intended use",
        "",
        "This dataset contains three independent Sony VTC6A 21700 thermal-runaway tests with internal gas pressure and multiple temperature measurements. It is being evaluated as the highest-priority external cylindrical-cell dataset for event-aligned thermo-pressure validation.",
        "",
        "The next step is to map each file to an experiment, identify the exact time/pressure/temperature channels and freeze source-supported soft-vent/first-vent landmarks before any predictive evaluation.",
    ]
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nManifest:\n", pd.DataFrame(manifest).to_string(index=False))
    if len(candidate):
        print("\nCandidate arrays:\n", candidate.to_string(index=False))


if __name__ == "__main__":
    main()
