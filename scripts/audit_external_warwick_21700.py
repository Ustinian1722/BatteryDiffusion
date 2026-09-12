#!/usr/bin/env python3
"""Audit the public Warwick/Mendeley 21700 TR dataset without redistributing raw files.

Dataset: 10.17632/rgfhdhcd9k.1 (the Data in Brief article cites the same
record family as 10.17632/rgfhdhcd9k.2). Public files are discovered through
Mendeley Data's unauthenticated public file endpoint, downloaded only to a
temporary runner directory, and reduced to schema/statistical summaries.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat

DATASET_ID = "rgfhdhcd9k"
VERSION = 1
PUBLIC_API = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files?folder_id=root&version={VERSION}"
DATASET_PAGE = f"https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}"
# The interactive site and public API can reject obvious hosted-runner bot user
# agents even for CC-BY public records. Use ordinary browser negotiation while
# keeping the request read-only and unauthenticated.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": DATASET_PAGE,
}


def get_json(url: str):
    r = requests.get(url, timeout=60, headers=HEADERS)
    if r.ok:
        try:
            return r.json()
        except Exception:
            return None
    return None


def discover_files() -> tuple[list[dict], dict]:
    """Return public file records and an access audit."""
    audit = {"public_api": PUBLIC_API, "dataset_page": DATASET_PAGE}
    r = requests.get(PUBLIC_API, timeout=60, headers=HEADERS)
    audit["public_api_status"] = r.status_code
    audit["public_api_content_type"] = r.headers.get("content-type")
    try:
        data = r.json() if r.ok else None
    except Exception:
        data = None
    audit["public_api_payload_type"] = type(data).__name__ if data is not None else None

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
        details = row.get("content_details") or {}
        url = row.get("download_url") or details.get("download_url") or row.get("url")
        fid = row.get("id") or row.get("file_id")
        if not url and fid:
            url = f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}/files/{fid}/file_downloaded"
        if url:
            out.append({"name": str(name), "url": str(url), "id": str(fid or ""), "size": row.get("size")})
    if out:
        audit["discovery_method"] = "public_api"
        return out, audit

    # Fallback is diagnostic only. A failed page/API request is kept as an
    # access-layer result rather than interpreted as absence of public files.
    page_headers = dict(HEADERS)
    page_headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    page = requests.get(DATASET_PAGE, timeout=60, headers=page_headers)
    audit["dataset_page_status"] = page.status_code
    if page.ok:
        urls = sorted(set(re.findall(r'https://data\.mendeley\.com/public-files/datasets/[^"\\]+?/file_downloaded', page.text)))
        for i, url in enumerate(urls):
            out.append({"name": f"discovered_file_{i+1}", "url": url.replace("\\u002F", "/"), "id": "", "size": None})
        if out:
            audit["discovery_method"] = "page_embedded_url"
            return out, audit

    audit["discovery_method"] = "blocked_or_no_files"
    return [], audit


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


def write_source_mapping(out: Path) -> None:
    rows = [
        ("TestID", "-", "test identifier"),
        ("ExpTime", "s", "time base for pressure/voltage channels"),
        ("IntPre", "bar", "internal gas pressure"),
        ("CellVoltage", "V", "cell voltage"),
        ("ExpTimeTemp", "s", "temperature time base; 10 Hz acquisition"),
        ("MidIntTemp", "degC", "internal midpoint temperature"),
        ("MidSurfTemp", "degC", "midpoint surface temperature"),
        ("NegSurfTemp", "degC", "surface temperature 10 mm from negative terminal"),
        ("PosSurfTemp", "degC", "surface temperature 10 mm from positive terminal"),
        ("VentPos5mmAway", "degC", "vent temperature 5 mm from positive vent cap"),
        ("VentPos10mmAway", "degC", "vent temperature 10 mm from positive vent cap"),
    ]
    pd.DataFrame(rows, columns=["field", "unit", "source_description"]).to_csv(out / "source_supported_schema.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/external_warwick_21700"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    write_source_mapping(args.out)

    files, access = discover_files()
    (args.out / "access_audit.json").write_text(json.dumps(access, indent=2), encoding="utf-8")

    manifest = []
    schema_rows: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, meta in enumerate(files):
            r = requests.get(meta["url"], timeout=180, headers=HEADERS)
            r.raise_for_status()
            name = meta["name"]
            ext = infer_extension(name, r.headers.get("content-type"))
            if Path(name).suffix == "":
                name = f"file_{i+1}{ext}"
            local = td / Path(name).name
            local.write_bytes(r.content)
            manifest.append({
                "file_name": name,
                "bytes": len(r.content),
                "content_type": r.headers.get("content-type", ""),
                "source_file_id": meta.get("id", ""),
            })
            schema_rows.extend(inspect_file(local, name))

    pd.DataFrame(manifest, columns=["file_name", "bytes", "content_type", "source_file_id"]).to_csv(args.out / "file_manifest.csv", index=False)
    schema = pd.DataFrame(schema_rows, columns=["file", "key", "shape", "dtype", "size", "finite", "min", "max", "mean", "std"])
    schema.to_csv(args.out / "numeric_schema.csv", index=False)
    if len(schema):
        key_lower = schema["key"].astype(str).str.lower()
        candidate = schema[key_lower.str.contains("press|temp|volt|time|vent|gas", regex=True)].copy()
    else:
        candidate = schema.copy()
    candidate.to_csv(args.out / "candidate_signal_arrays.csv", index=False)

    status = "raw-file audit completed" if manifest else "raw-file discovery blocked/unavailable on CI; source-supported mapping only"
    lines = [
        "# Warwick/Mendeley 21700 external dataset audit",
        "",
        "Dataset record: **rgfhdhcd9k**, public version 1; the 2025 Data in Brief descriptor cites DOI **10.17632/rgfhdhcd9k.2** while linking to the version-1 public page.",
        "",
        f"- Status: **{status}**",
        f"- Public files discovered/downloaded: **{len(manifest)}**",
        f"- Numeric arrays/columns inspected directly: **{len(schema)}**",
        f"- Candidate thermo-pressure/time/voltage arrays directly inspected: **{len(candidate)}**",
        "- Raw external files are never committed by this audit.",
        "",
        "## Source-supported structure",
        "",
        "The open data descriptor reports three independent Sony VTC6A 21700 tests at 100% SOC, triggered by 40 W external heating. The processed MATLAB data contain internal pressure, voltage, internal/surface temperatures and two vent-temperature channels. The source reports a high-rate electrical/pressure time base and a 10 Hz temperature time base. Exact field names and units are recorded in `source_supported_schema.csv`.",
        "",
        "## Access note",
        "",
        "If `file_manifest.csv` is empty, that is an access-layer result rather than evidence that the dataset has no files. Mendeley may block hosted runner IPs. We therefore keep source-derived schema separate from raw-file-derived schema and do not fabricate file identifiers.",
        "",
        "## Intended use",
        "",
        "Once raw file access is resolved, these three tests are the highest-priority cylindrical external thermo-pressure cohort. Event landmarks will be frozen from source-supported vent stages / pressure landmarks before predictive evaluation.",
    ]
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nAccess audit:\n", json.dumps(access, indent=2))
    if manifest:
        print("\nManifest:\n", pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
