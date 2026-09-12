#!/usr/bin/env python3
"""Metadata/range audit of the 5 GB Virtual Vehicle 21700 TR package.

The Zenodo archive is deliberately NOT downloaded in full. We query the public
Zenodo API, then use HTTP Range requests through RemoteZip to inspect the ZIP
central directory. Only file names/sizes and derived summaries are committed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

RECORD_ID = "18849418"
API = f"https://zenodo.org/api/records/{RECORD_ID}"
HEADERS = {"User-Agent": "BatteryDiffusion-research-audit/1.0"}


def get_record() -> dict:
    r = requests.get(API, timeout=60, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def choose_zip(record: dict) -> dict:
    files = record.get("files") or []
    for f in files:
        key = str(f.get("key") or f.get("filename") or "")
        if key.lower().endswith(".zip"):
            return f
    if files:
        return files[0]
    raise RuntimeError("Zenodo record exposes no files")


def content_url(file_rec: dict) -> str:
    links = file_rec.get("links") or {}
    url = links.get("content") or links.get("self")
    if not url:
        raise RuntimeError("No downloadable URL in Zenodo file metadata")
    return str(url)


def classify(name: str) -> str:
    low = name.lower()
    if "raw" in low:
        return "raw_data"
    if any(x in low for x in ("report", "result", "summary")):
        return "report_or_result"
    if any(x in low for x in ("photo", "foto", "image", "picture")):
        return "image"
    if any(x in low for x in ("video", "movie")):
        return "video"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/external_virtual_vehicle_21700"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    record = get_record()
    f = choose_zip(record)
    url = content_url(f)
    meta = record.get("metadata") or {}
    audit = {
        "record_id": RECORD_ID,
        "doi": meta.get("doi") or record.get("doi"),
        "title": meta.get("title"),
        "publication_date": meta.get("publication_date"),
        "archive_key": f.get("key"),
        "archive_bytes": f.get("size"),
        "archive_checksum": f.get("checksum"),
        "central_directory_status": "not_attempted",
    }

    # Check whether the storage endpoint advertises byte-range support.
    try:
        hr = requests.head(url, timeout=60, allow_redirects=True, headers=HEADERS)
        audit["head_status"] = hr.status_code
        audit["accept_ranges"] = hr.headers.get("accept-ranges")
        audit["content_length_head"] = hr.headers.get("content-length")
    except Exception as e:
        audit["head_error"] = repr(e)

    rows = []
    range_error = None
    try:
        from remotezip import RemoteZip
        with RemoteZip(url, headers=HEADERS, initial_buffer_size=1024 * 1024) as rz:
            for z in rz.infolist():
                if z.is_dir():
                    continue
                p = Path(z.filename)
                rows.append({
                    "path": z.filename,
                    "basename": p.name,
                    "extension": p.suffix.lower(),
                    "uncompressed_bytes": int(z.file_size),
                    "compressed_bytes": int(z.compress_size),
                    "category": classify(z.filename),
                    "depth": len(p.parts),
                })
        audit["central_directory_status"] = "success"
    except Exception as e:
        range_error = repr(e)
        audit["central_directory_status"] = "failed"
        audit["central_directory_error"] = range_error

    manifest = pd.DataFrame(rows, columns=["path", "basename", "extension", "uncompressed_bytes", "compressed_bytes", "category", "depth"])
    manifest.to_csv(args.out / "archive_manifest.csv", index=False)

    if len(manifest):
        ext = (manifest.groupby("extension", dropna=False)
               .agg(files=("path", "size"), uncompressed_bytes=("uncompressed_bytes", "sum"), compressed_bytes=("compressed_bytes", "sum"))
               .reset_index().sort_values("uncompressed_bytes", ascending=False))
        ext.to_csv(args.out / "extension_summary.csv", index=False)
        cand_mask = manifest.extension.isin([".csv", ".txt", ".xlsx", ".xls", ".mat", ".tdms", ".json", ".h5", ".hdf5", ".parquet", ".dat"])
        cand = manifest[cand_mask].copy().sort_values(["category", "uncompressed_bytes"], ascending=[True, False])
        cand.to_csv(args.out / "candidate_data_files.csv", index=False)
        cats = manifest.category.value_counts().to_dict()
    else:
        ext = pd.DataFrame(columns=["extension", "files", "uncompressed_bytes", "compressed_bytes"])
        ext.to_csv(args.out / "extension_summary.csv", index=False)
        cand = pd.DataFrame(columns=manifest.columns)
        cand.to_csv(args.out / "candidate_data_files.csv", index=False)
        cats = {}

    audit["archive_entries"] = int(len(manifest))
    audit["candidate_data_files"] = int(len(cand))
    audit["category_counts"] = {str(k): int(v) for k, v in cats.items()}
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    size_gb = (float(f.get("size") or 0) / (1024**3))
    lines = [
        "# Virtual Vehicle BAK N21700CG-50 external package audit",
        "",
        "Zenodo record: **18849418**; package contains eight repeated BAK N21700CG-50 thermal-runaway tests at 60% SOC.",
        "",
        f"- Archive: `{f.get('key')}` (~{size_gb:.2f} GiB)",
        f"- ZIP central-directory audit: **{audit['central_directory_status']}**",
        f"- Archive file entries indexed without full download: **{len(manifest)}**",
        f"- Candidate structured/raw data files: **{len(cand)}**",
        "- The 5 GB raw package is not copied into this repository.",
        "",
        "## Why this matters",
        "",
        "This source adds **eight independent real destructive experiments**, which is more valuable for generalization claims than creating additional overlapping or synthetic windows. The Zenodo description states that the package includes cell temperature, vent-gas temperature, pressure, gas release and event information such as CID, burst-plate and thermal runaway.",
        "",
        "## Next integration gate",
        "",
        "Use `candidate_data_files.csv` to locate the smallest machine-readable raw/event files. Only those files should be selectively range-extracted for schema/event-time mapping; photos and videos are not needed for the first predictive protocol.",
    ]
    if range_error:
        lines += ["", "Range-based ZIP listing failed; `audit.json` records the storage response/error. The record-level metadata remains valid, but no archive-internal claims are made."]
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if len(cand):
        print("\nCandidate data files (first 80):\n", cand.head(80).to_string(index=False))


if __name__ == "__main__":
    main()
