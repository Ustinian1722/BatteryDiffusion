#!/usr/bin/env python3
"""Selectively inspect small machine-readable files inside Zenodo 18849418.

Uses HTTP Range access to the 4.6 GiB ZIP and reads only the report/readme and
small XLSX files needed for schema/event mapping. Raw files are never committed.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import pandas as pd
import requests
from remotezip import RemoteZip

RECORD_ID="18849418"
API=f"https://zenodo.org/api/records/{RECORD_ID}"
HEADERS={"User-Agent":"BatteryDiffusion-research-audit/1.0"}
KEYWORDS=("time","temp","temperature","pressure","press","thermal runaway","runaway","tr ","cid","burst","vent","gas","soc","voltage","event")


def record_and_url():
    r=requests.get(API,timeout=60,headers=HEADERS); r.raise_for_status(); rec=r.json()
    files=rec.get('files') or []; z=next(f for f in files if str(f.get('key','')).lower().endswith('.zip'))
    return rec,(z.get('links') or {}).get('content')


def normalize(v):
    if pd.isna(v): return ""
    s=str(v).replace('\n',' ').replace('\r',' ').strip()
    return re.sub(r'\s+',' ',s)[:500]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/external_virtual_vehicle_21700/selected_inspection'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    rec,url=record_and_url()
    structures=[]; hits=[]; selected=[]; readme_text=''
    with RemoteZip(url,headers=HEADERS,initial_buffer_size=1024*1024) as rz:
        infos=[z for z in rz.infolist() if not z.is_dir()]
        names=[z.filename for z in infos]
        # Read root report/readme plus all compact export/order workbooks. Include
        # the two ~4 MB MPo workbooks because together they are still tiny vs archive.
        selected=[n for n in names if n=='readme.txt' or n=='report_BAK_N21700CG-50_60SOC_TS0330.xlsx' or
                  (n.lower().endswith('.xlsx') and ('/export_' in n or '/order_' in n or '_mpo_channel_' in n))]
        for name in selected:
            b=rz.read(name)
            if name=='readme.txt':
                readme_text=b.decode('utf-8',errors='replace')
                continue
            xf=pd.ExcelFile(io.BytesIO(b),engine='openpyxl')
            for sheet in xf.sheet_names:
                # Header-free read keeps arbitrary report sheets intact.
                df=pd.read_excel(io.BytesIO(b),sheet_name=sheet,header=None,engine='openpyxl')
                nonempty=int(df.notna().sum().sum())
                structures.append({'file':name,'sheet':sheet,'rows':int(df.shape[0]),'cols':int(df.shape[1]),'nonempty_cells':nonempty,'bytes_selected':len(b)})
                # Keyword rows only; store a compact row rendering for event/schema mapping.
                limit=min(len(df),5000)
                for ridx in range(limit):
                    vals=[normalize(v) for v in df.iloc[ridx].tolist()]
                    joined=' | '.join(v for v in vals if v)
                    low=joined.lower()
                    matched=sorted({k.strip() for k in KEYWORDS if k in low})
                    if matched:
                        hits.append({'file':name,'sheet':sheet,'row_index':int(ridx),'keywords':'+'.join(matched),'row_text':joined[:3000]})
    pd.DataFrame(structures).to_csv(args.out/'workbook_structure.csv',index=False)
    pd.DataFrame(hits,columns=['file','sheet','row_index','keywords','row_text']).to_csv(args.out/'keyword_rows.csv',index=False)
    (args.out/'readme.txt').write_text(readme_text,encoding='utf-8')
    summary={'selected_files':len(selected),'workbook_sheets':len(structures),'keyword_rows':len(hits),'selected_paths':selected}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Selected-file inspection — Virtual Vehicle TS0330','',
           'Only compact XLSX/readme entries were range-extracted from the 4.6 GiB Zenodo archive; photos/videos and bulk files were not downloaded.','',
           f'- Selected archive entries: **{len(selected)}**',f'- Workbook sheets inspected: **{len(structures)}**',f'- Keyword-bearing rows retained for schema/event mapping: **{len(hits)}**','',
           'See `workbook_structure.csv`, `keyword_rows.csv`, and the package `readme.txt`. Raw workbook contents are not redistributed.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print('\nREADME:\n'+readme_text[:10000])
    if hits:
        print('\nKeyword rows (first 120):')
        print(pd.DataFrame(hits).head(120).to_string(index=False))

if __name__=='__main__': main()
