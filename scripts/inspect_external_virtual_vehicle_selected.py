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

import numpy as np
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
    links=z.get('links') or {}; url=links.get('content') or links.get('self')
    if not url: raise RuntimeError(f"No content/self download URL in Zenodo file links: {sorted(links)}")
    return rec,url


def normalize(v):
    if pd.isna(v): return ""
    s=str(v).replace('\n',' ').replace('\r',' ').strip()
    return re.sub(r'\s+',' ',s)[:500]


def exp_from_name(name: str)->str:
    m=re.search(r'TS0330([A-H])',name,re.I)
    return f"TS0330{m.group(1).upper()}" if m else ""


def as_float(v):
    try:
        x=float(v); return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/external_virtual_vehicle_21700/selected_inspection'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    rec,url=record_and_url()
    structures=[]; hits=[]; selected=[]; readme_text=''; order_events=[]; channel_schema=[]; channel_stats=[]; report_rows=[]

    with RemoteZip(url,headers=HEADERS,initial_buffer_size=1024*1024) as rz:
        infos=[z for z in rz.infolist() if not z.is_dir()]; names=[z.filename for z in infos]
        selected=[n for n in names if n=='readme.txt' or n=='report_BAK_N21700CG-50_60SOC_TS0330.xlsx' or
                  (n.lower().endswith('.xlsx') and ('/export_' in n or '/order_' in n or '_mpo_channel_' in n))]
        for name in selected:
            b=rz.read(name)
            if name=='readme.txt':
                readme_text=b.decode('utf-8',errors='replace'); continue
            xf=pd.ExcelFile(io.BytesIO(b),engine='openpyxl')
            for sheet in xf.sheet_names:
                df=pd.read_excel(io.BytesIO(b),sheet_name=sheet,header=None,engine='openpyxl')
                structures.append({'file':name,'sheet':sheet,'rows':int(df.shape[0]),'cols':int(df.shape[1]),'nonempty_cells':int(df.notna().sum().sum()),'bytes_selected':len(b)})
                limit=min(len(df),5000)
                for ridx in range(limit):
                    vals=[normalize(v) for v in df.iloc[ridx].tolist()]; joined=' | '.join(v for v in vals if v); low=joined.lower()
                    matched=sorted({k.strip() for k in KEYWORDS if k in low})
                    if matched: hits.append({'file':name,'sheet':sheet,'row_index':int(ridx),'keywords':'+'.join(matched),'row_text':joined[:3000]})

                exp=exp_from_name(name)
                lowname=name.lower()
                # order_*.xlsx is a compact event table. Row 1 is the header in
                # these workbooks; retain all subsequent non-empty event rows.
                if '/order_' in lowname and df.shape[0]>=3 and df.shape[1]>=4:
                    for ridx in range(2,len(df)):
                        vals=df.iloc[ridx,:4].tolist()
                        if all(pd.isna(v) for v in vals): continue
                        order_events.append({'experiment':exp,'source_row':ridx,
                                             'time_abs_s':as_float(vals[0]),'time_rel_s':as_float(vals[1]),
                                             'event_type':normalize(vals[2]),'caption':normalize(vals[3])})

                # export_*.xlsx contains a two-row channel dictionary followed by
                # compressed time-series samples. Summarize schema/ranges only.
                if '/export_' in lowname and df.shape[0]>=3:
                    tech=[normalize(v) for v in df.iloc[0].tolist()]
                    semantic=[normalize(v) for v in df.iloc[1].tolist()]
                    num=df.iloc[2:].apply(pd.to_numeric,errors='coerce')
                    time=num.iloc[:,0].dropna().to_numpy(dtype=float) if num.shape[1] else np.array([])
                    dt=float(np.median(np.diff(time))) if len(time)>1 else np.nan
                    for j in range(df.shape[1]):
                        channel_schema.append({'experiment':exp,'column_index':j,'technical_name':tech[j] if j<len(tech) else '',
                                               'description':semantic[j] if j<len(semantic) else ''})
                        x=num.iloc[:,j].dropna().to_numpy(dtype=float)
                        channel_stats.append({'experiment':exp,'column_index':j,'technical_name':tech[j] if j<len(tech) else '',
                                              'description':semantic[j] if j<len(semantic) else '',
                                              'finite_rows':int(len(x)),'min':float(np.min(x)) if len(x) else np.nan,
                                              'max':float(np.max(x)) if len(x) else np.nan,'median':float(np.median(x)) if len(x) else np.nan,
                                              'shared_time_dt_s':dt})

                # The root report has row 0 field names and row 1 units. Retain a
                # compact subset of event/safety fields for each experiment row.
                if name=='report_BAK_N21700CG-50_60SOC_TS0330.xlsx' and sheet=='MAIN' and df.shape[0]>=3:
                    headers=[normalize(v) for v in df.iloc[0].tolist()]
                    units=[normalize(v) for v in df.iloc[1].tolist()]
                    keep=[]
                    for j,h in enumerate(headers):
                        hl=h.lower()
                        if j==0 or any(k in hl for k in ('soc','vent','runaway','pressure','temperature','cellcaset','voltage_before','max_cell','experiment designation','cell_type')):
                            if h: keep.append(j)
                    for ridx in range(2,len(df)):
                        designation=normalize(df.iloc[ridx,0]) if df.shape[1] else ''
                        if not designation: continue
                        for j in keep:
                            report_rows.append({'experiment':designation,'field':headers[j],'unit':units[j] if j<len(units) else '',
                                                'value':normalize(df.iloc[ridx,j]),'source_row':ridx,'source_col':j})

    pd.DataFrame(structures).to_csv(args.out/'workbook_structure.csv',index=False)
    pd.DataFrame(hits,columns=['file','sheet','row_index','keywords','row_text']).to_csv(args.out/'keyword_rows.csv',index=False)
    pd.DataFrame(order_events,columns=['experiment','source_row','time_abs_s','time_rel_s','event_type','caption']).to_csv(args.out/'order_events.csv',index=False)
    pd.DataFrame(channel_schema).to_csv(args.out/'export_channel_schema.csv',index=False)
    pd.DataFrame(channel_stats).to_csv(args.out/'export_channel_stats.csv',index=False)
    pd.DataFrame(report_rows,columns=['experiment','field','unit','value','source_row','source_col']).to_csv(args.out/'report_selected_fields.csv',index=False)
    (args.out/'readme.txt').write_text(readme_text,encoding='utf-8')
    summary={'selected_files':len(selected),'workbook_sheets':len(structures),'keyword_rows':len(hits),'event_rows':len(order_events),
             'export_channels':len(channel_schema),'report_selected_cells':len(report_rows),'selected_paths':selected}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Selected-file inspection — Virtual Vehicle TS0330','',
           'Only compact XLSX/readme entries were range-extracted from the 4.6 GiB Zenodo archive; photos/videos and bulk files were not downloaded.','',
           f'- Selected archive entries: **{len(selected)}**',f'- Workbook sheets inspected: **{len(structures)}**',
           f'- Event rows recovered from `order_*.xlsx`: **{len(order_events)}**',f'- Export channel definitions mapped: **{len(channel_schema)}**',
           f'- Selected report event/safety cells mapped: **{len(report_rows)}**','',
           'Derived metadata are in `order_events.csv`, `export_channel_schema.csv`, `export_channel_stats.csv`, and `report_selected_fields.csv`. Raw workbook contents are not redistributed.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))
    if order_events: print('\nEvent rows:\n'+pd.DataFrame(order_events).to_string(index=False))
    if report_rows: print('\nSelected report rows (first 160):\n'+pd.DataFrame(report_rows).head(160).to_string(index=False))

if __name__=='__main__': main()
