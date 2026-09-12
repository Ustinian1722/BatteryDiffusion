#!/usr/bin/env python3
"""Download-to-runner and audit Zenodo 13981390 without redistributing raw files.

The public record contains two MAT files (NMC111 and NMC811) with temperature
and pressure measurements from thermal-abuse experiments. Raw bytes are kept in
a temporary directory during CI; only schema/statistical audit outputs are
committed to this repository.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat

FILES={
    'NMC111':'https://zenodo.org/records/13981390/files/NMC111.mat?download=1',
    'NMC811':'https://zenodo.org/records/13981390/files/NMC811.mat?download=1',
}


def summarize_obj(name,obj):
    arr=np.asarray(obj)
    rec={'key':name,'shape':list(arr.shape),'dtype':str(arr.dtype),'size':int(arr.size)}
    if np.issubdtype(arr.dtype,np.number) and arr.size:
        flat=arr.astype(float).ravel(); finite=flat[np.isfinite(flat)]
        if len(finite): rec.update(min=float(finite.min()),max=float(finite.max()),mean=float(finite.mean()),std=float(finite.std()),finite=int(len(finite)))
    return rec


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/external_zenodo_13981390'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; file_summary=[]
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for label,url in FILES.items():
            p=td/f'{label}.mat'; r=requests.get(url,timeout=120); r.raise_for_status(); p.write_bytes(r.content)
            rec={'experiment_label':label,'bytes':p.stat().st_size,'url':url}
            try:
                mat=loadmat(p,squeeze_me=True,struct_as_record=False)
                keys=[k for k in mat if not k.startswith('__')]
                rec['mat_format']='scipy_loadmat'; rec['top_level_keys']=keys
                for k in keys:
                    s=summarize_obj(k,mat[k]); s['experiment_label']=label; all_rows.append(s)
            except NotImplementedError:
                import h5py
                with h5py.File(p,'r') as f:
                    keys=list(f.keys()); rec['mat_format']='hdf5'; rec['top_level_keys']=keys
                    for k in keys:
                        obj=f[k]
                        try: arr=np.asarray(obj); s=summarize_obj(k,arr)
                        except Exception: s={'key':k,'shape':None,'dtype':'group','size':None}
                        s['experiment_label']=label; all_rows.append(s)
            file_summary.append(rec)
    pd.DataFrame(all_rows).to_csv(args.out/'mat_schema.csv',index=False)
    (args.out/'file_summary.json').write_text(json.dumps(file_summary,indent=2),encoding='utf-8')
    lines=['# External dataset audit — Zenodo 13981390','',
           'Record: **Modeling Thermal Runaway Mechanisms and Pressure Dynamics in Prismatic Lithium-Ion Batteries**.','',
           'The CI audit downloads the two public MAT files to a temporary runner directory and commits only schema/statistical metadata; raw external files are not redistributed here.','',
           '## Files','']
    for r in file_summary:
        lines += [f"### {r['experiment_label']}",f"- bytes: {r['bytes']:,}",f"- MAT reader: `{r['mat_format']}`",f"- top-level keys: `{r['top_level_keys']}`",'']
    lines += ['## Intended role','',
              'These two experiments are candidate **cross-form-factor thermo-pressure external tests**. They should not be pooled blindly with the Hanyang 2170 cohort because cell format/chemistry/apparatus differ. The next step is to identify exact time/temperature/pressure arrays and event timing from the MAT schema before defining any transfer experiment.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
