#!/usr/bin/env python3
"""Run one predeclared outer fold of the method-paper architecture benchmark."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np, pandas as pd, torch
from remotezip import RemoteZip
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url,load_one,EXPS,HEADERS
from evaluate_virtual_vehicle_time_to_vent_baseline import build_windows
from evaluate_method_paper_benchmark import channel_stats,norm_x,train_one,predict,metrics

NAMES=['cnn','lstm','gru','tcn','transformer','cnn_lstm','attention_gru','tcn_transformer','lath_net']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--heldout',required=True,choices=EXPS); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--window',type=int,default=120); ap.add_argument('--stride',type=int,default=10); ap.add_argument('--max-tau',type=float,default=600); ap.add_argument('--temp-cap',type=float,default=150)
    ap.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028]); args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
    data={}
    with RemoteZip(record_url(),headers=HEADERS,initial_buffer_size=1024*1024) as rz:
        for exp in EXPS:
            df,event,_=load_one(rz,exp); data[exp]=build_windows(df,event,args.window,args.stride,args.max_tau,args.temp_cap)
    hi=EXPS.index(args.heldout); val=EXPS[(hi+1)%len(EXPS)]; trains=[e for e in EXPS if e not in (args.heldout,val)]
    xtr=np.concatenate([data[e][0] for e in trains]); ytr=np.concatenate([data[e][1] for e in trains]); eidtr=np.concatenate([np.full(len(data[e][1]),EXPS.index(e),int) for e in trains])
    xv,yv,_=data[val]; xte,yte,meta=data[args.heldout]
    mu,sd=channel_stats(xtr); xtr=norm_x(xtr,mu,sd); xv=norm_x(xv,mu,sd); xte=norm_x(xte,mu,sd)
    ytrn=(ytr/args.max_tau).astype(np.float32); yvn=(yv/args.max_tau).astype(np.float32)
    rows=[]; preds=[]
    for name in NAMES:
        for seed0 in args.seeds:
            seed=seed0+100*hi
            model,vmae,epochs,params=train_one(name,xtr,ytrn,eidtr,xv,yvn,seed)
            pn,risk=predict(model,xte,torch.device('cpu')); p=np.clip(pn*args.max_tau,0,args.max_tau)
            rows.append({'heldout':args.heldout,'validation':val,'train_experiments':'+'.join(trains),'model':name,'seed':seed,'params':params,'epochs':epochs,'val_mae_s':vmae*args.max_tau,'n_train':len(ytr),'n_test':len(yte),**metrics(yte,p)})
            for j,(yy,pp) in enumerate(zip(yte,p)): preds.append({'heldout':args.heldout,'validation':val,'model':name,'seed':seed,'index':j,'tau_true_s':float(yy),'tau_pred_s':float(pp)})
    pd.DataFrame(rows).to_csv(args.out,index=False); pd.DataFrame(preds).to_csv(args.out.with_name(args.out.stem+'_predictions.csv'),index=False)
    audit={'heldout':args.heldout,'validation':val,'train_experiments':trains,'train_windows':len(ytr),'validation_windows':len(yv),'test_windows':len(yte),'models':NAMES,'seeds':args.seeds,'normalization':'training experiments only'}
    args.out.with_suffix('.json').write_text(json.dumps(audit,indent=2),encoding='utf-8'); print(pd.DataFrame(rows).groupby('model').mae_s.mean().sort_values().to_string())
if __name__=='__main__': main()
