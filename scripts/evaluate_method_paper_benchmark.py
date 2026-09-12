#!/usr/bin/env python3
"""Leakage-safe method-paper benchmark for pre-vent time-to-event prediction.

The protocol is frozen in reports/method_paper_target_freeze_v1.md before model
scores are inspected. Each outer fold holds out one independent TS0330 test and
uses the next experiment cyclically as validation; the remaining six tests are
training only. All models receive exactly the same causal temperature+pressure
windows and training-only normalization.

Primary target: remaining time to package-provided first `vent_gas`, restricted
to 0 < tau <= 600 s, with 120 s causal context and max input T <= 150 C.

LATH-Net = Lag-Aware Thermo-pressure Hazard Network. Its auxiliary monotone
multi-horizon risk head and within-experiment progress ranking are training-time
regularizers; the headline comparison remains continuous TTV regression for all
models.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from remotezip import RemoteZip
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, EXPS, HEADERS
from evaluate_virtual_vehicle_time_to_vent_baseline import build_windows

HORIZONS = torch.tensor([60.0, 120.0, 180.0, 300.0, 450.0])


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def channel_stats(x: np.ndarray):
    mu=x.mean(axis=(0,2)).astype(np.float32); sd=x.std(axis=(0,2)).astype(np.float32)
    return mu,np.maximum(sd,1e-6)


def norm_x(x,mu,sd):
    return ((x-mu[None,:,None])/sd[None,:,None]).astype(np.float32)


class CNN(nn.Module):
    def __init__(self,c=2,h=32):
        super().__init__(); self.f=nn.Sequential(
            nn.Conv1d(c,h,5,padding=2),nn.GELU(),nn.Conv1d(h,h,5,padding=2),nn.GELU(),
            nn.AdaptiveAvgPool1d(1)); self.out=nn.Linear(h,1)
    def forward(self,x): return self.out(self.f(x).squeeze(-1)).squeeze(-1),None


class RNNReg(nn.Module):
    def __init__(self,kind='gru',c=2,h=32):
        super().__init__(); R=nn.GRU if kind=='gru' else nn.LSTM
        self.r=R(c,h,num_layers=2,batch_first=True,dropout=0.1); self.out=nn.Linear(h,1)
    def forward(self,x):
        y,_=self.r(x.transpose(1,2)); return self.out(y[:,-1]).squeeze(-1),None


class ResidualTCNBlock(nn.Module):
    def __init__(self,h,d):
        super().__init__(); pad=d*2
        self.c1=nn.Conv1d(h,h,3,padding=pad,dilation=d); self.c2=nn.Conv1d(h,h,3,padding=pad,dilation=d)
        self.n1=nn.GroupNorm(4,h); self.n2=nn.GroupNorm(4,h); self.act=nn.GELU()
        self.crop=pad
    def forward(self,x):
        y=self.c1(x); y=y[...,:x.shape[-1]]; y=self.act(self.n1(y))
        y=self.c2(y); y=y[...,:x.shape[-1]]; y=self.n2(y)
        return self.act(x+y)


class TCN(nn.Module):
    def __init__(self,c=2,h=32):
        super().__init__(); self.inp=nn.Conv1d(c,h,1); self.blocks=nn.Sequential(*[ResidualTCNBlock(h,d) for d in (1,2,4,8)]); self.out=nn.Linear(h,1)
    def forward(self,x):
        z=self.blocks(self.inp(x)); z=0.5*(z.mean(-1)+z.amax(-1)); return self.out(z).squeeze(-1),None


class TransformerReg(nn.Module):
    def __init__(self,c=2,d=32,layers=2):
        super().__init__(); self.proj=nn.Linear(c,d); self.pos=nn.Parameter(torch.zeros(1,120,d));
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=64,dropout=0.1,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(enc,num_layers=layers); self.out=nn.Linear(d,1)
    def forward(self,x):
        s=x.transpose(1,2); z=self.proj(s)+self.pos[:,:s.shape[1]]; z=self.enc(z).mean(1); return self.out(z).squeeze(-1),None


class CNNLSTM(nn.Module):
    def __init__(self,c=2,h=32):
        super().__init__(); self.conv=nn.Sequential(nn.Conv1d(c,h,5,padding=2),nn.GELU(),nn.Conv1d(h,h,3,padding=1),nn.GELU())
        self.r=nn.LSTM(h,h,batch_first=True); self.out=nn.Linear(h,1)
    def forward(self,x):
        z=self.conv(x).transpose(1,2); z,_=self.r(z); return self.out(z[:,-1]).squeeze(-1),None


class AttentionGRU(nn.Module):
    def __init__(self,c=2,h=32):
        super().__init__(); self.r=nn.GRU(c,h,batch_first=True); self.score=nn.Linear(h,1); self.out=nn.Linear(h,1)
    def forward(self,x):
        z,_=self.r(x.transpose(1,2)); a=torch.softmax(self.score(z).squeeze(-1),dim=1); pooled=(z*a.unsqueeze(-1)).sum(1)
        return self.out(pooled).squeeze(-1),None


class TCNTransformer(nn.Module):
    def __init__(self,c=2,h=32):
        super().__init__(); self.inp=nn.Conv1d(c,h,1); self.blocks=nn.Sequential(*[ResidualTCNBlock(h,d) for d in (1,2,4)])
        enc=nn.TransformerEncoderLayer(d_model=h,nhead=4,dim_feedforward=64,dropout=0.1,batch_first=True,activation='gelu')
        self.tr=nn.TransformerEncoder(enc,num_layers=1); self.out=nn.Linear(h,1)
    def forward(self,x):
        z=self.blocks(self.inp(x)).transpose(1,2); z=self.tr(z).mean(1); return self.out(z).squeeze(-1),None


class MultiScale1D(nn.Module):
    def __init__(self,h=24):
        super().__init__(); self.stem=nn.Conv1d(1,h,1)
        self.short=nn.Sequential(nn.Conv1d(h,h,3,padding=1),nn.GELU(),nn.Conv1d(h,h,3,padding=1),nn.GELU())
        self.long=nn.Sequential(nn.Conv1d(h,h,5,padding=8,dilation=4),nn.GELU(),nn.Conv1d(h,h,3,padding=8,dilation=8),nn.GELU())
        self.mix=nn.Conv1d(2*h,h,1)
    def forward(self,x):
        b=self.stem(x); s=self.short(b); l=self.long(b)[...,:x.shape[-1]]; return torch.nn.functional.gelu(self.mix(torch.cat([s,l],1)))


def shift_past(z: torch.Tensor, lag: int) -> torch.Tensor:
    if lag==0: return z
    out=torch.zeros_like(z); out[:,:,lag:]=z[:,:,:-lag]; return out


class LATHNet(nn.Module):
    """Lag-Aware Thermo-pressure Hazard Network."""
    def __init__(self,h=24,horizons=5):
        super().__init__(); self.tenc=MultiScale1D(h); self.penc=MultiScale1D(h); self.lags=(0,2,5,10,20,30)
        self.gate=nn.Conv1d(2*h,h,1); self.fuse=nn.Conv1d(4*h,2*h,1)
        self.post=nn.Sequential(ResidualTCNBlock(2*h,1),ResidualTCNBlock(2*h,2),ResidualTCNBlock(2*h,4))
        self.head=nn.Sequential(nn.Linear(4*h,2*h),nn.GELU(),nn.Dropout(0.1))
        self.tau=nn.Linear(2*h,1); self.risk=nn.Linear(2*h,horizons)
    def forward(self,x):
        t=self.tenc(x[:,0:1]); p=self.penc(x[:,1:2])
        shifted=[shift_past(p,l) for l in self.lags]
        scores=torch.stack([(t*s).mean(dim=(1,2)) for s in shifted],dim=1)
        w=torch.softmax(scores/math.sqrt(t.shape[1]),dim=1)
        pa=sum(w[:,i,None,None]*shifted[i] for i in range(len(shifted)))
        g=torch.sigmoid(self.gate(torch.cat([t,pa],1))); pg=g*pa
        z=torch.nn.functional.gelu(self.fuse(torch.cat([t,pg,t-pg,t*pg],1))); z=self.post(z)
        pooled=torch.cat([z.mean(-1),z.amax(-1)],dim=1); h=self.head(pooled)
        tau=self.tau(h).squeeze(-1)
        raw=self.risk(h); mono=torch.cat([raw[:,:1], raw[:,:1] + torch.cumsum(torch.nn.functional.softplus(raw[:,1:]),dim=1)],dim=1)
        return tau,mono


def build_model(name:str):
    return {
        'cnn':lambda:CNN(), 'lstm':lambda:RNNReg('lstm'), 'gru':lambda:RNNReg('gru'),
        'tcn':lambda:TCN(), 'transformer':lambda:TransformerReg(), 'cnn_lstm':lambda:CNNLSTM(),
        'attention_gru':lambda:AttentionGRU(), 'tcn_transformer':lambda:TCNTransformer(), 'lath_net':lambda:LATHNet(),
    }[name]()


def ranking_loss(pred,true,expid,margin=0.05):
    # pred/true are normalized by 600. Pair only windows from the same real experiment.
    pi=pred[:,None]; pj=pred[None,:]; ti=true[:,None]; tj=true[None,:]
    same=expid[:,None].eq(expid[None,:]); ordered=(tj-ti)>0.10  # >=60 s separation
    mask=same & ordered
    if not torch.any(mask): return pred.sum()*0.0
    return torch.relu(margin-(pj-pi))[mask].mean()


def loss_fn(name,pred,risk,y,expid):
    base=torch.nn.functional.smooth_l1_loss(pred,y,beta=0.05)
    if name!='lath_net': return base
    h=HORIZONS.to(y.device)/600.0; labels=(y[:,None]<=h[None,:]).float()
    bce=torch.nn.functional.binary_cross_entropy_with_logits(risk,labels)
    rank=ranking_loss(pred,y,expid)
    return base+0.20*bce+0.10*rank


def predict(model,x,device):
    model.eval(); out=[]; risks=[]
    with torch.no_grad():
        for i in range(0,len(x),128):
            p,r=model(torch.from_numpy(x[i:i+128]).to(device)); out.append(p.cpu().numpy())
            if r is not None: risks.append(torch.sigmoid(r).cpu().numpy())
    return np.concatenate(out), (np.concatenate(risks) if risks else None)


def train_one(name,xtr,ytr,eidtr,xv,yv,seed,max_epochs=220,patience=35):
    set_seed(seed); torch.set_num_threads(1); device=torch.device('cpu'); model=build_model(name).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max_epochs,eta_min=1e-5)
    ds=TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr.astype(np.float32)),torch.from_numpy(eidtr.astype(np.int64)))
    gen=torch.Generator().manual_seed(seed); loader=DataLoader(ds,batch_size=min(64,len(ds)),shuffle=True,generator=gen)
    best=None; best_mae=float('inf'); stale=0; history=[]
    for ep in range(max_epochs):
        model.train(); losses=[]
        for xb,yb,eb in loader:
            xb,yb,eb=xb.to(device),yb.to(device),eb.to(device); p,r=model(xb); loss=loss_fn(name,p,r,yb,eb)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
        sched.step(); pv,_=predict(model,xv,device); val_mae=float(np.mean(np.abs(pv-yv)))
        history.append((ep,float(np.mean(losses)),val_mae))
        if val_mae < best_mae-1e-5:
            best_mae=val_mae; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale+=1
        if stale>=patience: break
    model.load_state_dict(best); return model,best_mae,len(history),sum(p.numel() for p in model.parameters())


def metrics(y,p):
    return {'mae_s':float(mean_absolute_error(y,p)),'rmse_s':float(mean_squared_error(y,p)**0.5),
            'medae_s':float(median_absolute_error(y,p)),'r2':float(r2_score(y,p)),'bias_s':float(np.mean(p-y))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/method_paper_benchmark_v1'))
    ap.add_argument('--window',type=int,default=120); ap.add_argument('--stride',type=int,default=10); ap.add_argument('--max-tau',type=float,default=600); ap.add_argument('--temp-cap',type=float,default=150)
    ap.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028]); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if args.window!=120: raise ValueError('Transformer positional embedding is frozen for 120 s context in v1')
    data={}; support=[]
    with RemoteZip(record_url(),headers=HEADERS,initial_buffer_size=1024*1024) as rz:
        for ei,exp in enumerate(EXPS):
            df,event,_=load_one(rz,exp); x,y,m=build_windows(df,event,args.window,args.stride,args.max_tau,args.temp_cap)
            data[exp]=(x,y,m); support.append({'experiment':exp,'event_time_s':event,'windows':len(y),'tau_min_s':float(y.min()),'tau_max_s':float(y.max())})
    pd.DataFrame(support).to_csv(args.out/'support.csv',index=False)
    names=['cnn','lstm','gru','tcn','transformer','cnn_lstm','attention_gru','tcn_transformer','lath_net']
    rows=[]; preds=[]
    for hi,held in enumerate(EXPS):
        val=EXPS[(hi+1)%len(EXPS)]; trains=[e for e in EXPS if e not in (held,val)]
        xtr=np.concatenate([data[e][0] for e in trains]); ytr=np.concatenate([data[e][1] for e in trains]); eidtr=np.concatenate([np.full(len(data[e][1]),EXPS.index(e),int) for e in trains])
        xv,yv,_=data[val]; xte,yte,meta=data[held]
        mu,sd=channel_stats(xtr); xtr=norm_x(xtr,mu,sd); xv=norm_x(xv,mu,sd); xte=norm_x(xte,mu,sd)
        ytrn=(ytr/args.max_tau).astype(np.float32); yvn=(yv/args.max_tau).astype(np.float32)
        for name in names:
            for seed in args.seeds:
                model,vmae,epochs,params=train_one(name,xtr,ytrn,eidtr,xv,yvn,seed)
                pn,risk=predict(model,xte,torch.device('cpu')); p=np.clip(pn*args.max_tau,0,args.max_tau)
                rec={'heldout':held,'validation':val,'train_experiments':'+'.join(trains),'model':name,'seed':seed,'params':params,'epochs':epochs,'val_mae_s':vmae*args.max_tau,'n_train':len(ytr),'n_test':len(yte),**metrics(yte,p)}; rows.append(rec)
                for j,yy,pp in zip(range(len(yte)),yte,p): preds.append({'heldout':held,'validation':val,'model':name,'seed':seed,'index':j,'tau_true_s':float(yy),'tau_pred_s':float(pp)})
    df=pd.DataFrame(rows); df.to_csv(args.out/'fold_seed_metrics.csv',index=False); pd.DataFrame(preds).to_csv(args.out/'predictions.csv',index=False)
    fold=df.groupby(['heldout','model']).agg(mae_s=('mae_s','mean'),rmse_s=('rmse_s','mean'),medae_s=('medae_s','mean'),r2=('r2','mean'),params=('params','first')).reset_index(); fold.to_csv(args.out/'heldout_metrics.csv',index=False)
    macro=fold.groupby('model').agg(mae_mean_s=('mae_s','mean'),mae_worst_s=('mae_s','max'),rmse_mean_s=('rmse_s','mean'),medae_mean_s=('medae_s','mean'),r2_mean=('r2','mean'),params=('params','first')).reset_index().sort_values('mae_mean_s'); macro.to_csv(args.out/'macro_metrics.csv',index=False)
    lines=['# Method-paper benchmark v1','',f'Frozen task: {args.window} s causal T+P context; 0<tau<={args.max_tau:g} s to first package `vent_gas`; max input T<={args.temp_cap:g} C; stride {args.stride} s. Outer test is one entire experiment; validation is the next experiment cyclically; six remaining experiments train the model. Three frozen seeds are averaged within each held-out experiment.','', '| model | mean MAE | worst MAE | mean RMSE | mean MedAE | mean R2 | params |','|---|---:|---:|---:|---:|---:|---:|']
    for _,r in macro.iterrows(): lines.append(f"| {r.model} | {r.mae_mean_s:.1f}s | {r.mae_worst_s:.1f}s | {r.rmse_mean_s:.1f}s | {r.medae_mean_s:.1f}s | {r.r2_mean:.3f} | {int(r.params):,} |")
    lines += ['', '## Guardrail','', 'This is the architecture benchmark. It is **real-only** for every model: RA-CDiff is intentionally excluded so architecture contribution cannot be confounded with data augmentation. The independent statistical unit remains the held-out destructive experiment (n=8).']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); (args.out/'protocol.json').write_text(json.dumps(vars(args),default=str,indent=2),encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
