#!/usr/bin/env python3
"""Real-anchored conditional diffusion augmentation for very small TR datasets.

Instead of sampling a complete trajectory window from pure Gaussian noise, this
variant starts from a *real training window*, perturbs it to an intermediate
noise level, and denoises it back under the same aging/stage condition. This is
closer to the actual goal of data augmentation in an n≈few-experiment regime:
create plausible local variants without pretending to invent new experimental
regimes.

Synthetic windows remain training augmentation only. Every later downstream
validation fold must fit this generator using training experiments only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=t.device) * -scale)
        x = t.float()[:, None] * freq[None, :]
        emb = torch.cat([x.sin(), x.cos()], dim=1)
        return F.pad(emb, (0, max(self.dim - emb.shape[1], 0)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.cond = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond(cond)[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ConditionalDenoiser(nn.Module):
    def __init__(self, in_channels: int = 2, hidden: int = 32, time_dim: int = 64):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.age_emb = nn.Embedding(2, 16)
        self.stage_emb = nn.Embedding(3, 16)
        self.cond_mlp = nn.Sequential(nn.Linear(time_dim + 32, 128), nn.SiLU(), nn.Linear(128, 128))
        self.in_proj = nn.Conv1d(in_channels, hidden, 5, padding=2)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, 128, d) for d in (1,2,4,8,16,8,4,2,1)])
        self.out_norm = nn.GroupNorm(4, hidden)
        self.out_proj = nn.Conv1d(hidden, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, age: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        cond = self.cond_mlp(torch.cat([self.time(t), self.age_emb(age), self.stage_emb(stage)], dim=1))
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h, cond)
        return self.out_proj(F.silu(self.out_norm(h)))


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    abar = torch.cos(((x / steps + s) / (1 + s)) * math.pi * 0.5) ** 2
    abar = abar / abar[0]
    return (1.0 - abar[1:] / abar[:-1]).clamp(1e-5, 0.999).float()


class DDPM:
    def __init__(self, steps: int, device: torch.device):
        self.steps = steps; self.device = device
        self.beta = cosine_beta_schedule(steps).to(device)
        self.alpha = 1.0 - self.beta
        self.abar = torch.cumprod(self.alpha, dim=0)
        prev = torch.cat([torch.ones(1, device=device), self.abar[:-1]])
        self.posterior_var = self.beta * (1.0 - prev) / (1.0 - self.abar)
        self.posterior_var[0] = 1e-20

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.abar[t][:, None, None]
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    @torch.no_grad()
    def denoise_from(self, model: nn.Module, x_t: torch.Tensor, start_t: int,
                     age: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        x = x_t
        for i in reversed(range(start_t + 1)):
            t = torch.full((len(x),), i, device=x.device, dtype=torch.long)
            eps = model(x, t, age, stage)
            a = self.alpha[i]; ab = self.abar[i]
            mean = (x - (self.beta[i] / torch.sqrt(1.0 - ab)) * eps) / torch.sqrt(a)
            if i > 0:
                x = mean + torch.sqrt(self.posterior_var[i]) * torch.randn_like(x)
            else:
                x = mean
        return x


def features(x: np.ndarray) -> np.ndarray:
    rows=[]
    for sample in x:
        row=[]
        for c in range(sample.shape[0]):
            s=sample[c]; d=np.diff(s)
            ac1=float(np.corrcoef(s[:-1],s[1:])[0,1]) if len(s)>2 and np.std(s[:-1])>0 and np.std(s[1:])>0 else 0.0
            row += [float(np.mean(s)),float(np.std(s)),float(np.min(s)),float(np.max(s)),
                    float(np.mean(np.abs(d))),float(np.max(np.abs(d))) if len(d) else 0.0,ac1]
        cross=float(np.corrcoef(sample[0],sample[1])[0,1]) if sample.shape[0]>=2 and np.std(sample[0])>0 and np.std(sample[1])>0 else 0.0
        row.append(cross); rows.append(row)
    return np.asarray(rows,dtype=np.float32)


def downsample_flat(x: np.ndarray, points: int=32) -> np.ndarray:
    idx=np.linspace(0,x.shape[-1]-1,points).round().astype(int)
    return x[:,:,idx].reshape(len(x),-1)


def nearest_dist(a: np.ndarray,b: np.ndarray,self_compare: bool=False)->np.ndarray:
    out=np.full(len(a),np.inf,dtype=np.float32)
    for i in range(0,len(a),128):
        aa=a[i:i+128]; d=np.sqrt(((aa[:,None,:]-b[None,:,:])**2).mean(axis=2))
        if self_compare:
            for local,gi in enumerate(range(i,min(i+len(aa),len(a)))):
                if gi<d.shape[1]: d[local,gi]=np.inf
        out[i:i+len(aa)]=d.min(axis=1)
    return out


def balanced_indices(age: np.ndarray,stage: np.ndarray,n: int,rng: np.random.Generator)->np.ndarray:
    pairs=np.unique(np.stack([age,stage],axis=1),axis=0)
    chosen=rng.integers(0,len(pairs),size=n); out=np.empty(n,dtype=np.int64)
    for j,p in enumerate(chosen):
        a,s=pairs[p]; pool=np.flatnonzero((age==a)&(stage==s)); out[j]=rng.choice(pool)
    return out


def build_anchored_candidates(x: np.ndarray,age: np.ndarray,stage: np.ndarray,n: int,
                              t_min: int,t_max: int,rng: np.random.Generator):
    pairs=np.unique(np.stack([age,stage],axis=1),axis=0)
    reps=int(np.ceil(n/len(pairs))); pair_seq=np.tile(pairs,(reps,1))[:n].copy(); rng.shuffle(pair_seq)
    anchor_idx=[]; starts=[]; ages=[]; stages=[]
    for a,s in pair_seq:
        pool=np.flatnonzero((age==a)&(stage==s))
        anchor_idx.append(int(rng.choice(pool)))
        starts.append(int(rng.integers(t_min,t_max+1)))
        ages.append(int(a)); stages.append(int(s))
    return np.asarray(anchor_idx),np.asarray(starts),np.asarray(ages),np.asarray(stages)


def quality_gate(real: np.ndarray,synth: np.ndarray,real_age: np.ndarray,real_stage: np.ndarray,
                 synth_age: np.ndarray,synth_stage: np.ndarray,anchor_idx: np.ndarray,start_t: np.ndarray):
    accepted=np.zeros(len(synth),dtype=bool); rows=[]; thresholds={}
    fs=features(synth); real_flat=downsample_flat(real); syn_flat=downsample_flat(synth)
    for age in (0,1):
        for stage in (0,1,2):
            si=np.flatnonzero((synth_age==age)&(synth_stage==stage))
            if len(si)==0: continue
            ri_exact=np.flatnonzero((real_age==age)&(real_stage==stage)); ri=ri_exact
            if len(ri)<3: ri=np.flatnonzero(real_age==age)
            if len(ri)<3: ri=np.arange(len(real))
            fr=features(real[ri]); mu=fr.mean(axis=0); sd=np.maximum(fr.std(axis=0),0.05)
            z=np.max(np.abs((fs[si]-mu)/sd),axis=1)
            rr=nearest_dist(downsample_flat(real[ri]),downsample_flat(real[ri]),True); rr=rr[np.isfinite(rr)]
            sr=nearest_dist(syn_flat[si],real_flat[ri],False)
            rr_med=float(np.median(rr)) if len(rr) else 0.03; rr95=float(np.percentile(rr,95)) if len(rr) else 0.15
            # Overlapping windows make rr_med tiny. Use it only as a memorization floor;
            # feature consistency and a moderate absolute ceiling govern plausibility.
            low=max(rr_med*0.20,0.002)
            high=min(max(rr95*5.0,rr_med*12.0,0.20),0.80)
            max_abs=np.max(np.abs(synth[si]),axis=(1,2))
            pass_feature=z<=5.0; pass_low=sr>=low; pass_high=sr<=high; pass_amp=max_abs<=3.5
            ok=pass_feature&pass_low&pass_high&pass_amp; accepted[si]=ok
            key=f"age{age}_stage{stage}"; thresholds[key]={
                "n_real_exact_condition":int(len(ri_exact)),"n_real_reference":int(len(ri)),
                "feature_z_max":5.0,"nearest_real_min":low,"nearest_real_max":high,
                "real_real_nn_median":rr_med,"real_real_nn_p95":rr95,"max_abs_normalized":3.5}
            for local,cand in enumerate(si):
                rows.append({"candidate":int(cand),"age":int(age),"stage":int(stage),"anchor_index":int(anchor_idx[cand]),
                             "noise_start_t":int(start_t[cand]),"accepted":int(ok[local]),
                             "pass_feature":int(pass_feature[local]),"pass_memorization_floor":int(pass_low[local]),
                             "pass_ood_ceiling":int(pass_high[local]),"pass_amplitude":int(pass_amp[local]),
                             "feature_z_max":float(z[local]),"nearest_real_distance":float(sr[local]),
                             "max_abs_normalized":float(max_abs[local])})
    return accepted,pd.DataFrame(rows).sort_values("candidate"),thresholds


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=1200); ap.add_argument("--diffusion-steps",type=int,default=100)
    ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--hidden",type=int,default=32); ap.add_argument("--lr",type=float,default=1.5e-3)
    ap.add_argument("--candidates",type=int,default=360); ap.add_argument("--noise-min",type=float,default=0.20); ap.add_argument("--noise-max",type=float,default=0.50)
    ap.add_argument("--seed",type=int,default=2026); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); args.out.mkdir(parents=True,exist_ok=True)
    data=np.load(args.input); x_np=data["x"].astype(np.float32); age_np=data["age"].astype(np.int64); stage_np=data["stage"].astype(np.int64)
    center=data["center"].astype(np.float32); scale=data["scale"].astype(np.float32); exp_ids=data["experiment_id"].astype(str)
    x=torch.from_numpy(x_np).to(device); age=torch.from_numpy(age_np).to(device); stage=torch.from_numpy(stage_np).to(device)
    model=ConditionalDenoiser(in_channels=x.shape[1],hidden=args.hidden).to(device); ddpm=DDPM(args.diffusion_steps,device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); history=[]; model.train()
    for step in range(1,args.steps+1):
        ids=balanced_indices(age_np,stage_np,args.batch_size,rng); idx=torch.from_numpy(ids).long().to(device)
        xb,ab,sb=x[idx],age[idx],stage[idx]; t=torch.randint(0,args.diffusion_steps,(len(idx),),device=device); noise=torch.randn_like(xb)
        pred=model(ddpm.q_sample(xb,t,noise),t,ab,sb); loss=F.mse_loss(pred,noise)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%50==0 or step==args.steps:
            history.append({"step":step,"loss":float(loss.detach().cpu())}); print(f"step={step} loss={loss.item():.6f}")

    tmin=max(1,int(round(args.noise_min*(args.diffusion_steps-1)))); tmax=min(args.diffusion_steps-2,int(round(args.noise_max*(args.diffusion_steps-1))))
    anchor_idx,start_ts,syn_age_np,syn_stage_np=build_anchored_candidates(x_np,age_np,stage_np,args.candidates,tmin,tmax,rng)
    synth=np.empty((args.candidates,x_np.shape[1],x_np.shape[2]),dtype=np.float32); model.eval()
    # Batch candidates by identical start timestep so reverse loops can be shared.
    with torch.no_grad():
        for st in sorted(set(start_ts.tolist())):
            ids=np.flatnonzero(start_ts==st)
            for k in range(0,len(ids),48):
                sel=ids[k:k+48]; anchor=torch.from_numpy(x_np[anchor_idx[sel]]).to(device)
                a=torch.from_numpy(syn_age_np[sel]).long().to(device); s=torch.from_numpy(syn_stage_np[sel]).long().to(device)
                tt=torch.full((len(sel),),int(st),device=device,dtype=torch.long); xt=ddpm.q_sample(anchor,tt,torch.randn_like(anchor))
                synth[sel]=ddpm.denoise_from(model,xt,int(st),a,s).cpu().numpy().astype(np.float32)

    accepted,qtable,thresholds=quality_gate(x_np,synth,age_np,stage_np,syn_age_np,syn_stage_np,anchor_idx,start_ts)
    raw=synth*scale[None,:,None]+center[None,:,None]
    np.savez_compressed(args.out/"synthetic_candidates.npz",x_normalized=synth,x_raw=raw.astype(np.float32),age=syn_age_np,stage=syn_stage_np,anchor_index=anchor_idx,noise_start_t=start_ts)
    np.savez_compressed(args.out/"synthetic_windows.npz",x_normalized=synth[accepted],x_raw=raw[accepted].astype(np.float32),age=syn_age_np[accepted],stage=syn_stage_np[accepted],anchor_index=anchor_idx[accepted],noise_start_t=start_ts[accepted],center=center,scale=scale)
    qtable.to_csv(args.out/"synthetic_quality.csv",index=False); pd.DataFrame(history).to_csv(args.out/"training_history.csv",index=False)
    torch.save({"state_dict":model.state_dict(),"hidden":args.hidden,"diffusion_steps":args.diffusion_steps,"schedule":"cosine","generation":"real_anchored","center":center.tolist(),"scale":scale.tolist()},args.out/"model.pt")
    groups=qtable.groupby(["age","stage"])["accepted"].agg(["count","sum"]).reset_index(); report={
        "version":"v3_real_anchored_cosine","device":str(device),"real_windows":int(len(x_np)),"independent_experiment_ids":sorted(set(exp_ids.tolist())),
        "training_steps":args.steps,"diffusion_steps":args.diffusion_steps,"noise_t_range":[tmin,tmax],"candidate_synthetic_windows":int(len(synth)),
        "accepted_synthetic_windows":int(accepted.sum()),"acceptance_rate":float(accepted.mean()),"condition_group_counts":groups.to_dict(orient="records"),
        "quality_thresholds":thresholds,"important_caveat":"Real-anchored synthetic windows are local augmentation variants, not new independent TR experiments."}
    (args.out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); (args.out/"report.md").write_text(
        "# Diffusion augmentation v3 — real-anchored conditional perturbation\n\n"
        f"- Real windows: **{len(x_np)}** from **{', '.join(report['independent_experiment_ids'])}**\n"
        f"- Noise-start range: **t={tmin}–{tmax} / {args.diffusion_steps}**\n"
        f"- Synthetic candidates: **{len(synth)}**\n- Accepted: **{accepted.sum()} ({100*accepted.mean():.1f}%)**\n\n"
        "Generation is anchored to real training windows, then condition-preserving diffusion perturbation/denoising creates local variants.\n\n"
        "These samples are training augmentation only.\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
