#!/usr/bin/env python3
"""Train a compact age+shape-stage conditional 1D DDPM.

This script performs generative augmentation only. Synthetic windows are never
counted as independent TR experiments. For any later downstream claim, the
whole generator must be re-fit inside the training portion of each
experiment-blocked fold.
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
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
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
        self.age_emb = nn.Embedding(2, 12)
        self.stage_emb = nn.Embedding(3, 12)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim + 24, 96), nn.SiLU(), nn.Linear(96, 96)
        )
        self.in_proj = nn.Conv1d(in_channels, hidden, 5, padding=2)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, 96, d) for d in (1, 2, 4, 8, 4, 2, 1)])
        self.out_norm = nn.GroupNorm(4, hidden)
        self.out_proj = nn.Conv1d(hidden, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, age: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        cond = self.cond_mlp(torch.cat([self.time(t), self.age_emb(age), self.stage_emb(stage)], dim=1))
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(F.silu(self.out_norm(h)))


class DDPM:
    def __init__(self, steps: int, device: torch.device):
        self.steps = steps
        self.device = device
        # Linear schedule is deliberately simple/reproducible for the CPU pilot.
        self.beta = torch.linspace(1e-4, 0.02, steps, device=device)
        self.alpha = 1.0 - self.beta
        self.abar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.abar[t][:, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, int, int], age: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.steps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps = model(x, t, age, stage)
            a = self.alpha[i]
            ab = self.abar[i]
            b = self.beta[i]
            mean = (x - ((1 - a) / torch.sqrt(1 - ab)) * eps) / torch.sqrt(a)
            x = mean + (torch.sqrt(b) * torch.randn_like(x) if i > 0 else 0.0)
        return x


def features(x: np.ndarray) -> np.ndarray:
    feats = []
    for sample in x:
        row = []
        for c in range(sample.shape[0]):
            s = sample[c]
            d = np.diff(s)
            ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if len(s) > 2 and np.std(s[:-1]) > 0 and np.std(s[1:]) > 0 else 0.0
            row += [float(np.mean(s)), float(np.std(s)), float(np.min(s)), float(np.max(s)),
                    float(np.mean(np.abs(d))), float(np.max(np.abs(d))) if len(d) else 0.0, ac1]
        cross = float(np.corrcoef(sample[0], sample[1])[0, 1]) if sample.shape[0] >= 2 and np.std(sample[0]) > 0 and np.std(sample[1]) > 0 else 0.0
        row.append(cross)
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def downsample_flat(x: np.ndarray, points: int = 32) -> np.ndarray:
    idx = np.linspace(0, x.shape[-1] - 1, points).round().astype(int)
    return x[:, :, idx].reshape(len(x), -1)


def nearest_dist(a: np.ndarray, b: np.ndarray, self_compare: bool = False) -> np.ndarray:
    out = np.full(len(a), np.inf, dtype=np.float32)
    for i in range(0, len(a), 128):
        aa = a[i:i + 128]
        d = np.sqrt(((aa[:, None, :] - b[None, :, :]) ** 2).mean(axis=2))
        if self_compare:
            for local, global_i in enumerate(range(i, min(i + 128, len(a)))):
                if global_i < d.shape[1]:
                    d[local, global_i] = np.inf
        out[i:i + len(aa)] = d.min(axis=1)
    return out


def grouped_quality_gate(real: np.ndarray, synth: np.ndarray,
                         real_age: np.ndarray, real_stage: np.ndarray,
                         synth_age: np.ndarray, synth_stage: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, dict]:
    accepted = np.zeros(len(synth), dtype=bool)
    rows = []
    group_thresholds: dict[str, dict] = {}
    fs_all = features(synth)
    real_flat_all = downsample_flat(real)
    synth_flat_all = downsample_flat(synth)

    for age in (0, 1):
        for stage in (0, 1, 2):
            si = np.where((synth_age == age) & (synth_stage == stage))[0]
            if len(si) == 0:
                continue
            ri = np.where((real_age == age) & (real_stage == stage))[0]
            if len(ri) < 8:
                ri = np.where(real_age == age)[0]
            if len(ri) < 8:
                ri = np.arange(len(real))

            ref = real[ri]
            fr = features(ref)
            mu = np.mean(fr, axis=0)
            sd = np.maximum(np.std(fr, axis=0), 0.05)
            z = np.max(np.abs((fs_all[si] - mu) / sd), axis=1)

            rr = nearest_dist(downsample_flat(ref), downsample_flat(ref), self_compare=True)
            rr = rr[np.isfinite(rr)]
            sr = nearest_dist(synth_flat_all[si], real_flat_all[ri], self_compare=False)
            low = max(float(np.percentile(rr, 10) * 0.15), 1e-4)
            # Allow meaningful novelty while keeping samples near the real manifold.
            high = max(float(np.percentile(rr, 95) * 6.0), 0.35)
            max_abs = np.max(np.abs(synth[si]), axis=(1, 2))
            ok = (z <= 6.0) & (sr >= low) & (sr <= high) & (max_abs <= 4.0)
            accepted[si] = ok
            key = f"age{age}_stage{stage}"
            group_thresholds[key] = {
                "n_real_reference": int(len(ri)), "feature_z_max": 6.0,
                "nearest_real_min": low, "nearest_real_max": high,
                "real_real_nn_median": float(np.median(rr)), "max_abs_normalized": 4.0,
            }
            for local, cand in enumerate(si):
                rows.append({
                    "candidate": int(cand), "age": int(age), "stage": int(stage),
                    "accepted": int(ok[local]), "feature_z_max": float(z[local]),
                    "nearest_real_distance": float(sr[local]), "max_abs_normalized": float(max_abs[local]),
                })

    return accepted, pd.DataFrame(rows).sort_values("candidate"), group_thresholds


def sample_conditions(age: np.ndarray, stage: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pairs, counts = np.unique(np.stack([age, stage], axis=1), axis=0, return_counts=True)
    probs = counts / counts.sum()
    idx = rng.choice(len(pairs), size=n, p=probs)
    sampled = pairs[idx]
    # Ensure every observed condition is represented when candidate budget permits.
    for i, pair in enumerate(pairs[: min(len(pairs), n)]):
        sampled[i] = pair
    return sampled[:, 0].astype(np.int64), sampled[:, 1].astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/diffusion_pilot"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--diffusion-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input)
    x_np = data["x"].astype(np.float32)
    age_np = data["age"].astype(np.int64)
    stage_np = data["stage"].astype(np.int64)
    center = data["center"].astype(np.float32)
    scale = data["scale"].astype(np.float32)
    exp_ids = data["experiment_id"].astype(str)
    if len(x_np) < 8:
        raise RuntimeError("Not enough windows for diffusion pilot")

    x = torch.from_numpy(x_np).to(device)
    age = torch.from_numpy(age_np).to(device)
    stage = torch.from_numpy(stage_np).to(device)
    model = ConditionalDenoiser(in_channels=x.shape[1], hidden=args.hidden).to(device)
    ddpm = DDPM(args.diffusion_steps, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(x), (min(args.batch_size, len(x)),), device=device)
        xb, ab, sb = x[idx], age[idx], stage[idx]
        t = torch.randint(0, args.diffusion_steps, (len(idx),), device=device)
        noise = torch.randn_like(xb)
        pred = model(ddpm.q_sample(xb, t, noise), t, ab, sb)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
            print(f"step={step} loss={loss.item():.6f}")

    synth_age_np, synth_stage_np = sample_conditions(age_np, stage_np, args.candidates, rng)
    synth_age = torch.from_numpy(synth_age_np).long().to(device)
    synth_stage = torch.from_numpy(synth_stage_np).long().to(device)
    model.eval(); chunks = []
    with torch.no_grad():
        for i in range(0, args.candidates, 64):
            a = synth_age[i:i + 64]; s = synth_stage[i:i + 64]
            chunks.append(ddpm.sample(model, (len(a), x.shape[1], x.shape[2]), a, s).cpu().numpy())
    synth = np.concatenate(chunks, axis=0).astype(np.float32)

    accepted, qtable, thresholds = grouped_quality_gate(x_np, synth, age_np, stage_np, synth_age_np, synth_stage_np)
    synth_acc, age_acc, stage_acc = synth[accepted], synth_age_np[accepted], synth_stage_np[accepted]
    raw_all = synth * scale[None, :, None] + center[None, :, None]
    raw_acc = raw_all[accepted]

    # Keep candidate set for audit, accepted set for actual augmentation.
    np.savez_compressed(args.out / "synthetic_candidates.npz", x_normalized=synth, x_raw=raw_all.astype(np.float32), age=synth_age_np, stage=synth_stage_np)
    np.savez_compressed(args.out / "synthetic_windows.npz", x_normalized=synth_acc, x_raw=raw_acc.astype(np.float32), age=age_acc, stage=stage_acc, center=center, scale=scale)
    qtable.to_csv(args.out / "synthetic_quality.csv", index=False)
    pd.DataFrame(history).to_csv(args.out / "training_history.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "diffusion_steps": args.diffusion_steps, "center": center.tolist(), "scale": scale.tolist()}, args.out / "pilot_model.pt")

    group_counts = qtable.groupby(["age", "stage"])["accepted"].agg(["count", "sum"]).reset_index().to_dict(orient="records")
    report = {
        "input_file": str(args.input), "device": str(device), "real_windows": int(len(x_np)),
        "independent_experiment_ids": sorted(set(exp_ids.tolist())), "training_steps": args.steps,
        "candidate_synthetic_windows": int(len(synth)), "accepted_synthetic_windows": int(accepted.sum()),
        "acceptance_rate": float(accepted.mean()), "condition_group_counts": group_counts,
        "quality_thresholds": thresholds,
        "important_caveat": "Synthetic windows are augmentation samples, not new independent TR experiments.",
    }
    (args.out / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "pilot_report.md").write_text(
        "# Age + shape-stage conditional diffusion pilot\n\n"
        f"- Real training windows: **{report['real_windows']}**\n"
        f"- Independent experiment IDs used: **{', '.join(report['independent_experiment_ids'])}**\n"
        f"- Candidate synthetic windows: **{report['candidate_synthetic_windows']}**\n"
        f"- Accepted after grouped quality gate: **{report['accepted_synthetic_windows']}** ({100*report['acceptance_rate']:.1f}%)\n"
        f"- Device: **{report['device']}**\n\n"
        "Stage labels are coarse temperature-shape labels (`pre_rapid`, `rapid_peak`, `post_peak`), not venting ground truth.\n\n"
        "Accepted windows are training augmentation only and must not be counted as additional independent experiments.\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
