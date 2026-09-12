#!/usr/bin/env python3
"""Train a compact conditional 1D DDPM and generate a pilot synthetic corpus.

This is a *pilot augmentation model*, not a downstream predictive experiment.
It is intentionally compact enough for CPU GitHub Actions. Final application
claims must re-fit the generator inside each experiment-blocked validation fold.
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
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


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
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim + 16, 96), nn.SiLU(), nn.Linear(96, 96)
        )
        self.in_proj = nn.Conv1d(in_channels, hidden, 5, padding=2)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, 96, d) for d in (1, 2, 4, 8, 4, 2, 1)
        ])
        self.out_norm = nn.GroupNorm(4, hidden)
        self.out_proj = nn.Conv1d(hidden, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        cond = self.cond_mlp(torch.cat([self.time(t), self.age_emb(age)], dim=1))
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(F.silu(self.out_norm(h)))


class DDPM:
    def __init__(self, steps: int, device: torch.device):
        self.steps = steps
        self.device = device
        beta = torch.linspace(1e-4, 0.02, steps, device=device)
        alpha = 1.0 - beta
        abar = torch.cumprod(alpha, dim=0)
        self.beta = beta
        self.alpha = alpha
        self.abar = abar

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.abar[t][:, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, int, int], age: torch.Tensor) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.steps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps = model(x, t, age)
            a = self.alpha[i]
            ab = self.abar[i]
            b = self.beta[i]
            mean = (x - ((1 - a) / torch.sqrt(1 - ab)) * eps) / torch.sqrt(a)
            if i > 0:
                x = mean + torch.sqrt(b) * torch.randn_like(x)
            else:
                x = mean
        return x


def features(x: np.ndarray) -> np.ndarray:
    """Window descriptors for real-vs-synthetic quality gating."""
    # x [N,C,L]
    feats = []
    for sample in x:
        row = []
        for c in range(sample.shape[0]):
            s = sample[c]
            d = np.diff(s)
            row += [
                float(np.mean(s)), float(np.std(s)), float(np.min(s)), float(np.max(s)),
                float(np.mean(np.abs(d))), float(np.max(np.abs(d))) if len(d) else 0.0,
                float(np.corrcoef(s[:-1], s[1:])[0, 1]) if len(s) > 2 and np.std(s[:-1]) > 0 and np.std(s[1:]) > 0 else 0.0,
            ]
        if sample.shape[0] >= 2 and np.std(sample[0]) > 0 and np.std(sample[1]) > 0:
            row.append(float(np.corrcoef(sample[0], sample[1])[0, 1]))
        else:
            row.append(0.0)
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def downsample_flat(x: np.ndarray, points: int = 32) -> np.ndarray:
    idx = np.linspace(0, x.shape[-1] - 1, points).round().astype(int)
    return x[:, :, idx].reshape(len(x), -1)


def nearest_dist(a: np.ndarray, b: np.ndarray, self_compare: bool = False) -> np.ndarray:
    out = np.full(len(a), np.inf, dtype=np.float32)
    block = 128
    for i in range(0, len(a), block):
        aa = a[i:i+block]
        d = ((aa[:, None, :] - b[None, :, :]) ** 2).mean(axis=2) ** 0.5
        if self_compare:
            for local, global_i in enumerate(range(i, min(i + block, len(a)))):
                if global_i < d.shape[1]:
                    d[local, global_i] = np.inf
        out[i:i+len(aa)] = d.min(axis=1)
    return out


def quality_gate(real: np.ndarray, synth: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, dict]:
    fr = features(real)
    fs = features(synth)
    med = np.median(fr, axis=0)
    q25 = np.percentile(fr, 25, axis=0)
    q75 = np.percentile(fr, 75, axis=0)
    iqr = np.maximum(q75 - q25, 1e-4)
    robust_z = np.max(np.abs((fs - med) / iqr), axis=1)

    rr = nearest_dist(downsample_flat(real), downsample_flat(real), self_compare=True)
    sr = nearest_dist(downsample_flat(synth), downsample_flat(real), self_compare=False)
    finite_rr = rr[np.isfinite(rr)]
    low = max(float(np.percentile(finite_rr, 10) * 0.20), 1e-4)
    high = float(np.percentile(finite_rr, 95) * 2.5)

    # Broad normalized-domain guardrail: allows tail events but rejects explosions.
    max_abs = np.max(np.abs(synth), axis=(1, 2))
    accepted = (robust_z <= 8.0) & (sr >= low) & (sr <= high) & (max_abs <= 8.0)
    table = pd.DataFrame({
        "candidate": np.arange(len(synth)),
        "accepted": accepted.astype(int),
        "feature_robust_z_max": robust_z,
        "nearest_real_distance": sr,
        "max_abs_normalized": max_abs,
    })
    thresholds = {
        "feature_robust_z_max": 8.0,
        "nearest_real_distance_min": low,
        "nearest_real_distance_max": high,
        "max_abs_normalized": 8.0,
        "real_real_nn_median": float(np.median(finite_rr)),
    }
    return accepted, table, thresholds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/diffusion_pilot"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--diffusion-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input)
    x_np = data["x"].astype(np.float32)
    age_np = data["age"].astype(np.int64)
    center = data["center"].astype(np.float32)
    scale = data["scale"].astype(np.float32)
    exp_ids = data["experiment_id"].astype(str)
    if len(x_np) < 8:
        raise RuntimeError("Not enough windows for diffusion pilot")

    x = torch.from_numpy(x_np).to(device)
    age = torch.from_numpy(age_np).to(device)
    model = ConditionalDenoiser(in_channels=x.shape[1], hidden=args.hidden).to(device)
    ddpm = DDPM(args.diffusion_steps, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(x), (min(args.batch_size, len(x)),), device=device)
        xb = x[idx]
        ab = age[idx]
        t = torch.randint(0, args.diffusion_steps, (len(idx),), device=device)
        noise = torch.randn_like(xb)
        xt = ddpm.q_sample(xb, t, noise)
        pred = model(xt, t, ab)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
            print(f"step={step} loss={loss.item():.6f}")

    # Candidate conditioning mirrors the observed fresh/aged proportions while
    # guaranteeing both conditions appear when possible.
    probs = np.bincount(age_np, minlength=2).astype(float)
    probs = probs / probs.sum()
    synth_age_np = np.random.choice([0, 1], size=args.candidates, p=probs)
    if args.candidates >= 2:
        synth_age_np[0], synth_age_np[1] = 0, 1
    synth_age = torch.from_numpy(synth_age_np).long().to(device)

    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, args.candidates, 64):
            a = synth_age[i:i+64]
            chunks.append(ddpm.sample(model, (len(a), x.shape[1], x.shape[2]), a).cpu().numpy())
    synth = np.concatenate(chunks, axis=0).astype(np.float32)

    accepted, qtable, thresholds = quality_gate(x_np, synth)
    synth_acc = synth[accepted]
    age_acc = synth_age_np[accepted]
    raw_acc = synth_acc * scale[None, :, None] + center[None, :, None]

    np.savez_compressed(
        args.out / "synthetic_windows.npz",
        x_normalized=synth_acc,
        x_raw=raw_acc.astype(np.float32),
        age=age_acc.astype(np.int64),
        center=center,
        scale=scale,
    )
    qtable.to_csv(args.out / "synthetic_quality.csv", index=False)
    pd.DataFrame(history).to_csv(args.out / "training_history.csv", index=False)
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": args.hidden,
        "diffusion_steps": args.diffusion_steps,
        "center": center.tolist(),
        "scale": scale.tolist(),
    }, args.out / "pilot_model.pt")

    report = {
        "input_file": str(args.input),
        "device": str(device),
        "real_windows": int(len(x_np)),
        "independent_experiment_ids": sorted(set(exp_ids.tolist())),
        "training_steps": args.steps,
        "candidate_synthetic_windows": int(len(synth)),
        "accepted_synthetic_windows": int(accepted.sum()),
        "acceptance_rate": float(accepted.mean()),
        "fresh_accepted": int((age_acc == 0).sum()),
        "aged_accepted": int((age_acc == 1).sum()),
        "quality_thresholds": thresholds,
        "important_caveat": "Synthetic windows are augmentation samples, not new independent TR experiments.",
    }
    (args.out / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "pilot_report.md").write_text(
        "# Conditional diffusion pilot\n\n"
        f"- Real training windows: **{report['real_windows']}**\n"
        f"- Independent experiment IDs used: **{', '.join(report['independent_experiment_ids'])}**\n"
        f"- Candidate synthetic windows: **{report['candidate_synthetic_windows']}**\n"
        f"- Accepted after quality gate: **{report['accepted_synthetic_windows']}** "
        f"({100*report['acceptance_rate']:.1f}%)\n"
        f"- Device: **{report['device']}**\n\n"
        "These accepted windows are training augmentation only and must not be counted as additional independent experiments.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
