#!/usr/bin/env python3
"""rtgraphs – research-grade latency visualiser for *rtbench* outputs
--------------------------------------------------------------------
Figures written to <results>/figures (or --outdir):

* timeline_mean.png  – mean ±95 % CI latency, 1 s bins
* heatmap.png        – mean latency heat-map (time × workload)
* ecdf.png           – ECDF of all latencies (log-x)
* boxplot.png        – run-to-run mean-latency variability
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy import stats

# ── tame noisy SciPy/NumPy warnings ───────────────────────────────────────
warnings.filterwarnings("ignore", message="Mean of empty slice")
try:
    from scipy.stats import SmallSampleWarning
    warnings.filterwarnings("ignore", category=SmallSampleWarning)
except ImportError:
    pass

# ── Matplotlib defaults (paper-friendly & colour-blind) ───────────────────
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
PALETTE = plt.get_cmap("tab10").colors

# ╔══════════════ helpers ══════════════════════════════════════════════════╗
def _ts(ts: str):
    try:
        return datetime.strptime(ts, "%a, %d %b %Y %H:%M:%S %z")
    except Exception:
        return None


def _json(p: Path):
    try:
        with p.open() as fp:
            return json.load(fp)
    except Exception:
        return None


def _expand(hist: Dict[str, int]) -> np.ndarray:
    if not hist:
        return np.empty(0, np.float32)
    bins, cnts = zip(*((int(k), int(v)) for k, v in hist.items()))
    return np.repeat(np.array(bins, np.float32), np.array(cnts, np.int64))


def latencies(run: Path) -> np.ndarray:
    """Return flat µs array from cyclictest.json (preferred) or histogram.txt."""
    js = run / "cyclictest.json"
    if js.exists():
        j = _json(js)
        if j and "thread" in j:
            arrs = [_expand(t.get("histogram", {})) for t in j["thread"].values()]
            if any(a.size for a in arrs):
                return np.concatenate(arrs)

    # fallback – histogram.txt lines “<bin>: <count>”
    txt = run / "histogram.txt"
    if not txt.exists():
        return np.empty(0, np.float32)

    bins, cnts = [], []
    for ln in txt.read_text().splitlines():
        if ":" in ln:
            try:
                b, c = ln.split(":", 1)
                bins.append(float(b)); cnts.append(int(c))
            except ValueError:
                pass
    return (
        np.repeat(np.array(bins, np.float32), np.array(cnts, np.int64))
        if bins
        else np.empty(0, np.float32)
    )


def summarise(run: Path):
    js = run / "cyclictest.json"
    if not js.exists():
        return None
    j = _json(js)
    if not j:
        return None

    start, end = _ts(j.get("start_time", "")), _ts(j.get("end_time", ""))
    if not (start and end):
        return None
    dur = (end - start).total_seconds()

    sam = latencies(run)
    if sam.size == 0:
        return None

    return {
        "duration": dur,
        "samples": sam,
        "interval": dur / sam.size,
        "mean": float(np.mean(sam)),
    }


def load_results(root: Path):
    out: List[Dict] = []
    for bench in sorted(d for d in root.iterdir() if d.is_dir()):
        runs = [summarise(r) for r in sorted(bench.iterdir()) if r.is_dir()]
        runs = [r for r in runs if r]
        if runs:
            out.append({"name": bench.name, "runs": runs})
    return out


# ╔══════════════ statistics ═══════════════════════════════════════════════╗
def mean_ci(arr: np.ndarray, axis=0, conf=0.95):
    μ = np.nanmean(arr, axis=axis)
    sem = stats.sem(arr, axis=axis, nan_policy="omit", ddof=0)
    n = arr.shape[axis]
    h = sem * stats.t.ppf((1 + conf) / 2.0, n - 1) if n > 1 else np.zeros_like(μ)
    return μ, h


# ╔══════════════ plots ════════════════════════════════════════════════════╗
def plot_timeline(summary, out: Path):
    if not summary:
        return
    max_sec = int(np.ceil(max(r["duration"] for b in summary for r in b["runs"])))
    secs = np.arange(max_sec)

    plt.figure(figsize=(4.8, 3.0))
    for j, ent in enumerate(summary):
        per_run = np.full((len(ent["runs"]), max_sec), np.nan)
        for i, r in enumerate(ent["runs"]):
            sam = r["samples"]
            times = np.arange(sam.size) * r["interval"]
            for s in secs:
                sel = (times >= s) & (times < s + 1)
                if sel.any():
                    per_run[i, s] = np.mean(sam[sel])
            del sam, times
        μ, h = mean_ci(per_run)
        col = PALETTE[j % 10]
        plt.plot(secs, μ, lw=1.3, color=col, label=ent["name"])
        plt.fill_between(secs, μ - h, μ + h, color=col, alpha=0.25)
        del per_run
    gc.collect()

    plt.xlabel("Time [s]")
    plt.ylabel("Latency [µs]")
    plt.title("Mean scheduling latency (±95 % CI)")
    plt.grid(True, linestyle=":")

    # ── legend outside ────────────────────────────────────────────────────
    plt.subplots_adjust(right=0.78)
    plt.legend(title="Workload",
               bbox_to_anchor=(1.02, 1),
               loc="upper left",
               borderaxespad=0.0)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_box(summary, out: Path):
    if not summary:
        return
    data = [[r["mean"] for r in ent["runs"]] for ent in summary]
    labels = [ent["name"] for ent in summary]

    plt.figure(figsize=(4.2, 3.0))
    bp = plt.boxplot(data, patch_artist=True, widths=0.6,
                     medianprops=dict(color="black", lw=1.2))
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)

    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Run-mean latency [µs]")
    plt.title("Run-to-run variability per workload")
    plt.grid(axis="y", linestyle=":")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_ecdf(summary, out: Path):
    if not summary:
        return
    plt.figure(figsize=(4.8, 3.0))
    for j, ent in enumerate(summary):
        max_bin = int(max(r["samples"].max() for r in ent["runs"]))
        hist = np.zeros(max_bin + 1, dtype=np.int64)
        for r in ent["runs"]:
            sam = r["samples"].astype(int)
            hist += np.bincount(sam, minlength=hist.size)
            del sam
        cdf = np.cumsum(hist, dtype=np.float64)
        cdf /= cdf[-1]
        xs = np.arange(hist.size)
        plt.step(xs, cdf, where="post", color=PALETTE[j % 10], label=ent["name"])
        del hist, cdf, xs
    gc.collect()

    plt.xscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("Latency [µs] (log)")
    plt.ylabel("Cumulative probability")
    plt.title("ECDF of all latencies")
    plt.grid(True, which="both", linestyle=":")

    # ── legend outside ────────────────────────────────────────────────────
    plt.subplots_adjust(right=0.78)
    plt.legend(title="Workload",
               bbox_to_anchor=(1.02, 1),
               loc="upper left",
               borderaxespad=0.0)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_heatmap(summary, out: Path):
    if not summary:
        return
    max_sec = int(np.ceil(max(r["duration"] for b in summary for r in b["runs"])))
    secs = np.arange(max_sec)
    grid = np.full((len(summary), max_sec), np.nan)

    for i, ent in enumerate(summary):
        per_run = np.full((len(ent["runs"]), max_sec), np.nan)
        for j, r in enumerate(ent["runs"]):
            times = np.arange(r["samples"].size) * r["interval"]
            for s in secs:
                sel = (times >= s) & (times < s + 1)
                if sel.any():
                    per_run[j, s] = np.mean(r["samples"][sel])
        grid[i] = np.nanmean(per_run, axis=0)

    plt.figure(figsize=(4.8, 3.0))
    im = plt.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, max_sec, -0.5, len(summary) - 0.5])
    plt.colorbar(im, label="Mean latency [µs]")
    plt.yticks(np.arange(len(summary)), [e["name"] for e in summary])
    plt.xlabel("Time [s]")
    plt.ylabel("Workload")
    plt.title("Latency heat-map (mean per second)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ╔══════════════ CLI / main ═══════════════════════════════════════════════╗
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate research-grade latency plots for rtbench results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("results", type=Path, help="rtbench results directory")
    p.add_argument("--outdir", type=Path, help="Where to save PNGs")
    return p.parse_args()


def main() -> None:
    args = cli()
    root = args.results.resolve()
    if not root.is_dir():
        sys.exit("Error: ‘results’ is not a directory")

    outdir = (args.outdir or root / "figures").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary = load_results(root)
    if not summary:
        sys.exit("Error: no valid runs found")

    summary.sort(key=lambda e: e["name"]) 
    plot_timeline(summary, outdir / "timeline_mean.png")
    plot_box(summary,      outdir / "boxplot.png")
    plot_ecdf(summary,     outdir / "ecdf.png")
    plot_heatmap(summary,  outdir / "heatmap.png")

    print("✓ Plots saved to", outdir)


if __name__ == "__main__":
    main()
