"""
plot_mamba_vs_opt.py  (expanded)
─────────────────────────────────
Thesis-quality figures and LaTeX tables comparing Mamba-130M vs OPT-125M
across multiple sequence lengths, derived from simulator results.

Outputs (written to ./figures/ and ./tables/):
  ── Original figures ──────────────────────────────────────────────────────
    fig1_total_cycles            – Total cycles vs sequence length
    fig2_wall_clock              – Wall-clock simulation time vs seq len
    fig3_pe_util                 – PE utilisation % vs seq len
    fig4_dram_reads              – Total DRAM reads vs seq len
    fig5_dram_bw_util            – DRAM BW utilisation % vs seq len
    fig6_sram_hitrates           – Input-SRAM & Acc-SRAM hit rates
    fig7_memory_bound_ratio      – Memory-bound ratio vs seq len
    fig8_speedup                 – Mamba speedup over OPT vs seq len
    fig9_sa_util                 – Systolic-array utilisation % vs seq len
    fig10_memory_idle            – Memory-idle % vs seq len
    fig11_radar                  – Radar / spider chart at seq_len=1
    fig12_layer_pie              – Layer-cycle breakdown pie charts

  ── NEW: Object histogram figures ─────────────────────────────────────────
    fig13_hist_total_bytes       – Total core→L2 memory-request bytes vs seq len
    fig14_hist_avg_object_size   – Average object size vs seq len
    fig15_hist_top2_bar          – Top-2 size-class bytes share (grouped bar)
    fig16_hist_size_class_heatmap– Size-class byte-fraction heatmap (all seq lens)

  ── NEW: L2 cache figures ──────────────────────────────────────────────────
    fig17_l2_hit_miss_rate       – L2 hit/miss rate vs seq len
    fig18_l2_cache_util          – L2 cache utilisation % vs seq len
    fig19_l2_hit_miss_abs        – L2 total hits & misses (absolute) vs seq len
    fig20_l2_evictions_writebacks– L2 evictions & writebacks vs seq len
    fig21_l2_bank_variance       – L2 hit-rate min/max/variance across banks

  ── NEW: SRAM figures ─────────────────────────────────────────────────────
    fig22_sram_hits_misses       – SRAM total hits & misses vs seq len
    fig23_sram_bytes             – SRAM bytes requested vs received vs seq len
    fig24_acc_sram_hits_misses   – Acc-SRAM total hits & misses vs seq len

  ── NEW: DRAM figures ─────────────────────────────────────────────────────
    fig25_dram_reads_writes      – DRAM reads & writes side-by-side vs seq len
    fig26_dram_req_counts        – DRAM read/write request counts vs seq len
    fig27_dram_row_rates         – DRAM row hit/miss/conflict rates vs seq len
    fig28_dram_channels_bw       – Per-channel BW utilisation breakdown

  ── NEW: Core utilisation figures ─────────────────────────────────────────
    fig29_core_util_stack        – Stacked util breakdown per model at each seq len
    fig30_core_idle_breakdown    – Core-idle vs memory-idle vs active vs seq len
    fig31_inst_per_tile          – Instructions per tile vs seq len
    fig32_tiles_finished         – Tiles finished per core vs seq len
    fig33_systolic_issue_preload – Systolic issue count vs preload count vs seq len
    fig34_util_heatmap           – Utilisation heatmap (models × metrics × seq lens)

  ── Original tables ───────────────────────────────────────────────────────
    tab_core_stats.tex
    tab_memory_stats.tex
    tab_seq_scaling.tex
    tab_layer_breakdown.tex

  ── NEW tables ────────────────────────────────────────────────────────────
    tab_object_histogram.tex     – Object histogram stats per model
    tab_l2_detailed.tex          – L2 cache detailed stats per model
    tab_dram_detailed.tex        – DRAM bandwidth & request stats
    tab_core_utilisation.tex     – Full core utilisation breakdown
    tab_sram_detailed.tex        – SRAM & Acc-SRAM detailed stats
    tab_seq_scaling_extended.tex – Extended scaling table (more metrics)

Usage:
    python plot_mamba_vs_opt.py [--data-dir /path/to/csvs]

Requirements:
    pip install pandas matplotlib numpy seaborn
"""

import argparse
import os
import textwrap
import warnings
from pathlib import Path

import matplotlib
from pyparsing import col
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
FIGURES = HERE / "figures"
TABLES  = HERE / "tables"
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

C_MAMBA   = "#1f77b4"
C_OPT     = "#ff7f0e"
C_MAMBA2  = "#aec7e8"   # lighter blue for secondary bars
C_OPT2    = "#ffbb78"   # lighter orange for secondary bars
C_HIT     = "#2ca02c"
C_MISS    = "#d62728"
C_EVICT   = "#9467bd"
C_WB      = "#8c564b"

MARKER_MAMBA = "o"
MARKER_OPT   = "s"

FONT_TITLE  = 13
FONT_AXIS   = 11
FONT_TICK   = 10
FONT_LEGEND = 10

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        FONT_TICK,
    "axes.titlesize":   FONT_TITLE,
    "axes.labelsize":   FONT_AXIS,
    "legend.fontsize":  FONT_LEGEND,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

SEQ_LENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
            10000, 20000, 30000]

# ──────────────────────────────────────────────────────────────────────────────
# Data loading & preparation
# ──────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: Path):
    detailed = pd.read_csv(data_dir / "results_detailed_summary.csv")
    per_log  = pd.read_csv(data_dir / "results_per_log.csv")
    summary  = pd.read_csv(data_dir / "results_summary.csv")
    return detailed, per_log, summary


def extract_seq_len(log_file: str) -> int:
    """Extract sequence length from log filename reliably."""
    import re
    s = str(log_file)
    
    # New combined names: mamba_tiny_s10000.log or tiny_s10000.log
    m = re.search(r"_s(\d{4,})\.log", s)          # catches s10000, s20000, s30000
    if m:
        return int(m.group(1))
    
    # Old power-of-two names: mamba_tiny_s8192.log, tiny_s16384.log
    m = re.search(r"[_-]s(\d+)\.log", s)
    if m:
        return int(m.group(1))
    
    # Fallback for any number before .log
    m = re.search(r"_(\d{1,})\.log", s)
    if m:
        return int(m.group(1))
    
    return -1


def prepare_per_log(per_log: pd.DataFrame) -> pd.DataFrame:
    df = per_log.copy()
    df["seq_len"] = df["log_file"].apply(extract_seq_len)
    df["model"]   = df["log_file"].apply(
        lambda x: "Mamba" if "mamba" in str(x).lower() else "OPT")
    df = df[df["seq_len"] > 0].copy()
    df = df.sort_values(["model", "seq_len"]).reset_index(drop=True)
    return df


def split_models(df: pd.DataFrame):
    mamba = df[df["model"] == "Mamba"].sort_values("seq_len")
    opt   = df[df["model"] == "OPT"].sort_values("seq_len")
    return mamba, opt

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_fig(fig, name: str):
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    print(f"  saved: figures/{name}.png")


def logx_ax(ax, seq_lens=SEQ_LENS):
    ax.set_xscale("log", base=2)

    # Force all ticks
    ax.set_xticks(seq_lens)

    # Show exact values (no scientific notation)
    ax.set_xticklabels([str(x) for x in seq_lens])

    # Disable minor ticks (important!)
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Rotate labels for readability
    ax.tick_params(axis='x', rotation=45)

    ax.set_xlabel("Sequence Length")


def add_legend(ax, **kw):
    ax.legend(loc="upper left", framealpha=0.9, **kw)


def plot_metric_vs_seqlen(mamba, opt, col, ylabel, title, name,
                          yscale="linear", annotate_ratio=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mamba["seq_len"], mamba[col], color=C_MAMBA, marker=MARKER_MAMBA,
            linewidth=2, label="Mamba-130M")
    ax.plot(opt["seq_len"],   opt[col],   color=C_OPT,   marker=MARKER_OPT,
            linewidth=2, label="OPT-125M")
    if annotate_ratio:
        m_aligned = mamba.set_index("seq_len")
        o_aligned = opt.set_index("seq_len")
        common = m_aligned.index.intersection(o_aligned.index)
        for sl in common:
            mv = m_aligned.loc[sl, col]
            ov = o_aligned.loc[sl, col]
            if ov and ov > 0:
                ratio = mv / ov
                y = max(mv, ov)
                ax.annotate(f"{ratio:.1f}×", xy=(sl, y), xytext=(0, 6),
                            textcoords="offset points", ha="center",
                            fontsize=8, color="gray")
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    logx_ax(ax)
    add_legend(ax)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, name)


def dual_line_ax(ax, mamba, opt, col1, col2,
                 label1m, label1o, label2m, label2o,
                 color1m=C_MAMBA, color1o=C_OPT,
                 color2m=C_MAMBA2, color2o=C_OPT2,
                 ls2="--"):
    """Draw two metrics for both models on one axis."""
    ax.plot(mamba["seq_len"], mamba[col1], color=color1m, marker=MARKER_MAMBA,
            linewidth=2, label=label1m)
    ax.plot(opt["seq_len"],   opt[col1],   color=color1o, marker=MARKER_OPT,
            linewidth=2, label=label1o)
    ax.plot(mamba["seq_len"], mamba[col2], color=color2m, marker=MARKER_MAMBA,
            linewidth=2, linestyle=ls2, label=label2m)
    ax.plot(opt["seq_len"],   opt[col2],   color=color2o, marker=MARKER_OPT,
            linewidth=2, linestyle=ls2, label=label2o)


# ──────────────────────────────────────────────────────────────────────────────
# ── ORIGINAL FIGURES 1-12 ────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig1_total_cycles(mamba, opt):
    plot_metric_vs_seqlen(mamba, opt, "timing_total_cycles", "Total Cycles",
        "Fig 1 — Total Simulation Cycles vs Sequence Length",
        "fig1_total_cycles", yscale="log", annotate_ratio=True)

def fig2_wall_clock(mamba, opt):
    plot_metric_vs_seqlen(mamba, opt, "timing_wall_clock_seconds",
        "Wall-Clock Time (s)",
        "Fig 2 — Simulation Wall-Clock Time vs Sequence Length",
        "fig2_wall_clock", yscale="log", annotate_ratio=True)

def fig3_pe_util(mamba, opt):
    plot_metric_vs_seqlen(mamba, opt, "compute_avg_pe_util_pct",
        "PE Utilisation (%)",
        "Fig 3 — Average PE Utilisation vs Sequence Length",
        "fig3_pe_util")

def fig4_dram_reads(mamba, opt):
    plot_metric_vs_seqlen(mamba, opt, "dram_total_reads", "Total DRAM Reads",
        "Fig 4 — Total DRAM Read Requests vs Sequence Length",
        "fig4_dram_reads", yscale="log", annotate_ratio=True)

def fig5_dram_bw_util(mamba, opt):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for data, label, color, marker in [
        (mamba, "Mamba-130M", C_MAMBA, MARKER_MAMBA),
        (opt,   "OPT-125M",   C_OPT,   MARKER_OPT)
    ]:
        bw = data["dram_avg_bw_utilization_pct"].clip(upper=100)
        ax.plot(data["seq_len"], bw, color=color, marker=marker,
                linewidth=2, label=label)
    ax.set_ylabel("DRAM BW Utilisation (%)")
    ax.set_title("Fig 5 — Average DRAM Bandwidth Utilisation vs Sequence Length")
    logx_ax(ax); add_legend(ax)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout(); save_fig(fig, "fig5_dram_bw_util")

def fig6_sram_hitrates(mamba, opt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, ylabel, title in [
        (axes[0], "sram_hit_rate",     "Input-SRAM Hit Rate",  "Input-SRAM Hit Rate"),
        (axes[1], "acc_sram_hit_rate", "Acc-SRAM Hit Rate",    "Accumulator-SRAM Hit Rate"),
    ]:
        ax.plot(mamba["seq_len"], mamba[col], color=C_MAMBA, marker=MARKER_MAMBA,
                linewidth=2, label="Mamba-130M")
        ax.plot(opt["seq_len"],   opt[col],   color=C_OPT,   marker=MARKER_OPT,
                linewidth=2, label="OPT-125M")
        ax.set_yscale("log"); ax.set_ylabel(ylabel); ax.set_title(title)
        logx_ax(ax); ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 6 — On-Chip SRAM Hit Rates vs Sequence Length", fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig6_sram_hitrates")

def fig7_memory_bound_ratio(detailed):
    mamba = detailed[detailed["model"] == "Mamba"].sort_values("seq_len")
    opt   = detailed[detailed["model"] == "OPT|tiny"].sort_values("seq_len")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mamba["seq_len"], mamba["memory_bound_ratio"].clip(upper=2e5),
            color=C_MAMBA, marker=MARKER_MAMBA, linewidth=2, label="Mamba-130M")
    ax.plot(opt["seq_len"],   opt["memory_bound_ratio"].clip(upper=2e5),
            color=C_OPT,   marker=MARKER_OPT,   linewidth=2, label="OPT-125M")
    ax.set_yscale("log"); ax.set_ylabel("Memory-Bound Ratio (cycles/byte)")
    ax.set_title("Fig 7 — Memory-Bound Ratio vs Sequence Length")
    logx_ax(ax); add_legend(ax)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout(); save_fig(fig, "fig7_memory_bound_ratio")

def fig8_speedup(detailed):
    mamba   = detailed[detailed["model"] == "Mamba"].sort_values("seq_len").set_index("seq_len")
    opt     = detailed[detailed["model"] == "OPT|tiny"].sort_values("seq_len").set_index("seq_len")
    common  = mamba.index.intersection(opt.index)
    speedup = opt.loc[common, "timing_total_cycles"] / mamba.loc[common, "timing_total_cycles"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(common, speedup.values, color=C_MAMBA, marker=MARKER_MAMBA,
            linewidth=2, label="Mamba-130M speedup")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="OPT baseline")
    ax.set_ylabel("Speedup (OPT Cycles / Mamba Cycles)")
    ax.set_title("Fig 8 — Mamba Cycle-Count Speedup over OPT vs Sequence Length")
    logx_ax(ax); add_legend(ax)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    for x, y in zip(common, speedup.values):
        ax.annotate(f"{y:.2f}×", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout(); save_fig(fig, "fig8_speedup")

def fig9_sa_util(mamba, opt):
    plot_metric_vs_seqlen(mamba, opt, "compute_avg_systolic_util_pct",
        "Systolic-Array Utilisation (%)",
        "Fig 9 — Systolic-Array Utilisation vs Sequence Length", "fig9_sa_util")

def fig10_memory_idle(mamba, opt):
    plot_metric_vs_seqlen(mamba, opt, "compute_avg_memory_idle_pct",
        "Memory-Idle (%)",
        "Fig 10 — Average Memory-Idle Percentage vs Sequence Length", "fig10_memory_idle")

def fig11_radar(mamba_s1, opt_s1):
    metrics = [
        ("PE Util %",      "compute_avg_pe_util_pct",       "max"),
        ("SA Util %",      "compute_avg_systolic_util_pct", "max"),
        ("SRAM Hit",       "sram_hit_rate",                 "max"),
        ("Acc-SRAM Hit",   "acc_sram_hit_rate",             "max"),
        ("Mem-Idle %",     "compute_avg_memory_idle_pct",   "min"),
        ("DRAM BW Util %", "dram_avg_bw_utilization_pct",   "max"),
    ]
    labels = [m[0] for m in metrics]
    N      = len(labels)
    def get_vals(row):
        return [row[m[1]] for m in metrics]
    m_vals   = np.array(get_vals(mamba_s1.iloc[0]))
    o_vals   = np.array(get_vals(opt_s1.iloc[0]))
    combined = np.stack([m_vals, o_vals])
    mn = combined.min(axis=0); mx = combined.max(axis=0)
    denom = np.where((mx - mn) == 0, 1, mx - mn)
    m_norm = (m_vals - mn) / denom
    o_norm = (o_vals - mn) / denom
    for i, m in enumerate(metrics):
        if m[2] == "min":
            m_norm[i] = 1 - m_norm[i]; o_norm[i] = 1 - o_norm[i]
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    m_norm  = np.append(m_norm, m_norm[0])
    o_norm  = np.append(o_norm, o_norm[0])
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, m_norm, color=C_MAMBA, linewidth=2, label="Mamba-130M")
    ax.fill(angles, m_norm, color=C_MAMBA, alpha=0.20)
    ax.plot(angles, o_norm, color=C_OPT,   linewidth=2, label="OPT-125M")
    ax.fill(angles, o_norm, color=C_OPT,   alpha=0.20)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("Fig 11 — Normalised Hardware Metrics (seq_len=1)", pad=20, fontsize=FONT_TITLE)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), framealpha=0.9)
    fig.tight_layout(); save_fig(fig, "fig11_radar")

def fig12_layer_pie(summary):
    mamba_row = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt_row   = summary[summary["log_file"].str.contains("opt|tiny",   case=False, na=False)].iloc[0]
    mamba_layers = {
        "GEMM":           mamba_row.get("layer_gemm_cycles", 0) or 0,
        "ElementwiseMul": mamba_row.get("layer_elementwise_mul_cycles", 0) or 0,
        "ElementwiseAdd": mamba_row.get("layer_elementwise_add_cycles", 0) or 0,
        "Other":          mamba_row.get("layer_other_cycles", 0) or 0,
    }
    opt_layers = {
        "FFN FC2":    opt_row.get("layer_ffn_fc2_cycles", 0) or 0,
        "FFN FC1":    opt_row.get("layer_ffn_fc1_cycles", 0) or 0,
        "QKV Proj":   opt_row.get("layer_QKV_projection_cycles", 0) or 0,
        "Attn Proj":  opt_row.get("layer_attn_projection_cycles", 0) or 0,
        "Attention":  opt_row.get("layer_attention_cycles", 0) or 0,
        "Other":      opt_row.get("layer_other_cycles", 0) or 0,
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, layers, title in [
        (axes[0], mamba_layers, "Mamba-130M Layer Cycle Distribution"),
        (axes[1], opt_layers,   "OPT-125M Layer Cycle Distribution"),
    ]:
        labels_f = [l for l, s in layers.items() if s > 0]
        sizes_f  = [s for s in layers.values() if s > 0]
        if not sizes_f:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            ax.pie(sizes_f, labels=labels_f, autopct="%1.1f%%", startangle=140,
                   wedgeprops=dict(linewidth=0.8, edgecolor="white"))
        ax.set_title(title, fontsize=FONT_TITLE)
    fig.suptitle("Fig 12 — Per-Layer Cycle Breakdown (7-step simulation)", fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig12_layer_pie")


# ──────────────────────────────────────────────────────────────────────────────
# ── NEW FIGURES 13-16: Object Histogram ──────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig13_hist_total_bytes(mamba, opt):
    """Total core→L2 memory-request bytes vs sequence length."""
    plot_metric_vs_seqlen(mamba, opt, "hist_total_bytes",
        "Total Core→L2 Request Bytes",
        "Fig 13 — Object Histogram: Total Memory-Request Bytes to L2 vs Sequence Length",
        "fig13_hist_total_bytes", yscale="log", annotate_ratio=True)


def fig14_hist_avg_object_size(mamba, opt):
    """Average object (request) size in bytes vs sequence length."""
    plot_metric_vs_seqlen(mamba, opt, "hist_avg_object_size",
        "Average Object Size (bytes)",
        "Fig 14 — Object Histogram: Average Core→L2 Request Size vs Sequence Length",
        "fig14_hist_avg_object_size")


def fig15_hist_top2_bar(mamba, opt):
    """
    Grouped bar chart: top-1 and top-2 size-class byte shares at each seq len,
    for both models side by side.
    """
    seq_lens_m = mamba["seq_len"].tolist()
    seq_lens_o = opt["seq_len"].tolist()
    all_sl     = sorted(set(seq_lens_m) | set(seq_lens_o))

    m_idx = mamba.set_index("seq_len")
    o_idx = opt.set_index("seq_len")

    x    = np.arange(len(all_sl))
    w    = 0.20
    fig, ax = plt.subplots(figsize=(12, 5))

    def safe_col(df, sl, col):
        try:
            return df.loc[sl, col] if sl in df.index else 0.0
        except Exception:
            return 0.0

    m_top1 = [safe_col(m_idx, sl, "hist_top1_bytes_pct") for sl in all_sl]
    m_top2 = [safe_col(m_idx, sl, "hist_top2_bytes_pct") for sl in all_sl]
    o_top1 = [safe_col(o_idx, sl, "hist_top1_bytes_pct") for sl in all_sl]
    o_top2 = [safe_col(o_idx, sl, "hist_top2_bytes_pct") for sl in all_sl]

    ax.bar(x - 1.5*w, m_top1, w, label="Mamba Top-1", color=C_MAMBA)
    ax.bar(x - 0.5*w, m_top2, w, label="Mamba Top-2", color=C_MAMBA2, edgecolor=C_MAMBA, linewidth=0.8)
    ax.bar(x + 0.5*w, o_top1, w, label="OPT Top-1",   color=C_OPT)
    ax.bar(x + 1.5*w, o_top2, w, label="OPT Top-2",   color=C_OPT2,   edgecolor=C_OPT,   linewidth=0.8)

    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in all_sl])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Share of Total Request Bytes (%)")
    ax.set_title("Fig 15 — Object Histogram: Top-2 Size-Class Byte Share vs Sequence Length")
    ax.legend(framealpha=0.9, ncol=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(); save_fig(fig, "fig15_hist_top2_bar")


def fig16_hist_size_class_heatmap(mamba, opt):
    """
    Heatmap of top-1 and top-2 size classes (in bytes) across sequence lengths
    for each model, illustrating which object sizes dominate traffic.
    """
    m_idx = mamba.set_index("seq_len")
    o_idx = opt.set_index("seq_len")
    all_sl = sorted(set(mamba["seq_len"]) | set(opt["seq_len"]))

    def safe(df, sl, col):
        try:
            return float(df.loc[sl, col]) if sl in df.index else np.nan
        except Exception:
            return np.nan

    rows = {
        "Mamba top-1 size (B)": [safe(m_idx, sl, "hist_top1_size_bytes") for sl in all_sl],
        "Mamba top-2 size (B)": [safe(m_idx, sl, "hist_top2_size_bytes") for sl in all_sl],
        "OPT   top-1 size (B)": [safe(o_idx, sl, "hist_top1_size_bytes") for sl in all_sl],
        "OPT   top-2 size (B)": [safe(o_idx, sl, "hist_top2_size_bytes") for sl in all_sl],
    }
    data_mat = np.array(list(rows.values()), dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(all_sl)*1.2), 3.5))
    im = ax.imshow(data_mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(all_sl))); ax.set_xticklabels([str(s) for s in all_sl])
    ax.set_yticks(range(len(rows)));   ax.set_yticklabels(list(rows.keys()))
    ax.set_xlabel("Sequence Length")
    ax.set_title("Fig 16 — Object Histogram: Dominant Size Classes Across Sequence Lengths")
    plt.colorbar(im, ax=ax, label="Object Size (bytes)")
    for i in range(data_mat.shape[0]):
        for j in range(data_mat.shape[1]):
            v = data_mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8,
                        color="black" if v < data_mat.max()*0.7 else "white")
    fig.tight_layout(); save_fig(fig, "fig16_hist_size_class_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# ── NEW FIGURES 17-21: L2 Cache ───────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig17_l2_hit_miss_rate(mamba, opt):
    """L2 hit and miss rates on the same axes (solid=hit, dashed=miss)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dual_line_ax(ax, mamba, opt,
                 "l2_overall_hit_rate",  "l2_overall_miss_rate",
                 "Mamba hit rate", "OPT hit rate",
                 "Mamba miss rate", "OPT miss rate",
                 color1m=C_MAMBA, color1o=C_OPT,
                 color2m=C_MAMBA, color2o=C_OPT)
    ax.set_ylabel("Rate (0–1)"); ax.set_title("Fig 17 — L2 Hit & Miss Rate vs Sequence Length")
    logx_ax(ax); ax.legend(framealpha=0.9, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout(); save_fig(fig, "fig17_l2_hit_miss_rate")


def fig18_l2_cache_util(mamba, opt):
    plot_metric_vs_seqlen(mamba, opt, "l2_cache_util_pct",
        "L2 Cache Utilisation (%)",
        "Fig 18 — L2 Cache Utilisation % vs Sequence Length",
        "fig18_l2_cache_util")


def fig19_l2_hit_miss_abs(mamba, opt):
    """L2 total hits and total misses (log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, col, title in [
        (axes[0], "l2_total_hits",   "L2 Total Hits"),
        (axes[1], "l2_total_misses", "L2 Total Misses"),
    ]:
        print(col, "Mamba:", mamba[col])
        print(col, "OPT:", opt[col])
        ax.plot(mamba["seq_len"], mamba[col], color=C_MAMBA, marker=MARKER_MAMBA,
                linewidth=2, label="Mamba-130M")
        ax.plot(opt["seq_len"],   opt[col],   color=C_OPT,   marker=MARKER_OPT,
                linewidth=2, label="OPT-125M")
        ax.set_yscale("log"); ax.set_title(title)
        logx_ax(ax); ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 19 — L2 Absolute Hit & Miss Counts vs Sequence Length", fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig19_l2_hit_miss_abs")


def fig20_l2_evictions_writebacks(mamba, opt):
    """L2 evictions and writebacks — two models, two metrics."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dual_line_ax(ax, mamba, opt,
                 "l2_total_evictions", "l2_total_writebacks",
                 "Mamba evictions", "OPT evictions",
                 "Mamba writebacks", "OPT writebacks")
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Fig 20 — L2 Evictions & Writebacks vs Sequence Length")
    logx_ax(ax); ax.legend(framealpha=0.9, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout(); save_fig(fig, "fig20_l2_evictions_writebacks")


def fig21_l2_bank_variance(mamba, opt):
    """L2 per-bank hit-rate spread: min, mean, max with filled band."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, label, color in [
        (axes[0], mamba, "Mamba-130M", C_MAMBA),
        (axes[1], opt,   "OPT-125M",   C_OPT),
    ]:
        sl   = data["seq_len"]
        mn   = data.get("l2_min_bank_hit_rate",  pd.Series(dtype=float))
        mx   = data.get("l2_max_bank_hit_rate",  pd.Series(dtype=float))
        mean = data.get("l2_overall_hit_rate",   pd.Series(dtype=float))
        ax.plot(sl, mean, color=color, marker="o", linewidth=2, label="Mean hit rate")
        if mn is not None and mx is not None:
            ax.fill_between(sl, mn, mx, color=color, alpha=0.2, label="Min–Max band")
        ax.set_title(f"{label} — L2 Bank Hit-Rate Spread")
        ax.set_ylabel("Hit Rate (0–1)")
        logx_ax(ax); ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 21 — L2 Per-Bank Hit-Rate Variance vs Sequence Length", fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig21_l2_bank_variance")


# ──────────────────────────────────────────────────────────────────────────────
# ── NEW FIGURES 22-24: SRAM ───────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig22_sram_hits_misses(mamba, opt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, model_data, label, color in [
        (axes[0], mamba, "Mamba-130M", C_MAMBA),
        (axes[1], opt,   "OPT-125M",   C_OPT),
    ]:
        ax.plot(model_data["seq_len"], model_data["sram_total_hits"],
                color=C_HIT,  marker="o", linewidth=2, label="Hits")
        ax.plot(model_data["seq_len"], model_data["sram_total_misses"],
                color=C_MISS, marker="s", linewidth=2, label="Misses")
        ax.set_yscale("log"); ax.set_title(f"{label} — Input-SRAM Hits & Misses")
        ax.set_ylabel("Count (log scale)")
        logx_ax(ax); ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 22 — Input-SRAM Hit & Miss Counts vs Sequence Length", fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig22_sram_hits_misses")


def fig23_sram_bytes(mamba, opt):
    """SRAM bytes requested vs bytes received — efficiency gap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, model_data, label in [
        (axes[0], mamba, "Mamba-130M"),
        (axes[1], opt,   "OPT-125M"),
    ]:
        req = model_data.get("sram_avg_bytes_req", pd.Series(dtype=float))
        rcv = model_data.get("sram_avg_bytes_rcv", pd.Series(dtype=float))
        if req is not None:
            ax.plot(model_data["seq_len"], req, color=C_MAMBA if "Mamba" in label else C_OPT,
                    marker="o", linewidth=2, label="Bytes Requested")
        if rcv is not None:
            ax.plot(model_data["seq_len"], rcv,
                    color=C_MAMBA2 if "Mamba" in label else C_OPT2,
                    marker="s", linewidth=2, linestyle="--", label="Bytes Received")
        ax.set_yscale("log"); ax.set_title(f"{label} — SRAM Bytes Req vs Received")
        ax.set_ylabel("Bytes (log scale)")
        logx_ax(ax); ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 23 — Average SRAM Bytes Requested vs Received per Core", fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig23_sram_bytes")


def fig24_acc_sram_hits_misses(mamba, opt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, model_data, label in [
        (axes[0], mamba, "Mamba-130M"),
        (axes[1], opt,   "OPT-125M"),
    ]:
        ax.plot(model_data["seq_len"], model_data["acc_sram_total_hits"],
                color=C_HIT,  marker="o", linewidth=2, label="Acc-SRAM Hits")
        ax.plot(model_data["seq_len"], model_data["acc_sram_total_misses"],
                color=C_MISS, marker="s", linewidth=2, label="Acc-SRAM Misses")
        ax.set_yscale("log"); ax.set_title(f"{label} — Accumulator-SRAM Hits & Misses")
        ax.set_ylabel("Count (log scale)")
        logx_ax(ax); ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 24 — Accumulator-SRAM Hit & Miss Counts vs Sequence Length", fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig24_acc_sram_hits_misses")


# ──────────────────────────────────────────────────────────────────────────────
# ── NEW FIGURES 25-28: DRAM ───────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig25_dram_reads_writes(mamba, opt):
    """DRAM total reads and writes on one plot per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, model_data, label, color in [
        (axes[0], mamba, "Mamba-130M", C_MAMBA),
        (axes[1], opt,   "OPT-125M",   C_OPT),
    ]:
        ax.plot(model_data["seq_len"], model_data["dram_total_reads"],
                color=color, marker="o", linewidth=2, label="Total Reads")
        ax.plot(model_data["seq_len"], model_data["dram_total_writes"],
                color=color, marker="s", linewidth=2, linestyle="--", label="Total Writes")
        ax.set_yscale("log"); ax.set_title(f"{label} — DRAM Reads & Writes")
        ax.set_ylabel("Count (log scale)")
        logx_ax(ax); ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 25 — DRAM Total Read & Write Traffic vs Sequence Length", fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig25_dram_reads_writes")


def fig26_dram_req_counts(mamba, opt):
    """DRAM per-channel avg read/write request counts."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dual_line_ax(ax, mamba, opt,
                 "dram_avg_read_requests", "dram_avg_write_requests",
                 "Mamba read reqs", "OPT read reqs",
                 "Mamba write reqs", "OPT write reqs")
    ax.set_yscale("log")
    ax.set_ylabel("Avg Requests per Channel (log scale)")
    ax.set_title("Fig 26 — DRAM Average Read & Write Request Counts vs Sequence Length")
    logx_ax(ax); ax.legend(framealpha=0.9, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout(); save_fig(fig, "fig26_dram_req_counts")


def fig27_dram_row_rates(mamba, opt):
    """DRAM row hit/miss/conflict rates — shows memory access pattern quality."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, model_data, label in [
        (axes[0], mamba, "Mamba-130M"),
        (axes[1], opt,   "OPT-125M"),
    ]:
        for col, lbl, color in [
            ("dram_avg_row_hit_rate_pct",      "Row Hit Rate %",      C_HIT),
            ("dram_avg_row_miss_rate_pct",      "Row Miss Rate %",     C_MISS),
            ("dram_avg_row_conflict_rate_pct",  "Row Conflict Rate %", C_EVICT),
        ]:
            ax.plot(model_data["seq_len"], model_data[col],
                    color=color, marker="o", linewidth=2, label=lbl)
        ax.set_title(f"{label} — DRAM Row Access Rates")
        ax.set_ylabel("Rate (%)")
        logx_ax(ax); ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 27 — DRAM Row Hit / Miss / Conflict Rates vs Sequence Length",
                 fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig27_dram_row_rates")


def fig28_dram_channels_bw(mamba, opt):
    """DRAM BW utilisation + number of channels — dual y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, model_data, label, color in [
        (axes[0], mamba, "Mamba-130M", C_MAMBA),
        (axes[1], opt,   "OPT-125M",   C_OPT),
    ]:
        ax2 = ax.twinx()
        ax.plot(model_data["seq_len"],
                model_data["dram_avg_bw_utilization_pct"].clip(upper=100),
                color=color, marker="o", linewidth=2, label="BW Util %")
        ax2.plot(model_data["seq_len"], model_data["dram_num_channels"],
                 color="gray", marker="s", linewidth=1.5, linestyle="--",
                 label="# Channels")
        ax.set_ylabel("BW Utilisation (%)")
        ax2.set_ylabel("Number of Channels")
        ax.set_title(f"{label} — DRAM BW Util & Channel Count")
        logx_ax(ax)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 28 — DRAM Bandwidth Utilisation & Channel Count vs Sequence Length",
                 fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig28_dram_channels_bw")


# ──────────────────────────────────────────────────────────────────────────────
# ── NEW FIGURES 29-34: Core Utilisation ──────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def _safe_series(df, col):
    if col in df.columns:
        return df[col].fillna(0)
    return pd.Series([0] * len(df), index=df.index)


def fig29_core_util_stack(mamba, opt):
    """
    Stacked area chart: systolic util, PE util, memory idle, core idle
    for each model across sequence lengths.
    """
    util_cols = [
        ("compute_avg_systolic_util_pct", "Systolic Util",  "#2ca02c"),
        ("compute_avg_pe_util_pct",        "PE Util",        "#1f77b4"),
        ("compute_avg_memory_idle_pct",    "Memory Idle",    "#ff7f0e"),
        ("compute_avg_core_idle_pct",      "Core Idle",      "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, model_data, label in [
        (axes[0], mamba, "Mamba-130M"),
        (axes[1], opt,   "OPT-125M"),
    ]:
        sl  = model_data["seq_len"].tolist()
        bot = np.zeros(len(sl))
        for col, lbl, color in util_cols:
            vals = _safe_series(model_data, col).values
            ax.bar(range(len(sl)), vals, bottom=bot, label=lbl,
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            bot += vals
        ax.set_xticks(range(len(sl))); ax.set_xticklabels([str(s) for s in sl])
        ax.set_xlabel("Sequence Length"); ax.set_ylabel("% (stacked)")
        ax.set_title(f"{label} — Core Utilisation Breakdown")
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("Fig 29 — Stacked Core Utilisation Breakdown vs Sequence Length",
                 fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig29_core_util_stack")


def fig30_core_idle_breakdown(mamba, opt):
    """Memory-idle vs core-idle side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, ylabel, title in [
        (axes[0], "compute_avg_memory_idle_pct", "Memory-Idle (%)",
         "Memory-Idle % vs Sequence Length"),
        (axes[1], "compute_avg_core_idle_pct",   "Core-Idle (%)",
         "Core-Idle % vs Sequence Length"),
    ]:
        ax.plot(mamba["seq_len"], mamba[col], color=C_MAMBA, marker="o",
                linewidth=2, label="Mamba-130M")
        ax.plot(opt["seq_len"],   opt[col],   color=C_OPT,   marker="s",
                linewidth=2, label="OPT-125M")
        ax.set_ylabel(ylabel); ax.set_title(title)
        logx_ax(ax); ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Fig 30 — Memory-Idle & Core-Idle Breakdown vs Sequence Length",
                 fontsize=FONT_TITLE)
    fig.tight_layout(); save_fig(fig, "fig30_core_idle_breakdown")


def fig31_inst_per_tile(mamba, opt):
    plot_metric_vs_seqlen(mamba, opt, "core_avg_instructions_per_tile",
        "Instructions per Tile",
        "Fig 31 — Average Instructions per Tile vs Sequence Length",
        "fig31_inst_per_tile")


def fig32_tiles_finished(mamba, opt):
    plot_metric_vs_seqlen(mamba, opt, "core_avg_tiles_finished",
        "Tiles Finished per Core",
        "Fig 32 — Average Tiles Finished per Core vs Sequence Length",
        "fig32_tiles_finished", yscale="log")


def fig33_systolic_issue_preload(mamba, opt):
    """Systolic instruction issue count vs preload count — pipeline balance."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dual_line_ax(ax, mamba, opt,
                 "core_avg_systolic_inst_issue", "core_avg_systolic_preload",
                 "Mamba issue", "OPT issue",
                 "Mamba preload", "OPT preload")
    ax.set_yscale("log")
    ax.set_ylabel("Count per Core (log scale)")
    ax.set_title("Fig 33 — Systolic Instruction Issue vs Preload Count vs Sequence Length")
    logx_ax(ax); ax.legend(framealpha=0.9, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout(); save_fig(fig, "fig33_systolic_issue_preload")


def fig34_util_heatmap(mamba, opt):
    """
    Heatmap: rows = utilisation metrics, columns = sequence lengths,
    two sub-heatmaps (Mamba top, OPT bottom).
    """
    metrics = {
        "SA Util %":       "compute_avg_systolic_util_pct",
        "PE Util %":       "compute_avg_pe_util_pct",
        "Mem Idle %":      "compute_avg_memory_idle_pct",
        "Core Idle %":     "compute_avg_core_idle_pct",
        "DRAM BW %":       "dram_avg_bw_utilization_pct",
        "L2 Util %":       "l2_cache_util_pct",
        "L2 Hit Rate":     "l2_overall_hit_rate",
        "SRAM Hit Rate":   "sram_hit_rate",
        "Acc-SRAM Hit":    "acc_sram_hit_rate",
    }
    all_sl = sorted(set(mamba["seq_len"]) | set(opt["seq_len"]))
    m_idx  = mamba.set_index("seq_len")
    o_idx  = opt.set_index("seq_len")

    def build_mat(df_idx):
        rows = []
        for col in metrics.values():
            row = []
            for sl in all_sl:
                try:
                    v = float(df_idx.loc[sl, col]) if sl in df_idx.index else np.nan
                except Exception:
                    v = np.nan
                row.append(v)
            rows.append(row)
        return np.array(rows, dtype=float)

    m_mat = build_mat(m_idx)
    o_mat = build_mat(o_idx)
    ylabels = list(metrics.keys())
    xlabels = [str(s) for s in all_sl]

    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(all_sl)*1.3), 7),
                              gridspec_kw={"hspace": 0.5})
    for ax, mat, title in [
        (axes[0], m_mat, "Mamba-130M"),
        (axes[1], o_mat, "OPT-125M"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels)
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
        ax.set_xlabel("Sequence Length"); ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.03)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=7.5,
                            color="black" if 20 < v < 80 else "white")
    fig.suptitle("Fig 34 — Utilisation Metric Heatmap (Mamba vs OPT)", fontsize=FONT_TITLE)
    save_fig(fig, "fig34_util_heatmap")
def fig35_opt_cycle_distribution_all(per_log):
    """
    Fig 35: Single clean graph showing cycle distribution for OPT-125M 
    across ALL sequence lengths (1 to 30000).
    """
    # Correct filtering for OPT / tiny
    opt = per_log[per_log["model"].isin(["OPT", "tiny"])].copy()
    opt = opt.sort_values("seq_len").reset_index(drop=True)
    
    if opt.empty:
        print("Warning: No OPT / tiny data found for fig35")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(opt))
    bottom = np.zeros(len(opt))

    # Main layer components for OPT
    components = [
        ("Attention",          "layer_attention_cycles",     '#d62728'),   # Red
        ("FFN FC1",            "layer_ffn_fc1_cycles",       '#1f77b4'),   # Blue
        ("FFN FC2",            "layer_ffn_fc2_cycles",       '#ff7f0e'),   # Orange
        ("QKV Projection",     "layer_QKV_projection_cycles",'#2ca02c'),   # Green
        ("Attn Projection",    "layer_attn_projection_cycles",'#9467bd'),  # Purple
        ("GEMM",               "layer_gemm_cycles",          '#8c564b'),   # Brown
        ("Other",              "layer_other_cycles",         '#7f7f7f'),   # Gray
    ]

    for label, col, color in components:
        values = opt[col].fillna(0).values
        ax.bar(x, values, label=label, color=color, alpha=0.85, 
               edgecolor='white', linewidth=0.4)
        bottom += values

    # Total cycles line on secondary axis
    ax2 = ax.twinx()
    ax2.plot(x, opt["timing_total_cycles"], color='black', linewidth=3.5, 
             marker='o', markersize=8, label="Total Cycles")

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Cycles per Layer Component", fontsize=12)
    ax2.set_ylabel("Total Cycles", fontsize=12, color='black')
    
    ax.set_title("Fig 35 — OPT-125M Cycle Distribution by Layer Type\n"
                 "(All Sequence Lengths)", fontsize=14, pad=20)

    # X-axis with all your seq lens
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in opt["seq_len"]], rotation=45, ha='right')

    ax.set_yscale("log")
    ax2.set_yscale("log")

    # Legends
    ax.legend(title="Layer / Component", bbox_to_anchor=(1.02, 1), 
              loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    ax.grid(True, which="both", linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig35_opt_cycle_distribution_all")
    
    print(" saved: figures/fig35_opt_cycle_distribution_all.png")

# ──────────────────────────────────────────────────────────────────────────────
# ── ORIGINAL LATEX TABLES 1-4 ────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fmt(val, fmt_str=".3g"):
    if pd.isna(val):
        return "---"
    try:
        return f"{val:{fmt_str}}"
    except (ValueError, TypeError):
        return str(val)


def write_tex(name: str, content: str):
    path = TABLES / f"{name}.tex"
    path.write_text(content)
    print(f"  saved: tables/{name}.tex")


def tab_core_stats(summary):
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt|tiny",   case=False, na=False)].iloc[0]
    rows  = [
        ("Total Cycles",                   "timing_total_cycles",                    ".6g"),
        ("Simulation Time (\\textmu s)",    "timing_total_us",                        ".6g"),
        ("Total Compute Cycles",           "timing_total_compute_cycles",             ".4g"),
        ("Wall-Clock Sim Time (s)",        "timing_wall_clock_seconds",              ".2f"),
        ("Total Tiles",                    "timing_total_tiles",                     ".4g"),
        ("Tiles per Second",               "timing_tiles_per_second",               ".4f"),
        ("Systolic Array Util (\\%)",      "compute_avg_systolic_util_pct",          ".4f"),
        ("PE Utilisation (\\%)",           "compute_avg_pe_util_pct",               ".4f"),
        ("Avg Memory-Idle (\\%)",          "compute_avg_memory_idle_pct",           ".4f"),
        ("Avg Core-Idle (\\%)",            "compute_avg_core_idle_pct",             ".4f"),
        ("Total Compute (GFLOPs)",         "compute_total_GFLOPs",                  ".4f"),
        ("GEMM Memory-Bound (\\%)",        "compute_gemm_memory_bound_pct",          ".1f"),
        ("GEMM Avg Arith. Intensity",      "compute_gemm_avg_arithmetic_intensity", ".4g"),
    ]
    body = "".join(
        f"        {l} & {fmt(mamba.get(c, float('nan')), f)} & {fmt(opt.get(c, float('nan')), f)} \\\\\n"
        for l, c, f in rows)
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{Core Execution Statistics (7-step simulation)}
    \label{tab:core_stats_gen}
    \begin{tabular}{lcc}\hline
    Metric & Mamba-130M & OPT-125M \\\hline
    """ + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_core_stats", tex)


def tab_memory_stats(summary):
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt|tiny",   case=False, na=False)].iloc[0]
    rows  = [
        ("Weight Size (GB)",             "model_weight_size_GB",             ".3f"),
        ("Scratchpad Size (KB)",         "hw_spad_size_KB",                   "g"),
        ("Accumulation Buffer (KB)",     "hw_accumulator_size_KB",            "g"),
        ("Input SRAM Hit Rate",          "sram_hit_rate",                    ".4f"),
        ("Acc-SRAM Hit Rate",            "acc_sram_hit_rate",               ".6f"),
        ("Total DRAM Reads",             "dram_total_reads",                 ".5g"),
        ("Total DRAM Writes",            "dram_total_writes",                ".5g"),
        ("DRAM Channels",                "dram_num_channels",                  "g"),
        ("Avg DRAM BW Util (\\%)",       "dram_avg_bw_utilization_pct",      ".2f"),
        ("Avg Row Hit Rate (\\%)",       "dram_avg_row_hit_rate_pct",        ".2f"),
        ("Avg Row Conflict Rate (\\%)",  "dram_avg_row_conflict_rate_pct",   ".2f"),
        ("L2 Overall Hit Rate",          "l2_overall_hit_rate",              ".4f"),
        ("L2 Cache Utilisation (\\%)",   "l2_cache_util_pct",                ".2f"),
        ("L2 Total Misses",              "l2_total_misses",                    "g"),
    ]
    body = "".join(
        f"        {l} & {fmt(mamba.get(c, float('nan')), f)} & {fmt(opt.get(c, float('nan')), f)} \\\\\n"
        for l, c, f in rows)
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{Memory Hierarchy Statistics (7-step simulation)}
    \label{tab:memory_stats_gen}
    \begin{tabular}{lcc}\hline
    Metric & Mamba-130M & OPT-125M \\\hline
    """ + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_memory_stats", tex)


def tab_seq_scaling(detailed):
    mamba = detailed[detailed["model"] == "Mamba"].sort_values("seq_len").set_index("seq_len")
    opt   = detailed[detailed["model"] == "OPT|tiny"].sort_values("seq_len").set_index("seq_len")
    header = (
        r"    Seq Len & \multicolumn{2}{c}{Total Cycles} & "
        r"\multicolumn{2}{c}{Wall-Clock Time (s)} & Speedup \\" + "\n"
        r"            & Mamba & OPT & Mamba & OPT & (OPT/Mamba) \\\hline"
    )
    body = ""
    for sl in SEQ_LENS:
        if sl not in mamba.index or sl not in opt.index:
            continue
        mc = mamba.loc[sl, "timing_total_cycles"]
        oc = opt.loc[sl,   "timing_total_cycles"]
        mw = mamba.loc[sl, "timing_wall_clock_seconds"]
        ow = opt.loc[sl,   "timing_wall_clock_seconds"]
        sp = oc / mc if mc > 0 else float("nan")
        body += (f"    {sl:>5} & {mc:.3e} & {oc:.3e} & "
                 f"{mw:.1f} & {ow:.1f} & {sp:.3f} \\\\\n")
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{Cycle-count and wall-clock time scaling with sequence length}
    \label{tab:seq_scaling}
    \begin{tabular}{r cc cc c}\hline
    """ + header + "\n" + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_seq_scaling", tex)


def tab_layer_breakdown(summary):
    mamba   = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt     = summary[summary["log_file"].str.contains("opt|tiny",   case=False, na=False)].iloc[0]
    total_m = mamba.get("timing_total_cycles", 1) or 1
    total_o = opt.get("timing_total_cycles",   1) or 1
    ops = [
        ("FFN FC2",         None,                          "layer_ffn_fc2_cycles"),
        ("FFN FC1",         None,                          "layer_ffn_fc1_cycles"),
        ("QKV Projection",  None,                          "layer_QKV_projection_cycles"),
        ("Attn Projection", None,                          "layer_attn_projection_cycles"),
        ("Attention",       None,                          "layer_attention_cycles"),
        ("GEMM",            "layer_gemm_cycles",           None),
        ("Elementwise Mul", "layer_elementwise_mul_cycles", None),
        ("Elementwise Add", "layer_elementwise_add_cycles", None),
        ("Other",           "layer_other_cycles",          "layer_other_cycles"),
    ]
    body = ""
    for label, mc, oc in ops:
        m_cyc = (mamba.get(mc, 0) or 0) if mc else 0
        o_cyc = (opt.get(oc,   0) or 0) if oc else 0
        m_str = f"{m_cyc/total_m*100:.2f}\\%" if m_cyc > 0 else "N/A"
        o_str = f"{o_cyc/total_o*100:.2f}\\%" if o_cyc > 0 else "N/A"
        body += f"    {label} & {m_str} & {o_str} \\\\\n"
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{Per-operation cycle distribution}
    \label{tab:layer_breakdown}
    \begin{tabular}{lcc}\hline
    Operation & Mamba-130M (\%) & OPT-125M (\%) \\\hline
    """ + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_layer_breakdown", tex)


# ──────────────────────────────────────────────────────────────────────────────
# ── NEW LATEX TABLES ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def tab_object_histogram(summary):
    """Object histogram statistics for both models."""
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt|tiny",   case=False, na=False)].iloc[0]
    rows  = [
        ("Total Core\\textrightarrow{}L2 Bytes", "hist_total_bytes",      ".4g"),
        ("Total Request Count",                   "hist_total_count",      ".4g"),
        ("Average Object Size (bytes)",           "hist_avg_object_size",  ".2f"),
        ("Number of Size Classes",               "hist_num_size_classes",   "g"),
        ("Top-1 Size Class (bytes)",             "hist_top1_size_bytes",    "g"),
        ("Top-1 Total Bytes",                    "hist_top1_total_bytes",  ".4g"),
        ("Top-1 Share of Traffic (\\%)",         "hist_top1_bytes_pct",   ".2f"),
        ("Top-2 Size Class (bytes)",             "hist_top2_size_bytes",    "g"),
        ("Top-2 Total Bytes",                    "hist_top2_total_bytes",  ".4g"),
        ("Top-2 Share of Traffic (\\%)",         "hist_top2_bytes_pct",   ".2f"),
    ]
    body = "".join(
        f"    {l} & {fmt(mamba.get(c, float('nan')), f)} & {fmt(opt.get(c, float('nan')), f)} \\\\\n"
        for l, c, f in rows)
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{Object-Size Histogram: Core$\rightarrow$L2 Memory-Request Traffic}
    \label{tab:object_histogram}
    \begin{tabular}{lcc}\hline
    Metric & Mamba-130M & OPT-125M \\\hline
    """ + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_object_histogram", tex)


def tab_l2_detailed(summary):
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt|tiny",   case=False, na=False)].iloc[0]
    rows  = [
        ("Number of Banks",         "l2_num_banks",           "g"),
        ("Number of Ways",          "l2_num_ways",            "g"),
        ("Number of Sets",          "l2_num_sets",            "g"),
        ("Overall Hit Rate",        "l2_overall_hit_rate",   ".4f"),
        ("Overall Miss Rate",       "l2_overall_miss_rate",  ".4f"),
        ("Cache Utilisation (\\%)", "l2_cache_util_pct",     ".2f"),
        ("Total Hits",              "l2_total_hits",          ".4g"),
        ("Total Misses",            "l2_total_misses",        ".4g"),
        ("Total Evictions",         "l2_total_evictions",     ".4g"),
        ("Total Writebacks",        "l2_total_writebacks",    ".4g"),
        ("Min Bank Hit Rate",       "l2_min_bank_hit_rate",  ".4f"),
        ("Max Bank Hit Rate",       "l2_max_bank_hit_rate",  ".4f"),
        ("Hit-Rate Variance",       "l2_hit_rate_variance",  ".4g"),
    ]
    body = "".join(
        f"    {l} & {fmt(mamba.get(c, float('nan')), f)} & {fmt(opt.get(c, float('nan')), f)} \\\\\n"
        for l, c, f in rows)
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{L2 Cache Detailed Statistics}
    \label{tab:l2_detailed}
    \begin{tabular}{lcc}\hline
    Metric & Mamba-130M & OPT-125M \\\hline
    """ + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_l2_detailed", tex)


def tab_dram_detailed(summary):
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt|tiny",   case=False, na=False)].iloc[0]
    rows  = [
        ("Number of Channels",           "dram_num_channels",              "g"),
        ("Total Reads",                  "dram_total_reads",              ".4g"),
        ("Total Writes",                 "dram_total_writes",             ".4g"),
        ("Avg BW Utilisation (\\%)",     "dram_avg_bw_utilization_pct",  ".2f"),
        ("Avg Read Requests/channel",    "dram_avg_read_requests",        ".4g"),
        ("Avg Write Requests/channel",   "dram_avg_write_requests",       ".4g"),
        ("Avg Row Hit Rate (\\%)",       "dram_avg_row_hit_rate_pct",    ".2f"),
        ("Avg Row Miss Rate (\\%)",      "dram_avg_row_miss_rate_pct",   ".2f"),
        ("Avg Row Conflict Rate (\\%)",  "dram_avg_row_conflict_rate_pct",".2f"),
        ("Avg Memory System Cycles",     "dram_avg_memory_system_cycles", ".4g"),
    ]
    body = "".join(
        f"    {l} & {fmt(mamba.get(c, float('nan')), f)} & {fmt(opt.get(c, float('nan')), f)} \\\\\n"
        for l, c, f in rows)
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{DRAM Bandwidth and Request Statistics}
    \label{tab:dram_detailed}
    \begin{tabular}{lcc}\hline
    Metric & Mamba-130M & OPT-125M \\\hline
    """ + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_dram_detailed", tex)


def tab_core_utilisation(summary):
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt|tiny",   case=False, na=False)].iloc[0]
    rows  = [
        ("Systolic Array Util (\\%)",        "compute_avg_systolic_util_pct",     ".4f"),
        ("PE Utilisation (\\%)",             "compute_avg_pe_util_pct",          ".4f"),
        ("Vector Unit Util (\\%)",           "core_avg_vector_unit_util_pct",    ".4f"),
        ("Memory Idle (\\%)",               "compute_avg_memory_idle_pct",      ".4f"),
        ("Core Idle (\\%)",                  "compute_avg_core_idle_pct",        ".4f"),
        ("Avg Matmul Active Cycles",         "core_avg_matmul_active_cycles",    ".4g"),
        ("Avg Total Cycles/core",            "core_avg_total_cycles",            ".4g"),
        ("Avg Tiles Finished/core",          "core_avg_tiles_finished",          ".4g"),
        ("Avg Instructions Executed/core",   "core_avg_instructions_executed",   ".4g"),
        ("Avg Instructions per Tile",        "core_avg_instructions_per_tile",   ".4f"),
        ("Avg Systolic Issue Count",         "core_avg_systolic_inst_issue",     ".4g"),
        ("Avg Systolic Preload Count",       "core_avg_systolic_preload",        ".4g"),
    ]
    body = "".join(
        f"    {l} & {fmt(mamba.get(c, float('nan')), f)} & {fmt(opt.get(c, float('nan')), f)} \\\\\n"
        for l, c, f in rows)
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{Core Utilisation Breakdown}
    \label{tab:core_utilisation}
    \begin{tabular}{lcc}\hline
    Metric & Mamba-130M & OPT-125M \\\hline
    """ + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_core_utilisation", tex)


def tab_sram_detailed(summary):
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt|tiny",   case=False, na=False)].iloc[0]
    rows  = [
        ("Input-SRAM Total Hits",      "sram_total_hits",         ".4g"),
        ("Input-SRAM Total Misses",    "sram_total_misses",       ".4g"),
        ("Input-SRAM Hit Rate",        "sram_hit_rate",          ".6f"),
        ("Input-SRAM Avg Bytes Req",   "sram_avg_bytes_req",     ".4g"),
        ("Input-SRAM Avg Bytes Recv",  "sram_avg_bytes_rcv",     ".4g"),
        ("Acc-SRAM Total Hits",        "acc_sram_total_hits",    ".4g"),
        ("Acc-SRAM Total Misses",      "acc_sram_total_misses",  ".4g"),
        ("Acc-SRAM Hit Rate",          "acc_sram_hit_rate",     ".6f"),
    ]
    body = "".join(
        f"    {l} & {fmt(mamba.get(c, float('nan')), f)} & {fmt(opt.get(c, float('nan')), f)} \\\\\n"
        for l, c, f in rows)
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{On-Chip SRAM and Accumulator-SRAM Statistics}
    \label{tab:sram_detailed}
    \begin{tabular}{lcc}\hline
    Metric & Mamba-130M & OPT-125M \\\hline
    """ + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_sram_detailed", tex)


def tab_seq_scaling_extended(detailed):
    """Extended scaling table including PE util, DRAM BW, L2 hit rate."""
    mamba = detailed[detailed["model"] == "Mamba"].sort_values("seq_len").set_index("seq_len")
    opt   = detailed[detailed["model"] == "OPT|tiny"].sort_values("seq_len").set_index("seq_len")

    def g(df, sl, col):
        try:
            return df.loc[sl, col] if sl in df.index else float("nan")
        except Exception:
            return float("nan")

    header = (
        r"    Seq & \multicolumn{2}{c}{PE Util (\%)} & "
        r"\multicolumn{2}{c}{DRAM BW (\%)} & "
        r"\multicolumn{2}{c}{L2 Hit Rate} \\" + "\n"
        r"    Len & Mamba & OPT & Mamba & OPT & Mamba & OPT \\\hline"
    )
    body = ""
    for sl in SEQ_LENS:
        mp = g(mamba, sl, "compute_avg_pe_util_pct")
        op = g(opt,   sl, "compute_avg_pe_util_pct")
        mb = g(mamba, sl, "dram_avg_bw_utilization_pct")
        ob = g(opt,   sl, "dram_avg_bw_utilization_pct")
        ml = g(mamba, sl, "l2_overall_hit_rate")
        ol = g(opt,   sl, "l2_overall_hit_rate")
        body += (f"    {sl:>5} & {fmt(mp,'.2f')} & {fmt(op,'.2f')} & "
                 f"{fmt(mb,'.2f')} & {fmt(ob,'.2f')} & "
                 f"{fmt(ml,'.4f')} & {fmt(ol,'.4f')} \\\\\n")
    tex = textwrap.dedent(r"""
    \begin{table}[h]\centering
    \caption{Extended Scaling: PE Utilisation, DRAM BW, and L2 Hit Rate vs Sequence Length}
    \label{tab:seq_scaling_extended}
    \begin{tabular}{r cc cc cc}\hline
    """ + header + "\n" + body + r"""\hline\end{tabular}\end{table}""").strip()
    write_tex("tab_seq_scaling_extended", tex)
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Mamba vs OPT thesis plots (expanded)")
    parser.add_argument("--data-dir", default=".", type=Path,
                        help="Directory containing the CSV files (default: current dir)")
    args     = parser.parse_args()
    data_dir = args.data_dir

    print(f"\n{'='*65}")
    print(f"  Mamba-130M vs OPT-125M — Expanded Thesis Figure & Table Generator")
    print(f"{'='*65}\n")
    print(f"  Data dir : {data_dir.resolve()}")
    print(f"  Figures  : {FIGURES.resolve()}")
    print(f"  Tables   : {TABLES.resolve()}\n")

    detailed, per_log, summary = load_data(data_dir)
    per_log = prepare_per_log(per_log)
    mamba_pl, opt_pl = split_models(per_log)

    # ── Original figures ──────────────────────────────────────────────────────
    print("Generating original figures (1–12) …")
    fig1_total_cycles(mamba_pl, opt_pl)
    fig2_wall_clock(mamba_pl, opt_pl)
    fig3_pe_util(mamba_pl, opt_pl)
    fig4_dram_reads(mamba_pl, opt_pl)
    fig5_dram_bw_util(mamba_pl, opt_pl)
    fig6_sram_hitrates(mamba_pl, opt_pl)
    fig7_memory_bound_ratio(detailed)
    fig8_speedup(detailed)
    fig9_sa_util(mamba_pl, opt_pl)
    fig10_memory_idle(mamba_pl, opt_pl)
    m1 = mamba_pl[mamba_pl["seq_len"] == 1]
    o1 = opt_pl[opt_pl["seq_len"] == 1]
    if not m1.empty and not o1.empty:
        fig11_radar(m1, o1)
    else:
        print("  [skip] fig11_radar — no seq_len=1 rows found")
    fig12_layer_pie(summary)

    # ── Object histogram figures ──────────────────────────────────────────────
    print("\nGenerating object histogram figures (13–16) …")
    fig13_hist_total_bytes(mamba_pl, opt_pl)
    fig14_hist_avg_object_size(mamba_pl, opt_pl)
    fig15_hist_top2_bar(mamba_pl, opt_pl)
    fig16_hist_size_class_heatmap(mamba_pl, opt_pl)

    # ── L2 cache figures ──────────────────────────────────────────────────────
    # print("\nGenerating L2 cache figures (17–21) …")
    # fig17_l2_hit_miss_rate(mamba_pl, opt_pl)
    # fig18_l2_cache_util(mamba_pl, opt_pl)
    # fig19_l2_hit_miss_abs(mamba_pl, opt_pl)
    # fig20_l2_evictions_writebacks(mamba_pl, opt_pl)
    # fig21_l2_bank_variance(mamba_pl, opt_pl)

    # ── SRAM figures ──────────────────────────────────────────────────────────
    print("\nGenerating SRAM figures (22–24) …")
    fig22_sram_hits_misses(mamba_pl, opt_pl)
    fig23_sram_bytes(mamba_pl, opt_pl)
    fig24_acc_sram_hits_misses(mamba_pl, opt_pl)

    # ── DRAM figures ──────────────────────────────────────────────────────────
    print("\nGenerating DRAM figures (25–28) …")
    fig25_dram_reads_writes(mamba_pl, opt_pl)
    fig26_dram_req_counts(mamba_pl, opt_pl)
    fig27_dram_row_rates(mamba_pl, opt_pl)
    fig28_dram_channels_bw(mamba_pl, opt_pl)

    # ── Core utilisation figures ───────────────────────────────────────────────
    print("\nGenerating core utilisation figures (29–34) …")
    fig29_core_util_stack(mamba_pl, opt_pl)
    fig30_core_idle_breakdown(mamba_pl, opt_pl)
    fig31_inst_per_tile(mamba_pl, opt_pl)
    fig32_tiles_finished(mamba_pl, opt_pl)
    fig33_systolic_issue_preload(mamba_pl, opt_pl)
    fig34_util_heatmap(mamba_pl, opt_pl)
    fig35_opt_cycle_distribution_all(per_log)

    # ── Original tables ───────────────────────────────────────────────────────
    print("\nGenerating original LaTeX tables …")
    tab_core_stats(summary)
    tab_memory_stats(summary)
    tab_seq_scaling(detailed)
    tab_layer_breakdown(summary)

    # ── New tables ────────────────────────────────────────────────────────────
    print("\nGenerating new LaTeX tables …")
    tab_object_histogram(summary)
    tab_l2_detailed(summary)
    tab_dram_detailed(summary)
    tab_core_utilisation(summary)
    tab_sram_detailed(summary)
    tab_seq_scaling_extended(detailed)

    n_figs  = len(list(FIGURES.glob("*.pdf")))
    n_tabs  = len(list(TABLES.glob("*.tex")))
    print(f"\n✓ Done.  {n_figs} PDF figures, {n_tabs} LaTeX tables.\n")


if __name__ == "__main__":
    main()