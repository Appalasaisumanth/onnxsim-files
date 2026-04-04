"""
plot_mamba_vs_opt.py
────────────────────
Thesis-quality figures and LaTeX tables comparing Mamba-130M vs OPT-125M
across multiple sequence lengths, derived from simulator results.

Outputs (written to ./figures/ and ./tables/):
  Figures
    fig1_total_cycles.pdf/png         – Total cycles vs sequence length
    fig2_wall_clock.pdf/png           – Wall-clock simulation time vs seq len
    fig3_pe_util.pdf/png              – PE utilisation % vs seq len
    fig4_dram_reads.pdf/png           – Total DRAM reads vs seq len
    fig5_dram_bw_util.pdf/png         – DRAM BW utilisation % vs seq len
    fig6_sram_hitrates.pdf/png        – Input-SRAM & Acc-SRAM hit rates
    fig7_memory_bound_ratio.pdf/png   – Memory-bound ratio vs seq len
    fig8_speedup.pdf/png              – Mamba speedup over OPT vs seq len
    fig9_sa_util.pdf/png              – Systolic-array utilisation % vs seq len
    fig10_memory_idle.pdf/png         – Memory-idle % vs seq len
    fig11_radar.pdf/png               – Radar / spider chart at seq_len=1
    fig12_layer_pie.pdf/png           – Layer-cycle breakdown pie charts
  Tables (LaTeX)
    tab_core_stats.tex
    tab_memory_stats.tex
    tab_seq_scaling.tex
    tab_layer_breakdown.tex

Usage:
    python plot_mamba_vs_opt.py [--data-dir /path/to/csvs]

Requirements:
    pip install pandas matplotlib numpy
"""

import argparse
import os
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
FIGURES = HERE / "figures"
TABLES  = HERE / "tables"
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

# Colour palette
C_MAMBA = "#1f77b4"   # blue
C_OPT   = "#ff7f0e"   # orange

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

SEQ_LENS = [1, 2, 128, 256, 512, 1024, 2048]

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: Path):
    """Return (detailed_df, per_log_df, summary_df)."""
    detailed  = pd.read_csv(data_dir / "detailed_summary.csv")
    per_log   = pd.read_csv(data_dir / "results_per_log.csv")
    summary   = pd.read_csv(data_dir / "results_summary.csv")
    return detailed, per_log, summary


def extract_seq_len(log_file: str) -> int:
    """Parse sequence length from log filename, e.g. 'mamba-130m-b2-s128.log' → 128."""
    import re
    m = re.search(r"[_-]s(\d+)\.log", str(log_file))
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)\.log", str(log_file))
    if m:
        return int(m.group(1))
    return -1


def prepare_per_log(per_log: pd.DataFrame) -> pd.DataFrame:
    """Add model & seq_len columns to results_per_log."""
    df = per_log.copy()
    df["seq_len"] = df["log_file"].apply(extract_seq_len)
    # Identify model from log_file path
    df["model"] = df["log_file"].apply(
        lambda x: "Mamba" if "mamba" in str(x).lower() else "OPT"
    )
    df = df[df["seq_len"] > 0].copy()
    df = df.sort_values(["model", "seq_len"]).reset_index(drop=True)
    return df


def split_models(df: pd.DataFrame):
    mamba = df[df["model"] == "Mamba"].sort_values("seq_len")
    opt   = df[df["model"] == "OPT"].sort_values("seq_len")
    return mamba, opt


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def save_fig(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved: figures/{name}.pdf/.png")


def logx_ax(ax, seq_lens=SEQ_LENS):
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Sequence Length")


def add_legend(ax):
    ax.legend(loc="upper left", framealpha=0.9)


# ──────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric_vs_seqlen(mamba, opt, col: str, ylabel: str, title: str,
                          name: str, yscale="linear", annotate_ratio=False):
    """Generic line plot of one metric vs sequence length."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mamba["seq_len"], mamba[col], color=C_MAMBA, marker=MARKER_MAMBA,
            linewidth=2, label="Mamba-130M")
    ax.plot(opt["seq_len"],   opt[col],   color=C_OPT,   marker=MARKER_OPT,
            linewidth=2, label="OPT-125M")

    if annotate_ratio and len(mamba) == len(opt):
        for i, row in enumerate(zip(mamba.itertuples(), opt.itertuples())):
            m_val = getattr(row[0], col.replace("-","_"))
            o_val = getattr(row[1], col.replace("-","_"))
            if o_val > 0:
                ratio = m_val / o_val
                x = mamba.iloc[i]["seq_len"]
                y = max(m_val, o_val)
                ax.annotate(f"{ratio:.1f}×", xy=(x, y), xytext=(0, 6),
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


# ──────────────────────────────────────────────────────────────────────────────
# Individual figures
# ──────────────────────────────────────────────────────────────────────────────

def fig1_total_cycles(mamba, opt):
    plot_metric_vs_seqlen(
        mamba, opt, "timing_total_cycles",
        "Total Cycles",
        "Fig 1 — Total Simulation Cycles vs Sequence Length",
        "fig1_total_cycles", yscale="log", annotate_ratio=True)


def fig2_wall_clock(mamba, opt):
    plot_metric_vs_seqlen(
        mamba, opt, "timing_wall_clock_seconds",
        "Wall-Clock Time (s)",
        "Fig 2 — Simulation Wall-Clock Time vs Sequence Length",
        "fig2_wall_clock", yscale="log", annotate_ratio=True)


def fig3_pe_util(mamba, opt):
    plot_metric_vs_seqlen(
        mamba, opt, "compute_avg_pe_util_pct",
        "PE Utilisation (%)",
        "Fig 3 — Average PE Utilisation vs Sequence Length",
        "fig3_pe_util")


def fig4_dram_reads(mamba, opt):
    plot_metric_vs_seqlen(
        mamba, opt, "dram_total_reads",
        "Total DRAM Reads",
        "Fig 4 — Total DRAM Read Requests vs Sequence Length",
        "fig4_dram_reads", yscale="log", annotate_ratio=True)


def fig5_dram_bw_util(mamba, opt):
    """DRAM BW utilisation — clamp outliers from detailed_summary."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Use per_log data which is cleaner for this metric
    for data, label, color, marker in [(mamba, "Mamba-130M", C_MAMBA, MARKER_MAMBA),
                                        (opt,   "OPT-125M",   C_OPT,   MARKER_OPT)]:
        # cap at 100 to remove obvious sensor artifacts
        bw = data["dram_avg_bw_utilization_pct"].clip(upper=100)
        ax.plot(data["seq_len"], bw, color=color, marker=marker,
                linewidth=2, label=label)

    ax.set_ylabel("DRAM BW Utilisation (%)")
    ax.set_title("Fig 5 — Average DRAM Bandwidth Utilisation vs Sequence Length")
    logx_ax(ax)
    add_legend(ax)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, "fig5_dram_bw_util")


def fig6_sram_hitrates(mamba, opt):
    """Grouped bar chart: input-SRAM and acc-SRAM hit rates at each seq len."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, ylabel, title in [
        (axes[0], "sram_hit_rate",     "Input-SRAM Hit Rate",   "Input-SRAM Hit Rate"),
        (axes[1], "acc_sram_hit_rate", "Acc-SRAM Hit Rate",      "Accumulator-SRAM Hit Rate"),
    ]:
        ax.plot(mamba["seq_len"], mamba[col], color=C_MAMBA, marker=MARKER_MAMBA,
                linewidth=2, label="Mamba-130M")
        ax.plot(opt["seq_len"],   opt[col],   color=C_OPT,   marker=MARKER_OPT,
                linewidth=2, label="OPT-125M")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        logx_ax(ax)
        ax.legend(framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.suptitle("Fig 6 — On-Chip SRAM Hit Rates vs Sequence Length", fontsize=FONT_TITLE)
    fig.tight_layout()
    save_fig(fig, "fig6_sram_hitrates")


def fig7_memory_bound_ratio(detailed):
    """Memory-bound ratio from detailed_summary (log scale)."""
    mamba = detailed[detailed["model"] == "Mamba"].sort_values("seq_len")
    opt   = detailed[detailed["model"] == "OPT"].sort_values("seq_len")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Clamp extreme outliers
    mamba_r = mamba["memory_bound_ratio"].clip(upper=2e5)
    opt_r   = opt["memory_bound_ratio"].clip(upper=2e5)
    ax.plot(mamba["seq_len"], mamba_r, color=C_MAMBA, marker=MARKER_MAMBA,
            linewidth=2, label="Mamba-130M")
    ax.plot(opt["seq_len"],   opt_r,   color=C_OPT,   marker=MARKER_OPT,
            linewidth=2, label="OPT-125M")
    ax.set_yscale("log")
    ax.set_ylabel("Memory-Bound Ratio (cycles/byte)")
    ax.set_title("Fig 7 — Memory-Bound Ratio vs Sequence Length")
    logx_ax(ax)
    add_legend(ax)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, "fig7_memory_bound_ratio")


def fig8_speedup(detailed):
    """Mamba speedup over OPT (total cycles ratio OPT/Mamba)."""
    mamba = detailed[detailed["model"] == "Mamba"].sort_values("seq_len").set_index("seq_len")
    opt   = detailed[detailed["model"] == "OPT"].sort_values("seq_len").set_index("seq_len")
    common = mamba.index.intersection(opt.index)
    speedup = opt.loc[common, "timing_total_cycles"] / mamba.loc[common, "timing_total_cycles"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(common, speedup.values, color=C_MAMBA, marker=MARKER_MAMBA,
            linewidth=2, label="Mamba-130M speedup")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="OPT baseline")
    ax.set_ylabel("Speedup (OPT Cycles / Mamba Cycles)")
    ax.set_title("Fig 8 — Mamba Cycle-Count Speedup over OPT vs Sequence Length")
    logx_ax(ax)
    add_legend(ax)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    for x, y in zip(common, speedup.values):
        ax.annotate(f"{y:.2f}×", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig8_speedup")


def fig9_sa_util(mamba, opt):
    plot_metric_vs_seqlen(
        mamba, opt, "compute_avg_systolic_util_pct",
        "Systolic-Array Utilisation (%)",
        "Fig 9 — Systolic-Array Utilisation vs Sequence Length",
        "fig9_sa_util")


def fig10_memory_idle(mamba, opt):
    plot_metric_vs_seqlen(
        mamba, opt, "compute_avg_memory_idle_pct",
        "Memory-Idle (%)",
        "Fig 10 — Average Memory-Idle Percentage vs Sequence Length",
        "fig10_memory_idle")


def fig11_radar(mamba_s1, opt_s1):
    """Radar chart comparing key normalised metrics at seq_len=1."""
    metrics = [
        ("PE Util %",       "compute_avg_pe_util_pct",       "max"),
        ("SA Util %",       "compute_avg_systolic_util_pct", "max"),
        ("SRAM Hit",        "sram_hit_rate",                 "max"),
        ("Acc-SRAM Hit",    "acc_sram_hit_rate",             "max"),
        ("Mem-Idle %",      "compute_avg_memory_idle_pct",   "min"),  # lower = better compute
        ("DRAM BW Util %",  "dram_avg_bw_utilization_pct",   "max"),
    ]
    labels  = [m[0] for m in metrics]
    N = len(labels)

    def get_vals(row):
        return [row[m[1]] for m in metrics]

    m_vals = np.array(get_vals(mamba_s1.iloc[0]))
    o_vals = np.array(get_vals(opt_s1.iloc[0]))

    # Normalise each metric to [0,1] across both models
    combined = np.stack([m_vals, o_vals])
    mn = combined.min(axis=0)
    mx = combined.max(axis=0)
    denom = np.where((mx - mn) == 0, 1, mx - mn)
    m_norm = (m_vals - mn) / denom
    o_norm = (o_vals - mn) / denom

    # Invert "min-is-better" axes so outward = better
    for i, m in enumerate(metrics):
        if m[2] == "min":
            m_norm[i] = 1 - m_norm[i]
            o_norm[i] = 1 - o_norm[i]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    m_norm  = np.append(m_norm,  m_norm[0])
    o_norm  = np.append(o_norm,  o_norm[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, m_norm, color=C_MAMBA, linewidth=2, label="Mamba-130M")
    ax.fill(angles, m_norm, color=C_MAMBA, alpha=0.20)
    ax.plot(angles, o_norm, color=C_OPT,   linewidth=2, label="OPT-125M")
    ax.fill(angles, o_norm, color=C_OPT,   alpha=0.20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("Fig 11 — Normalised Hardware Metrics (seq_len=1)", pad=20,
                 fontsize=FONT_TITLE)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, "fig11_radar")


def fig12_layer_pie(summary: pd.DataFrame):
    """Pie chart of per-layer cycle contribution for each model (7-step summary)."""
    mamba_row = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt_row   = summary[summary["log_file"].str.contains("opt",   case=False, na=False)].iloc[0]

    mamba_layers = {
    # ---- Core Mamba projections ----
    "Mamba_in_proj":  mamba_row.get("layer_mamba_in_proj_cycles", 0) or 0,
    "Mamba_x_proj":   mamba_row.get("layer_mamba_x_proj_cycles", 0) or 0,
    "Mamba_dt_proj":  mamba_row.get("layer_mamba_dt_proj_cycles", 0) or 0,
    "Mamba_out_proj": mamba_row.get("layer_mamba_out_proj_cycles", 0) or 0,
    "Mamba_ssm":      mamba_row.get("layer_mamba_ssm_cycles", 0) or 0,

    # ---- Linear (GEMM) ----
    "GEMM":           mamba_row.get("layer_gemm_cycles", 0) or 0,

    # ---- Elementwise ops ----
    "ElemMul":        mamba_row.get("layer_elementwise_mul_cycles", 0) or 0,
    "ElemAdd":        mamba_row.get("layer_elementwise_add_cycles", 0) or 0,
    "ElemExp":        mamba_row.get("layer_elementwise_exp_cycles", 0) or 0,
    "ElemDiv":        mamba_row.get("layer_elementwise_div_cycles", 0) or 0,
    "ElemSub":        mamba_row.get("layer_elementwise_sub_cycles", 0) or 0,
    "ElemNeg":        mamba_row.get("layer_elementwise_neg_cycles", 0) or 0,

    # ---- Activations ----
    "Sigmoid":        mamba_row.get("layer_activation_sigmoid_cycles", 0) or 0,
    "SiLU":           mamba_row.get("layer_activation_silu_cycles", 0) or 0,

    # ---- Data movement / tensor ops ----
    "Slice":          mamba_row.get("layer_slice_cycles", 0) or 0,
    "Concat":         mamba_row.get("layer_concat_cycles", 0) or 0,
    "Expand":         mamba_row.get("layer_expand_cycles", 0) or 0,
    "Repeat":         mamba_row.get("layer_repeat_cycles", 0) or 0,
    "Reshape":        mamba_row.get("layer_reshape_cycles", 0) or 0,

    # ---- Fallback ----
    "Other":          mamba_row.get("layer_other_cycles", 0) or 0,
}
    opt_layers = {
        "FFN FC2":         opt_row.get("layer_ffn_fc2_cycles", 0) or 0,
        "FFN FC1":         opt_row.get("layer_ffn_fc1_cycles", 0) or 0,
        "QKV Proj":        opt_row.get("layer_QKV_projection_cycles", 0) or 0,
        "Attn Proj":       opt_row.get("layer_attn_projection_cycles", 0) or 0,
        "Attention":       opt_row.get("layer_attention_cycles", 0) or 0,
        "Other":           opt_row.get("layer_other_cycles", 0) or 0,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, layers, title, base_color in [
        (axes[0], mamba_layers, "Mamba-130M Layer Cycle Distribution", C_MAMBA),
        (axes[1], opt_layers,   "OPT-125M Layer Cycle Distribution",   C_OPT),
    ]:
        labels = list(layers.keys())
        sizes  = list(layers.values())
        total  = sum(sizes)
        sizes_pct = [s / total * 100 if total > 0 else 0 for s in sizes]
        # Filter zero slices
        labels_f = [l for l, s in zip(labels, sizes) if s > 0]
        sizes_f  = [s for s in sizes if s > 0]
        if not sizes_f:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue
        wedge_props = dict(linewidth=0.8, edgecolor="white")
        ax.pie(sizes_f, labels=labels_f, autopct="%1.1f%%",
               startangle=140, wedgeprops=wedge_props)
        ax.set_title(title, fontsize=FONT_TITLE)

    fig.suptitle("Fig 12 — Per-Layer Cycle Breakdown (7-step simulation)", fontsize=FONT_TITLE)
    fig.tight_layout()
    save_fig(fig, "fig12_layer_pie")


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table generators
# ──────────────────────────────────────────────────────────────────────────────

def fmt(val, fmt_str=".3g"):
    """Format a number for LaTeX."""
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


def tab_core_stats(summary: pd.DataFrame):
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt",   case=False, na=False)].iloc[0]

    rows = [
        ("Total Cycles",                    "timing_total_cycles",              ".6g"),
        ("Simulation Time (\\textmu s)",     "timing_total_us",                  ".6g"),
        ("Total Compute Cycles",            "timing_total_compute_cycles",       ".4g"),
        ("Wall-Clock Sim Time (s)",         "timing_wall_clock_seconds",        ".2f"),
        ("Total Tiles",                     "timing_total_tiles",               ".4g"),
        ("Tiles per Second (TPS)",          "timing_tiles_per_second",          ".4f"),
        ("Systolic Array Util (\\%)",       "compute_avg_systolic_util_pct",    ".4f"),
        ("PE Utilisation (\\%)",            "compute_avg_pe_util_pct",          ".4f"),
        ("Avg Memory-Idle (\\%)",           "compute_avg_memory_idle_pct",      ".4f"),
        ("Avg Core-Idle (\\%)",             "compute_avg_core_idle_pct",        ".4f"),
        ("Total Compute (GFLOPs)",          "compute_total_GFLOPs",             ".4f"),
        ("GEMM Memory-Bound (\\%)",         "compute_gemm_memory_bound_pct",    ".1f"),
        ("GEMM Avg Arithmetic Intensity",   "compute_gemm_avg_arithmetic_intensity", ".4g"),
    ]
    body = ""
    for label, col, fstr in rows:
        m_val = mamba.get(col, float("nan"))
        o_val = opt.get(col,   float("nan"))
        body += f"        {label} & {fmt(m_val, fstr)} & {fmt(o_val, fstr)} \\\\\n"

    tex = textwrap.dedent(r"""
    \begin{table}[h]
    \centering
    \caption{Core Execution Statistics (7-step simulation)}
    \label{tab:core_stats_gen}
    \begin{tabular}{lcc}
    \hline
    Metric & Mamba-130M & OPT-125M \\
    \hline
    """ + body + r"""\hline
    \end{tabular}
    \end{table}
    """).strip()
    write_tex("tab_core_stats", tex)


def tab_memory_stats(summary: pd.DataFrame):
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt",   case=False, na=False)].iloc[0]

    rows = [
        ("Weight Size (GB)",            "model_weight_size_GB",         ".3f"),
        ("Scratchpad Size (KB)",        "hw_spad_size_KB",              "g"),
        ("Accumulation Buffer (KB)",    "hw_accumulator_size_KB",       "g"),
        ("Input SRAM Hit Rate",         "sram_hit_rate",                ".4f"),
        ("Acc-SRAM Hit Rate",           "acc_sram_hit_rate",            ".6f"),
        ("Total DRAM Reads",            "dram_total_reads",             ".5g"),
        ("Total DRAM Writes",           "dram_total_writes",            ".5g"),
        ("DRAM Channels",               "dram_num_channels",            "g"),
        ("Avg DRAM BW Utilisation (\\%)", "dram_avg_bw_utilization_pct", ".2f"),
        ("Avg Row Hit Rate (\\%)",      "dram_avg_row_hit_rate_pct",    ".2f"),
        ("Avg Row Conflict Rate (\\%)", "dram_avg_row_conflict_rate_pct", ".2f"),
        ("Avg Memory System Cycles",    "dram_avg_memory_system_cycles","g"),
        ("L2 Overall Hit Rate",         "l2_overall_hit_rate",          ".4f"),
        ("L2 Cache Utilisation (\\%)",  "l2_cache_util_pct",            ".2f"),
        ("L2 Total Misses",             "l2_total_misses",              "g"),
    ]
    body = ""
    for label, col, fstr in rows:
        m_val = mamba.get(col, float("nan"))
        o_val = opt.get(col,   float("nan"))
        body += f"        {label} & {fmt(m_val, fstr)} & {fmt(o_val, fstr)} \\\\\n"

    tex = textwrap.dedent(r"""
    \begin{table}[h]
    \centering
    \caption{Memory Hierarchy Statistics (7-step simulation)}
    \label{tab:memory_stats_gen}
    \begin{tabular}{lcc}
    \hline
    Metric & Mamba-130M & OPT-125M \\
    \hline
    """ + body + r"""\hline
    \end{tabular}
    \end{table}
    """).strip()
    write_tex("tab_memory_stats", tex)


def tab_seq_scaling(detailed: pd.DataFrame):
    """Table of total cycles and wall-clock time for both models across seq lens."""
    mamba = detailed[detailed["model"] == "Mamba"].sort_values("seq_len").set_index("seq_len")
    opt   = detailed[detailed["model"] == "OPT"].sort_values("seq_len").set_index("seq_len")

    header = (
        r"    Seq Len & \multicolumn{2}{c}{Total Cycles} & "
        r"\multicolumn{2}{c}{Wall-Clock Time (s)} & Speedup \\" + "\n"
        r"            & Mamba & OPT & Mamba & OPT & (OPT/Mamba) \\" + "\n"
        r"    \hline"
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
        body += (
            f"    {sl:>5} & {mc:.3e} & {oc:.3e} & "
            f"{mw:.1f} & {ow:.1f} & {sp:.3f} \\\\\n"
        )

    tex = textwrap.dedent(r"""
    \begin{table}[h]
    \centering
    \caption{Cycle-count and wall-clock time scaling with sequence length}
    \label{tab:seq_scaling}
    \begin{tabular}{r cc cc c}
    \hline
    """ + header + "\n" + body + r"""\hline
    \end{tabular}
    \end{table}
    """).strip()
    write_tex("tab_seq_scaling", tex)


def tab_layer_breakdown(summary: pd.DataFrame):
    """Cycle breakdown by operation."""
    mamba = summary[summary["log_file"].str.contains("mamba", case=False, na=False)].iloc[0]
    opt   = summary[summary["log_file"].str.contains("opt",   case=False, na=False)].iloc[0]

    total_m = mamba.get("timing_total_cycles", 1) or 1
    total_o = opt.get("timing_total_cycles",   1) or 1

    ops = [
        ("FFN FC2",        None,                              "layer_ffn_fc2_cycles"),
        ("FFN FC1",        None,                              "layer_ffn_fc1_cycles"),
        ("QKV Projection", None,                              "layer_QKV_projection_cycles"),
        ("Attn Projection",None,                              "layer_attn_projection_cycles"),
        ("Attention",      None,                              "layer_attention_cycles"),
    ("GEMM",              "layer_gemm_cycles",                None),

    ("Mamba InProj",      "layer_mamba_in_proj_cycles",       None),
    ("Mamba XProj",       "layer_mamba_x_proj_cycles",        None),
    ("Mamba DtProj",      "layer_mamba_dt_proj_cycles",       None),
    ("Mamba OutProj",     "layer_mamba_out_proj_cycles",      None),
    ("Mamba SSM",         "layer_mamba_ssm_cycles",           None),

    ("Elementwise Mul",   "layer_elementwise_mul_cycles",     None),
    ("Elementwise Add",   "layer_elementwise_add_cycles",     None),
    ("Elementwise Exp",   "layer_elementwise_exp_cycles",     None),
    ("Elementwise Div",   "layer_elementwise_div_cycles",     None),
    ("Elementwise Sub",   "layer_elementwise_sub_cycles",     None),
    ("Elementwise Neg",   "layer_elementwise_neg_cycles",     None),

    ("Activation Sigmoid","layer_activation_sigmoid_cycles",  None),
    ("Activation SiLU",   "layer_activation_silu_cycles",     None),

    ("Slice",             "layer_slice_cycles",               None),
    ("Concat",            "layer_concat_cycles",              None),
    ("Expand",            "layer_expand_cycles",              None),
    ("Repeat",            "layer_repeat_cycles",              None),
    ("Reshape",           "layer_reshape_cycles",             None),

    ("Other",             "layer_other_cycles",               "layer_other_cycles")
    ]

    body = ""
    for label, mamba_col, opt_col in ops:
        m_cyc = mamba.get(mamba_col, 0) if mamba_col else 0
        o_cyc = opt.get(opt_col,   0) if opt_col else 0
        m_cyc = m_cyc or 0
        o_cyc = o_cyc or 0
        m_pct = m_cyc / total_m * 100
        o_pct = o_cyc / total_o * 100
        m_str = f"{m_pct:.2f}\\%" if m_cyc > 0 else "N/A"
        o_str = f"{o_pct:.2f}\\%" if o_cyc > 0 else "N/A"
        body += f"    {label} & {m_str} & {o_str} \\\\\n"

    tex = textwrap.dedent(r"""
    \begin{table}[h]
    \centering
    \caption{Per-operation cycle distribution}
    \label{tab:layer_breakdown}
    \begin{tabular}{lcc}
    \hline
    Operation & Mamba-130M Cycles (\%) & OPT-125M Cycles (\%) \\
    \hline
    """ + body + r"""\hline
    \end{tabular}
    \end{table}
    """).strip()
    write_tex("tab_layer_breakdown", tex)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Mamba vs OPT thesis plots")
    parser.add_argument("--data-dir", default=".", type=Path,
                        help="Directory containing the CSV files (default: current dir)")
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"\n{'='*60}")
    print(f"  Mamba-130M vs OPT-125M — Thesis Figure & Table Generator")
    print(f"{'='*60}\n")
    print(f"  Data dir : {data_dir.resolve()}")
    print(f"  Figures  : {FIGURES.resolve()}")
    print(f"  Tables   : {TABLES.resolve()}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    detailed, per_log, summary = load_data(data_dir)
    per_log = prepare_per_log(per_log)
    mamba_pl, opt_pl = split_models(per_log)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures …")
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
    # Radar at seq_len=1
    m1 = mamba_pl[mamba_pl["seq_len"] == 1]
    o1 = opt_pl[opt_pl["seq_len"] == 1]
    if not m1.empty and not o1.empty:
        fig11_radar(m1, o1)
    else:
        print("  [skip] fig11_radar — no seq_len=1 rows found")
    fig12_layer_pie(summary)

    # ── Tables ────────────────────────────────────────────────────────────────
    print("\nGenerating LaTeX tables …")
    tab_core_stats(summary)
    tab_memory_stats(summary)
    tab_seq_scaling(detailed)
    tab_layer_breakdown(summary)

    print(f"\n✓ Done.  {len(list(FIGURES.glob('*.pdf')))} PDFs, "
          f"{len(list(TABLES.glob('*.tex')))} TeX tables.\n")


if __name__ == "__main__":
    main()