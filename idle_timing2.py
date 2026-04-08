"""
plot_idle_timing_v2.py
======================
Correct parser + timing-diagram plotter for simulator logs.

KEY FIX (vs old script):
  _stat_memory_idle_cycle and _stat_idle_cycle in the C++ code are CUMULATIVE
  accumulators reset only at layer boundaries (via update_stats() called from
  print_stats() at layer end).  print_current_stats() (every 100 cycles) logs
  the running totals — so the correct per-interval metric is the DELTA between
  consecutive readings, divided by the print interval (100 cycles by default).

  Old bug: divided raw value by absolute cycle → gave ever-shrinking %, even
  when the core was 100 % idle.

Usage:
    python plot_idle_timing_v2.py                       # expects logs in cwd
    python plot_idle_timing_v2.py --log-dir /path/logs  # explicit directory
    python plot_idle_timing_v2.py --out-dir /tmp/plots  # output directory
"""

import re, os, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_LOGS = {
    "mamba_130m": "mamba_130m.log",
    "tiny_opt":   "tiny_opt.log",
}

# ── Colour palettes ────────────────────────────────────────────────────────────
CORE_COLORS   = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E"]
LAYER_PALETTE = [
    "#534AB7","#0F6E56","#993C1D","#993556","#185FA5","#3B6D11","#854F0B","#A32D2D",
    "#3C3489","#085041","#712B13","#72243E","#0C447C","#27500A","#633806","#791F1F",
    "#7F77DD","#1D9E75","#D85A30","#D4537E","#378ADD","#639922","#BA7517","#E24B4A",
    "#AFA9EC","#5DCAA5","#F0997B","#ED93B1","#85B7EB","#97C459","#EF9F27","#F09595",
    "#EEEDFE","#E1F5EE","#FAECE7","#FBEAF0",
]

# ── Regex patterns ─────────────────────────────────────────────────────────────
RE_LAYER   = re.compile(r"Start layer (.+)")
RE_IDLE    = re.compile(
    r"Core \[(\d+)\] : Memory unit idle cycle (\d+)"
    r" Systolic bubble cycle (\d+) Core idle cycle (\d+)")
RE_UTIL    = re.compile(
    r"Core \[(\d+)\].*?Systolic Array Utilization\(%\)\s*([\d.]+)"
    r".*?Vector Unit Utilization\(%\)\s*([\d.]+)"
    r".*?Total cycle:\s*(\d+)")
RE_CYCLE   = re.compile(r"cycle: \[(\d+)\]")


# ── Parser ─────────────────────────────────────────────────────────────────────
def parse_log(path):
    """
    Returns list of dicts, one per (core, interval).
    Each dict has:
        core, cycle, layer,
        mem_idle_pct   – % of last interval where memory was idle    [0-100]
        core_idle_pct  – % of last interval where core was idle      [0-100]
        sa_util_pct    – SA utilisation % of last interval (from log directly)
        vec_util_pct   – vector unit util % (from log directly)
        bubble_pct     – systolic bubble % of last interval          [0-100]
    """
    records = []

    # Per-core state between prints
    # We keep: last raw cumulative counter values and the cycle at that point
    prev   = {}   # core -> {mem, ci, bubble, cy}
    staged = {}   # core -> {mem_raw, ci_raw, bubble_raw, sa_util, vec_util, layer, abs_cycle}
    current_layer = "unknown"

    def reset_prev():
        nonlocal prev, staged
        prev   = {}
        staged = {}

    with open(path) as fh:
        for line in fh:

            # ── New layer → reset cumulative baselines ─────────────────────
            m = RE_LAYER.match(line.split("] [info] ", 1)[-1].strip() if "] [info] " in line else line.strip())
            if not m:
                m = RE_LAYER.search(line)
            if m:
                current_layer = m.group(1).strip()
                reset_prev()
                continue

            # ── Idle line: captures raw cumulative counts ──────────────────
            m = RE_IDLE.search(line)
            if m:
                core   = int(m.group(1))
                mem    = int(m.group(2))
                bubble = int(m.group(3))
                ci     = int(m.group(4))
                staged.setdefault(core, {}).update({
                    "mem_raw":    mem,
                    "ci_raw":     ci,
                    "bubble_raw": bubble,
                    "layer":      current_layer,
                })
                continue

            # ── Util line: captures SA/vec util and absolute cycle ─────────
            m = RE_UTIL.search(line)
            if m:
                core      = int(m.group(1))
                sa_util   = float(m.group(2))   # already % of interval
                vec_util  = float(m.group(3))   # already % of interval
                abs_cycle = int(m.group(4))
                staged.setdefault(core, {}).update({
                    "sa_util":   sa_util,
                    "vec_util":  vec_util,
                    "abs_cycle": abs_cycle,
                })
                continue

            # ── Cycle line: signals end of this core's stats block ─────────
            m = RE_CYCLE.search(line)
            if m:
                abs_cycle_tag = int(m.group(1))
                # Find which core just had its block terminated
                # Strategy: the most recently staged core without a cycle tag yet
                # Since each core emits: idle → util → cycle in order,
                # the last staged core that has abs_cycle matching this tag is the one
                for core, s in list(staged.items()):
                    s_cycle = s.get("abs_cycle", abs_cycle_tag)
                    if s_cycle != abs_cycle_tag and abs_cycle_tag != s_cycle:
                        # fall back: accept any staged that hasn't been flushed
                        pass

                    # Compute deltas vs previous reading for this core
                    p          = prev.get(core, {})
                    raw_mem    = s.get("mem_raw",    0)
                    raw_ci     = s.get("ci_raw",     0)
                    raw_bubble = s.get("bubble_raw", 0)
                    p_mem      = p.get("mem",    0)
                    p_ci       = p.get("ci",     0)
                    p_bubble   = p.get("bubble", 0)
                    p_cy       = p.get("cy",     0)

                    interval   = abs_cycle_tag - p_cy if abs_cycle_tag > p_cy else 100
                    interval   = max(interval, 1)

                    d_mem    = max(0, raw_mem    - p_mem)
                    d_ci     = max(0, raw_ci     - p_ci)
                    d_bubble = max(0, raw_bubble - p_bubble)

                    records.append({
                        "core":          core,
                        "cycle":         abs_cycle_tag,
                        "layer":         s.get("layer", current_layer),
                        # ── CORRECT METRICS: delta / interval * 100 ────────
                        "mem_idle_pct":  round(min(d_mem    / interval * 100, 100), 2),
                        "core_idle_pct": round(min(d_ci     / interval * 100, 100), 2),
                        "bubble_pct":    round(min(d_bubble / interval * 100, 100), 2),
                        # ── SA/vec util are already % of interval in log ───
                        "sa_util_pct":   round(min(s.get("sa_util",  0), 100), 2),
                        "vec_util_pct":  round(min(s.get("vec_util", 0), 100), 2),
                    })

                    prev[core] = {
                        "mem":    raw_mem,
                        "ci":     raw_ci,
                        "bubble": raw_bubble,
                        "cy":     abs_cycle_tag,
                    }

                staged.clear()

    return records


# ── Layer-band helper ──────────────────────────────────────────────────────────
def build_layer_bands(by_core, all_cycles, ref_core):
    """Return list of (start_cycle, end_cycle, layer_name) from ref_core's timeline."""
    c0 = by_core.get(ref_core, next(iter(by_core.values())))
    bands, prev_layer, band_start = [], None, None
    for cy in all_cycles:
        lay = c0.get(cy, {}).get("layer", prev_layer)
        if lay != prev_layer:
            if prev_layer is not None and band_start is not None:
                bands.append((band_start, cy, prev_layer))
            band_start = cy
            prev_layer = lay
    if prev_layer and band_start is not None:
        bands.append((band_start, all_cycles[-1], prev_layer))
    return bands


# ── Timing-diagram plot ────────────────────────────────────────────────────────
METRIC_LABELS = {
    "mem_idle_pct":  "Memory idle %",
    "core_idle_pct": "Core idle %",
    "sa_util_pct":   "SA utilisation %",
    "vec_util_pct":  "Vector util %",
    "bubble_pct":    "Systolic bubble %",
}

def plot_timing(records, model_name, metric="mem_idle_pct", out_path=None):
    if not records:
        print(f"[WARN] No records for {model_name}")
        return

    # Ordered unique layers
    seen, layers = set(), []
    for r in records:
        if r["layer"] not in seen:
            seen.add(r["layer"]); layers.append(r["layer"])
    layer_color = {l: LAYER_PALETTE[i % len(LAYER_PALETTE)] for i, l in enumerate(layers)}

    cores       = sorted(set(r["core"] for r in records))
    all_cycles  = sorted(set(r["cycle"] for r in records))
    by_core     = {c: {} for c in cores}
    for r in records:
        by_core[r["core"]][r["cycle"]] = {"val": r[metric], "layer": r["layer"]}

    bands      = build_layer_bands(by_core, all_cycles, cores[0])
    met_label  = METRIC_LABELS.get(metric, metric)
    y_max      = 105

    fig, axes = plt.subplots(
        len(cores), 1,
        figsize=(20, 3 * len(cores)),
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )
    if len(cores) == 1:
        axes = [axes]

    for ax, core in zip(axes, cores):
        # Layer background shading
        for bstart, bend, blayer in bands:
            ax.axvspan(bstart, bend, alpha=0.08, color=layer_color[blayer], linewidth=0)

        # Layer boundary lines + labels (top subplot only)
        drawn = set()
        for bstart, _, blayer in bands:
            ax.axvline(bstart, color="0.80", linewidth=0.4, zorder=1)
            if core == cores[0] and blayer not in drawn:
                ax.text(bstart, y_max - 1, blayer, fontsize=5.5, rotation=45,
                        va="bottom", ha="left", color=layer_color[blayer], clip_on=True)
                drawn.add(blayer)

        # Data line + fill
        cdata = by_core[core]
        xs    = sorted(cdata.keys())
        ys    = [cdata[cy]["val"] for cy in xs]
        ax.plot(xs, ys, color=CORE_COLORS[core % len(CORE_COLORS)], linewidth=1.2, zorder=3)
        ax.fill_between(xs, ys, alpha=0.15, color=CORE_COLORS[core % len(CORE_COLORS)], zorder=2)

        # 50% reference line
        ax.axhline(50, color="0.70", linewidth=0.5, linestyle="--", zorder=1)

        ax.set_ylabel(f"Core {core}\n{met_label}", fontsize=8)
        ax.set_ylim(0, y_max)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_tick_params(labelsize=7)
        ax.grid(axis="y", color="0.90", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Simulation cycle", fontsize=9)
    axes[-1].xaxis.set_tick_params(labelsize=8)
    axes[-1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v)))
    )

    # Layer legend
    patches = [mpatches.Patch(color=layer_color[l], alpha=0.7, label=l) for l in layers]
    fig.legend(handles=patches, title="Layers", loc="upper right",
               bbox_to_anchor=(1.0, 1.0), fontsize=6, title_fontsize=7,
               ncol=max(1, len(layers) // 12 + 1), framealpha=0.9)
    fig.suptitle(f"{model_name}  ·  {met_label}  per core  (delta / interval)",
                 fontsize=12, y=1.01)

    if out_path is None:
        out_path = f"{model_name}_{metric}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Heatmap (mean per layer per core) ─────────────────────────────────────────
def plot_heatmap(records, model_name, metric="mem_idle_pct", out_path=None):
    if not records:
        return

    seen, layers = set(), []
    for r in records:
        if r["layer"] not in seen:
            seen.add(r["layer"]); layers.append(r["layer"])
    cores  = sorted(set(r["core"] for r in records))
    sums   = defaultdict(float)
    counts = defaultdict(int)
    for r in records:
        k = (r["core"], r["layer"])
        sums[k] += r[metric]; counts[k] += 1

    data = np.array([[sums[(c, l)] / counts[(c, l)] if counts[(c, l)] else 0
                      for l in layers] for c in cores])

    met_label = METRIC_LABELS.get(metric, metric)
    fig, ax   = plt.subplots(figsize=(max(8, len(layers) * 0.6), max(2.5, len(cores) * 0.8)))
    im        = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_yticks(range(len(cores)))
    ax.set_yticklabels([f"Core {c}" for c in cores], fontsize=9)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{model_name}  ·  Mean {met_label} per layer × core", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(f"Mean {met_label}", fontsize=8)

    for ri in range(len(cores)):
        for ci in range(len(layers)):
            v = data[ri, ci]
            ax.text(ci, ri, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if v > 60 else "black")

    plt.tight_layout()
    if out_path is None:
        out_path = f"{model_name}_{metric}_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Summary overview: all metrics stacked for one model ───────────────────────
def plot_overview(records, model_name, out_path=None):
    """4-panel overview: mem_idle, core_idle, SA util, bubble — for all cores."""
    if not records:
        return

    metrics = [
        ("mem_idle_pct",  "Memory idle %",        "YlOrRd"),
        ("core_idle_pct", "Core idle %",           "PuRd"),
        ("sa_util_pct",   "SA utilisation %",      "Blues"),
        ("bubble_pct",    "Systolic bubble %",     "Oranges"),
    ]

    seen, layers = set(), []
    for r in records:
        if r["layer"] not in seen:
            seen.add(r["layer"]); layers.append(r["layer"])
    layer_color = {l: LAYER_PALETTE[i % len(LAYER_PALETTE)] for i, l in enumerate(layers)}

    cores      = sorted(set(r["core"] for r in records))
    all_cycles = sorted(set(r["cycle"] for r in records))
    by_core    = {c: {} for c in cores}
    for r in records:
        by_core[r["core"]][r["cycle"]] = r

    bands = build_layer_bands(
        {c: {cy: {"layer": by_core[c][cy]["layer"]} for cy in by_core[c]} for c in cores},
        all_cycles, cores[0])

    fig, axes = plt.subplots(len(metrics), 1, figsize=(22, 3.5 * len(metrics)),
                             sharex=True, gridspec_kw={"hspace": 0.10})

    for ax, (metric, label, _) in zip(axes, metrics):
        for bstart, bend, blayer in bands:
            ax.axvspan(bstart, bend, alpha=0.06, color=layer_color[blayer], linewidth=0)
        drawn = set()
        for bstart, _, blayer in bands:
            ax.axvline(bstart, color="0.82", linewidth=0.35, zorder=1)
            if blayer not in drawn:
                ax.text(bstart, 103, blayer, fontsize=5, rotation=45,
                        va="bottom", ha="left", color=layer_color[blayer], clip_on=True)
                drawn.add(blayer)

        for core in cores:
            cdata = by_core[core]
            xs    = sorted(cdata.keys())
            ys    = [cdata[cy].get(metric, 0) for cy in xs]
            ax.plot(xs, ys, color=CORE_COLORS[core % len(CORE_COLORS)],
                    linewidth=1.0, label=f"Core {core}", zorder=3)

        ax.axhline(50, color="0.70", linewidth=0.5, linestyle="--")
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(0, 107)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_tick_params(labelsize=7)
        ax.grid(axis="y", color="0.90", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # Core legend
    core_handles = [mpatches.Patch(color=CORE_COLORS[c % len(CORE_COLORS)], label=f"Core {c}")
                    for c in cores]
    axes[0].legend(handles=core_handles, fontsize=7, loc="upper left", framealpha=0.8)

    axes[-1].set_xlabel("Simulation cycle", fontsize=9)
    axes[-1].xaxis.set_tick_params(labelsize=8)
    axes[-1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v))))

    # Layer legend
    patches = [mpatches.Patch(color=layer_color[l], alpha=0.7, label=l) for l in layers]
    fig.legend(handles=patches, title="Layers", loc="upper right",
               bbox_to_anchor=(1.01, 1.0), fontsize=6, title_fontsize=7,
               ncol=max(1, len(layers) // 10 + 1), framealpha=0.9)

    fig.suptitle(f"{model_name}  ·  All metrics overview  (delta / interval)",
                 fontsize=13, y=1.01)

    if out_path is None:
        out_path = f"{model_name}_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for model_name, fname in DEFAULT_LOGS.items():
        log_path = os.path.join(args.log_dir, fname)
        if not os.path.exists(log_path):
            print(f"[SKIP] {log_path} not found")
            continue

        print(f"\nParsing {log_path} …")
        records = parse_log(log_path)
        print(f"  → {len(records)} records, "
              f"{len({r['layer'] for r in records})} layers, "
              f"max cycle {max(r['cycle'] for r in records):,}")

        # Per-metric timing diagrams + heatmaps
        for metric in ("mem_idle_pct", "core_idle_pct", "sa_util_pct", "bubble_pct"):
            plot_timing(records, model_name, metric=metric,
                        out_path=os.path.join(args.out_dir, f"{model_name}_{metric}.png"))
            plot_heatmap(records, model_name, metric=metric,
                         out_path=os.path.join(args.out_dir, f"{model_name}_{metric}_heatmap.png"))

        # All-in-one overview
        plot_overview(records, model_name,
                      out_path=os.path.join(args.out_dir, f"{model_name}_overview.png"))


if __name__ == "__main__":
    main()