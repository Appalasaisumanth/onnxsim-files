import re
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

MAMBA_LOG = "mamba_130m.log"
OPT_LOG   = "tiny_opt.log"

CORE_COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E"]

# 36-colour palette for layers
LAYER_PALETTE = [
    "#534AB7","#0F6E56","#993C1D","#993556","#185FA5","#3B6D11","#854F0B","#A32D2D",
    "#3C3489","#085041","#712B13","#72243E","#0C447C","#27500A","#633806","#791F1F",
    "#7F77DD","#1D9E75","#D85A30","#D4537E","#378ADD","#639922","#BA7517","#E24B4A",
    "#AFA9EC","#5DCAA5","#F0997B","#ED93B1","#85B7EB","#97C459","#EF9F27","#F09595",
    "#EEEDFE","#E1F5EE","#FAECE7","#FBEAF0",
]

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_log(path):
    """Return list of dicts with keys: core, cycle, layer, mem_idle_pct, core_idle_pct"""
    records = []
    current_layer = "unknown"
    layer_re  = re.compile(r"Start layer (.+)")
    idle_re   = re.compile(
        r"Core \[(\d+)\] : Memory unit idle cycle (\d+) "
        r"Systolic bubble cycle (\d+) Core idle cycle (\d+)"
    )
    cycle_re  = re.compile(r"Core \[(\d+)\].*Total cycle: (\d+)")
    pending   = {}

    with open(path) as fh:
        for line in fh:
            m = layer_re.search(line)
            if m:
                current_layer = m.group(1).strip()
                continue

            m = idle_re.search(line)
            if m:
                core = int(m.group(1))
                pending[core] = {
                    "mem_idle":  int(m.group(2)),
                    "core_idle": int(m.group(4)),
                    "layer":     current_layer,
                }
                continue

            m = cycle_re.search(line)
            if m:
                core  = int(m.group(1))
                cycle = int(m.group(2))
                if core in pending and "cycle" not in pending[core]:
                    p = pending[core]
                    p["cycle"] = cycle
                    tc = cycle or 1
                    records.append({
                        "core":         core,
                        "cycle":        cycle,
                        "layer":        p["layer"],
                        "mem_idle_pct": round(p["mem_idle"]  / tc * 100, 2),
                        "core_idle_pct":round(p["core_idle"] / tc * 100, 2),
                    })
                    del pending[core]

    return records


# ── Plot one model ────────────────────────────────────────────────────────────

def plot_model(records, model_name, metric="mem_idle_pct", out_path=None):
    """
    metric: 'mem_idle_pct' or 'core_idle_pct'
    """
    if not records:
        print(f"[WARN] No records for {model_name}")
        return

    # Unique layers in order of first appearance
    seen, layers = set(), []
    for r in records:
        if r["layer"] not in seen:
            seen.add(r["layer"])
            layers.append(r["layer"])

    layer_color = {l: LAYER_PALETTE[i % len(LAYER_PALETTE)] for i, l in enumerate(layers)}

    # Organise per core
    cores = sorted(set(r["core"] for r in records))
    by_core = {c: {} for c in cores}          # cycle -> {val, layer}
    for r in records:
        by_core[r["core"]][r["cycle"]] = {"val": r[metric], "layer": r["layer"]}

    all_cycles = sorted(set(r["cycle"] for r in records))

    # ── Layer background bands (based on core 0) ──────────────────────────────
    c0 = by_core.get(0, by_core[cores[0]])
    bands = []                                 # list of (start, end, layer)
    prev_layer, band_start = None, None
    for cy in all_cycles:
        lay = c0.get(cy, {}).get("layer", prev_layer)
        if lay != prev_layer:
            if prev_layer and band_start is not None:
                bands.append((band_start, cy, prev_layer))
            band_start = cy
            prev_layer = lay
    if prev_layer and band_start is not None:
        bands.append((band_start, all_cycles[-1], prev_layer))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        len(cores), 1,
        figsize=(18, 3 * len(cores)),
        sharex=True,
        gridspec_kw={"hspace": 0.08}
    )
    if len(cores) == 1:
        axes = [axes]

    metric_label = "Memory idle %" if metric == "mem_idle_pct" else "Core idle %"

    for ax, core in zip(axes, cores):
        # Layer bands
        for (bstart, bend, blayer) in bands:
            col = layer_color[blayer]
            ax.axvspan(bstart, bend, alpha=0.10, color=col, linewidth=0)

        # Layer boundary lines + labels on top subplot only
        drawn = set()
        for (bstart, _bend, blayer) in bands:
            ax.axvline(bstart, color="0.75", linewidth=0.4, zorder=1)
            if core == cores[0] and blayer not in drawn:
                ax.text(
                    bstart, 102, blayer,
                    fontsize=6, rotation=45, va="bottom", ha="left",
                    color=layer_color[blayer], clip_on=True
                )
                drawn.add(blayer)

        # Core line
        cdata = by_core[core]
        xs = sorted(cdata.keys())
        ys = [cdata[cy]["val"] for cy in xs]
        ax.plot(xs, ys, color=CORE_COLORS[core], linewidth=1.2,
                label=f"Core {core}", zorder=3)

        # Shade under the line
        ax.fill_between(xs, ys, alpha=0.15, color=CORE_COLORS[core], zorder=2)

        ax.set_ylabel(f"Core {core}\n{metric_label}", fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_tick_params(labelsize=7)
        ax.grid(axis="y", color="0.88", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # X-axis on last subplot
    axes[-1].set_xlabel("Simulation cycle", fontsize=9)
    axes[-1].xaxis.set_tick_params(labelsize=8)
    axes[-1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v)))
    )

    # Legend for layers
    patches = [
        mpatches.Patch(color=layer_color[l], alpha=0.6, label=l)
        for l in layers
    ]
    fig.legend(
        handles=patches,
        title="Layers",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=6,
        title_fontsize=7,
        ncol=max(1, len(layers) // 12 + 1),
        framealpha=0.9,
    )

    fig.suptitle(
        f"{model_name}  ·  {metric_label}  by core over time",
        fontsize=12, y=1.01
    )

    if out_path is None:
        out_path = f"{model_name}_{metric}.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Summary heatmap (all cores, all layers, mean idle) ───────────────────────

def plot_heatmap(records, model_name, metric="mem_idle_pct", out_path=None):
    if not records:
        return

    seen, layers = set(), []
    for r in records:
        if r["layer"] not in seen:
            seen.add(r["layer"])
            layers.append(r["layer"])

    cores = sorted(set(r["core"] for r in records))

    # mean idle per (core, layer)
    from collections import defaultdict
    sums   = defaultdict(float)
    counts = defaultdict(int)
    for r in records:
        key = (r["core"], r["layer"])
        sums[key]   += r[metric]
        counts[key] += 1

    data = np.array([
        [sums[(c, l)] / counts[(c, l)] if counts[(c, l)] else 0
         for l in layers]
        for c in cores
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.55), 3))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_yticks(range(len(cores)))
    ax.set_yticklabels([f"Core {c}" for c in cores], fontsize=9)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
    ax.set_title(
        f"{model_name}  ·  Mean {('memory' if metric=='mem_idle_pct' else 'core')} idle % per layer",
        fontsize=11
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Mean idle %", fontsize=8)

    # Annotate cells
    for r_i in range(len(cores)):
        for c_i in range(len(layers)):
            ax.text(c_i, r_i, f"{data[r_i, c_i]:.0f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if data[r_i, c_i] < 60 else "white")

    plt.tight_layout()
    if out_path is None:
        out_path = f"{model_name}_{metric}_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logs = {
        "mamba_130m": MAMBA_LOG,
        "tiny_opt":   OPT_LOG,
    }

    for model_name, log_path in logs.items():
        if not os.path.exists(log_path):
            print(f"[SKIP] {log_path} not found")
            continue

        print(f"\nParsing {log_path} …")
        records = parse_log(log_path)
        print(f"  {len(records)} records, "
              f"{len(set(r['layer'] for r in records))} layers, "
              f"max cycle {max(r['cycle'] for r in records):,}")

        for metric in ("mem_idle_pct", "core_idle_pct"):
            plot_model(records, model_name, metric=metric)
            plot_heatmap(records, model_name, metric=metric)


if __name__ == "__main__":
    main()