import re, os, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "font.size": 22,            # base font
    "axes.labelsize": 24,       # X and Y labels
    "axes.titlesize": 30,       # subplot titles (if any)
    "xtick.labelsize": 24,      # X tick labels
    "ytick.labelsize": 24,      # Y tick labels
    "legend.fontsize": 22,      # legend text
    "legend.title_fontsize": 24,# legend title
    "figure.titlesize": 34      # suptitle (main title)
})
DEFAULT_LOGS = {
    "mamba_130m": "mamba_130m.log",
    "tiny_opt":   "tiny_opt_gen.log",
    "mamba_2.8b": "mamba_2.8b.log",
    "llama":      "llama_gen.log",
}

LAYER_COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B3",
    "#CCB974", "#64B5CD", "#8C8C8C", "#E17C05",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
]

METRIC_LABELS = {
    "mem_idle_pct":  "Dispatch Stall (%)",       # was "Memory Utilization (%)" — wildly wrong name
    "sa_util_pct":   "Systolic Array Util (%)",  # was "PE Utilization (%)" — fine but more precise

    # "core_idle_pct": "Core Idle (%)",
    "vec_util_pct":  "Vector Utilization (%)",
    # "bubble_pct":    "Systolic Bubble (%)",
}

RE_LAYER  = re.compile(r"Start layer (.+)")
RE_FINISH = re.compile(r"Layer (.+) finish at (\d+)")
RE_SCHED  = re.compile(r"Schedule model")          # model restart boundary
RE_IDLE   = re.compile(
    r"Core \[(\d+)\] : Memory unit idle cycle (\d+)"
    r" Systolic bubble cycle (\d+) Core idle cycle (\d+)"
)
RE_UTIL   = re.compile(
    r"Core \[(\d+)\].*?Systolic Array Utilization\(%\)\s*([\d.]+)"
    r".*?Vector Unit Utilization\(%\)\s*([\d.]+)"
    r".*?Total cycle:\s*(\d+)"
)
RE_CYCLE  = re.compile(r"cycle: \[(\d+)\]")


# ── Layer Span Builder ─────────────────────────────────────────────────
def get_layer_spans(records):
    spans = []
    prev_layer = None
    start_cycle = None
    records = sorted(records, key=lambda x: x["cycle"])
    for r in records:
        lkey = r["layer_key"]
        if lkey != prev_layer:
            if prev_layer is not None:
                spans.append((start_cycle, r["cycle"], prev_layer))
            start_cycle = r["cycle"]
            prev_layer = lkey
    if prev_layer is not None:
        spans.append((start_cycle, records[-1]["cycle"], prev_layer))
    return spans


# ── Parser (pass-aware) ────────────────────────────────────────────────
def parse_log(path):
    records = []
    prev, staged = {}, {}
    current_layer = "unknown"
    last_core = None

    # pass tracking: every "Schedule model" line = new pass
    pass_idx = 0
    # track which base-layer names we've seen in this pass
    seen_in_pass = set()

    def reset_stage():
        nonlocal prev, staged
        prev, staged = {}, {}

    with open(path) as f:
        for line in f:

            # ── model restart → new pass ───────────────────────────────
            if RE_SCHED.search(line):
                pass_idx += 1
                seen_in_pass = set()
                reset_stage()
                continue

            # ── new layer ─────────────────────────────────────────────
            m = RE_LAYER.search(line)
            if m:
                base = m.group(1).strip()
                current_layer = base
                # build unique key: "pass{N}.{layer_name}"
                reset_stage()
                continue

            m = RE_IDLE.search(line)
            if m:
                core = int(m.group(1))
                last_core = core
                staged.setdefault(core, {}).update({
                    "mem":    int(m.group(2)),
                    "bubble": int(m.group(3)),
                    "ci":     int(m.group(4)),
                    "layer":  current_layer,
                    "pass":   pass_idx,
                })
                continue

            m = RE_UTIL.search(line)
            if m:
                core = int(m.group(1))
                last_core = core
                staged.setdefault(core, {}).update({
                    "sa":    float(m.group(2)),
                    "vec":   float(m.group(3)),
                    "cycle": int(m.group(4)),
                })
                continue

            m = RE_CYCLE.search(line)
            if m:
                abs_cycle = int(m.group(1))
                if last_core not in staged:
                    continue

                core = last_core
                s = staged[core]

                interval = abs_cycle - prev.get(core, {}).get("cycle", 0)
                if interval <= 0:
                    continue

                # unique label for legend / color mapping
                layer_key = f"p{s.get('pass', 0)}.{s.get('layer', 'unknown')}"

                records.append({
                    "core":          core,
                    "cycle":         abs_cycle,
                    "layer":         s.get("layer", "unknown"),
                    "pass":          s.get("pass", 0),
                    "layer_key":     layer_key,
                    "mem_idle_pct":  min(s.get("mem",    0) / interval * 100, 100),
                    "core_idle_pct": min(s.get("ci",     0) / interval * 100, 100),
                    "bubble_pct":    min(s.get("bubble", 0) / interval * 100, 100),
                    "sa_util_pct":   min(s.get("sa",     0), 100),
                    "vec_util_pct":  min(s.get("vec",    0), 100),
                })

                prev[core] = {"cycle": abs_cycle}
                staged.pop(core, None)

    return records


# ── Common Plot Finalizer ──────────────────────────────────────────────
def finalize_plot(ax, spans, layer_color_map, out_path):
    from matplotlib.patches import Patch

    ax.set_autoscale_on(False)
    ax.set_ylim(-5, 105)
    ax.set_yticks(np.linspace(0, 100, 11))
    ax.margins(y=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    core_legend = ax.legend(loc="upper left")
    ax.add_artist(core_legend)

    layer_patches = [
        Patch(facecolor=color, alpha=0.3, label=lkey)
        for lkey, color in layer_color_map.items()
    ]
    layer_legend = ax.legend(
        handles=layer_patches,
        title="Pass.Layer",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=12,
        title_fontsize=16,
        ncol=2,
        frameon=False,
    )
    ax.add_artist(layer_legend)
    ax.grid(alpha=0.3)
    plt.subplots_adjust(top=0.95, bottom=0.12, right=0.72)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


# ── Generic Metric Plot ────────────────────────────────────────────────
def plot_metric(records, model, metric, out_path):
    cores = sorted(set(r["core"] for r in records))
    spans = get_layer_spans(records)

    # assign colors per unique layer_key
    unique_keys = list(dict.fromkeys(s[2] for s in spans))
    layer_color_map = {k: LAYER_COLORS[i % len(LAYER_COLORS)]
                       for i, k in enumerate(unique_keys)}

    fig, ax = plt.subplots(figsize=(24, 6))

    for start, end, lkey in spans:
        color = layer_color_map[lkey]
        ax.axvspan(start, end, color=color, alpha=0.12)
        ax.axvline(start, color=color, linestyle=":", linewidth=0.8)

    for core in cores:
        data = sorted([r for r in records if r["core"] == core],
                      key=lambda x: x["cycle"])
        xs = [r["cycle"] for r in data]
        ys = [r[metric]  for r in data]
        ax.plot(xs, ys, label=f"Core {core}")

    ax.set_title(f"{model} — {METRIC_LABELS[metric]}  (all passes)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel(METRIC_LABELS[metric])
    finalize_plot(ax, spans, layer_color_map, out_path)


# ── All-metrics combined plot ──────────────────────────────────────────
def plot_all_metrics(records, model, out_path):
    """Single figure with one subplot per metric, all passes, all layers."""
    metrics = list(METRIC_LABELS.keys())
    spans   = get_layer_spans(records)
    unique_keys    = list(dict.fromkeys(s[2] for s in spans))
    layer_color_map = {k: LAYER_COLORS[i % len(LAYER_COLORS)]
                       for i, k in enumerate(unique_keys)}

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(28, 5 * len(metrics)),
                             sharex=True)
    fig.suptitle(f"{model} — all metrics, all passes", fontsize=14, y=1.001)

    cores = sorted(set(r["core"] for r in records))

    for ax, metric in zip(axes, metrics):
        for start, end, lkey in spans:
            color = layer_color_map[lkey]
            ax.axvspan(start, end, color=color, alpha=0.10)
            ax.axvline(start, color=color, linestyle=":", linewidth=0.7)

        for core in cores:
            data = sorted([r for r in records if r["core"] == core],
                          key=lambda x: x["cycle"])
            xs = [r["cycle"]  for r in data]
            ys = [r[metric]   for r in data]
            ax.plot(xs, ys, label=f"Core {core}", linewidth=1.2)

        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_ylim(-5, 105)
        ax.set_yticks(np.linspace(0, 100, 6))
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc="upper left", fontsize=12)

    axes[-1].set_xlabel("Cycle")

    # shared layer legend on the right
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=c, alpha=0.4, label=k)
               for k, c in layer_color_map.items()]
    fig.legend(handles=patches,
               title="Pass.Layer",
               bbox_to_anchor=(1.01, 0.98),
               loc="upper left",
               fontsize=12,
               title_fontsize=16,
               ncol=1,
               frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


# ── Pass-level timeline (Gantt) ────────────────────────────────────────
def plot_gantt(records, model, out_path):
    """Horizontal bar chart: each row = one pass.layer, x = cycle range."""
    spans = get_layer_spans(records)
    unique_keys = list(dict.fromkeys(s[2] for s in spans))
    layer_color_map = {k: LAYER_COLORS[i % len(LAYER_COLORS)]
                       for i, k in enumerate(unique_keys)}

    fig, ax = plt.subplots(figsize=(24, max(4, len(unique_keys) * 0.5)))

    yticks, ylabels = [], []
    for i, lkey in enumerate(unique_keys):
        yticks.append(i)
        ylabels.append(lkey)
        my_spans = [(s, e) for s, e, k in spans if k == lkey]
        for s, e in my_spans:
            ax.barh(i, e - s, left=s, height=0.6,
                    color=layer_color_map[lkey], alpha=0.8)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Cycle")
    ax.set_title(f"{model} — layer timeline (all passes)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


# ── Compute vs Memory ──────────────────────────────────────────────────
def plot_compute_vs_memory(records, model, out_path):
    cores = sorted(set(r["core"] for r in records))
    spans = get_layer_spans(records)
    unique_keys = list(dict.fromkeys(s[2] for s in spans))
    layer_color_map = {k: LAYER_COLORS[i % len(LAYER_COLORS)]
                       for i, k in enumerate(unique_keys)}

    fig, ax = plt.subplots(figsize=(24, 6))
    for start, end, lkey in spans:
        ax.axvspan(start, end, color=layer_color_map[lkey], alpha=0.10)

    for core in cores:
        data = sorted([r for r in records if r["core"] == core],
                      key=lambda x: x["cycle"])
        xs      = [r["cycle"] for r in data]
        sa_vec  = [r["sa_util_pct"] + r["vec_util_pct"] for r in data]
        mem_idle = [r["mem_idle_pct"] for r in data]
        ax.plot(xs, sa_vec,   label=f"C{core} Compute")
        ax.plot(xs, mem_idle, label=f"C{core} MemIdle", linestyle="--")

    ax.set_title(f"{model} — Compute vs Memory (all passes)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Utilization (%)")
    finalize_plot(ax, spans, layer_color_map, out_path)


# ── Single metric, single pass ────────────────────────────────────────
def plot_single_metric_pass(records, model, metric, phase_label, phase_title, out_path):
    """One plot = one metric for one phase (prompt or token_gen).
    X-axis rebased to 0. Layer bands annotated with base layer name."""
    from matplotlib.patches import Patch

    base = min(r["cycle"] for r in records)
    records = [{**r, "cycle": r["cycle"] - base} for r in records]

    cores = sorted(set(r["core"] for r in records))
    spans = get_layer_spans(records)

    def strip_pass(lkey):
        parts = lkey.split(".", 1)
        return parts[1] if len(parts) == 2 else lkey

    unique_base = list(dict.fromkeys(strip_pass(s[2]) for s in spans))
    base_cmap   = {k: LAYER_COLORS[i % len(LAYER_COLORS)]
                   for i, k in enumerate(unique_base)}

    fig, ax = plt.subplots(figsize=(20, 5))

    for start, end, lkey in spans:
        bkey  = strip_pass(lkey)
        color = base_cmap[bkey]
        ax.axvspan(start, end, color=color, alpha=0.13)
        ax.axvline(start, color=color, linestyle=":", linewidth=0.8)
        mid = (start + end) / 2
        ax.text(mid, 102, bkey, fontsize=6, ha="center", va="bottom",
                color=color, rotation=45, clip_on=True)

    for core in cores:
        data = sorted([r for r in records if r["core"] == core],
                      key=lambda x: x["cycle"])
        xs = [r["cycle"] for r in data]
        ys = [r[metric]  for r in data]
        ax.plot(xs, ys, label=f"Core {core}", linewidth=1.4)

    ax.set_title(f"{model}  |  {phase_title}  -  {METRIC_LABELS[metric]}",
                 fontsize=12)
    ax.set_xlabel("Cycle (rebased to 0)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(-5, 115)
    ax.set_yticks(np.linspace(0, 100, 11))
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    core_legend = ax.legend(loc="upper left", fontsize=12)
    ax.add_artist(core_legend)

    layer_patches = [Patch(facecolor=c, alpha=0.35, label=k)
                     for k, c in base_cmap.items()]
    ax.legend(handles=layer_patches, title="Layer", fontsize=12,
              title_fontsize=16, bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False, ncol=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

def plot_all_metrics_single_pass(records, model, pass_id, out_path):
    """
    One figure per pass, all metrics (subplots).
    X-axis rebased to 0 for clean comparison.
    """
    metrics = list(METRIC_LABELS.keys())

    # filter pass
    precs = [r for r in records if r["pass"] == pass_id]
    if not precs:
        print(f"[SKIP] No data for pass {pass_id}")
        return

    # rebase cycles
    base = min(r["cycle"] for r in precs)
    precs = [{**r, "cycle": r["cycle"] - base} for r in precs]

    spans = get_layer_spans(precs)

    # strip pass from label (only layer name)
    def strip_pass(lkey):
        return lkey.split(".", 1)[-1]

    unique_layers = list(dict.fromkeys(strip_pass(s[2]) for s in spans))
    layer_color_map = {
        l: LAYER_COLORS[i % len(LAYER_COLORS)]
        for i, l in enumerate(unique_layers)
    }

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(26, 5 * len(metrics)),
                             sharex=True)

    PHASE_NAMES = {0: "Prompt Phase", 1: "Token Generation Phase"}
    title = PHASE_NAMES.get(pass_id, f"Pass {pass_id}")
    fig.suptitle(f"{model} — {title} (All Metrics)", fontsize=14)

    cores = sorted(set(r["core"] for r in precs))

    for ax, metric in zip(axes, metrics):

        # layer shading
        for start, end, lkey in spans:
            lname = strip_pass(lkey)
            color = layer_color_map[lname]
            ax.axvspan(start, end, color=color, alpha=0.12)
            ax.axvline(start, color=color, linestyle=":", linewidth=0.7)

        # plot per core
        for core in cores:
            data = sorted([r for r in precs if r["core"] == core],
                          key=lambda x: x["cycle"])
            xs = [r["cycle"] for r in data]
            ys = [r[metric] for r in data]
            ax.plot(xs, ys, label=f"Core {core}", linewidth=1.2)
        # custom scaling per metric
        # if metric == "vec_util_pct":
        #     ax.set_ylim(0, 5)
        #     ax.set_yticks(np.linspace(0, 5, 6))
        # else:
        #     ax.set_ylim(0, 100)
        #     ax.set_yticks(np.linspace(0, 100, 6))

        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(-5, 105)
        ax.set_yticks(np.linspace(0, 100, 6))
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc="upper left", fontsize=24)

    axes[-1].set_xlabel("Cycle (rebased to 0)")

    # shared layer legend
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=c, alpha=0.4, label=l)
               for l, c in layer_color_map.items()]

    fig.legend(handles=patches,
               title="Layer",
               bbox_to_anchor=(0.96, 0.98),
               loc="upper left",
               fontsize=24,
               frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)
# ── Pass Comparison Plot ───────────────────────────────────────────────
def plot_pass_comparison(records, model, pa, pb, out_path):
    """
    Two-column figure: left = pass A, right = pass B.
    One row per metric. X-axis is re-zeroed per pass so cycles are
    directly comparable regardless of absolute start time.
    """
    metrics = list(METRIC_LABELS.keys())
    recs_a  = [r for r in records if r["pass"] == pa]
    recs_b  = [r for r in records if r["pass"] == pb]

    if not recs_a or not recs_b:
        print(f"[SKIP comparison] pass {pa} or {pb} has no data")
        return

    def rebase(recs):
        base = min(r["cycle"] for r in recs)
        return [{**r, "cycle": r["cycle"] - base} for r in recs]

    recs_a = rebase(recs_a)
    recs_b = rebase(recs_b)

    def build_spans_and_colors(recs):
        spans = get_layer_spans(recs)
        unique_keys = list(dict.fromkeys(s[2] for s in spans))
        cmap = {k: LAYER_COLORS[i % len(LAYER_COLORS)]
                for i, k in enumerate(unique_keys)}
        return spans, cmap

    spans_a, cmap_a = build_spans_and_colors(recs_a)
    spans_b, cmap_b = build_spans_and_colors(recs_b)

    # unify color map so same base layer name gets same color in both columns
    all_keys = list(dict.fromkeys(
        list(cmap_a.keys()) + list(cmap_b.keys())
    ))
    unified_cmap = {k: LAYER_COLORS[i % len(LAYER_COLORS)]
                    for i, k in enumerate(all_keys)}
    cmap_a = unified_cmap
    cmap_b = unified_cmap

    PHASE_NAMES = {0: "Prompt Phase", 1: "Token Generation Phase"}
    label_a = PHASE_NAMES.get(pa, f"Pass {pa}")
    label_b = PHASE_NAMES.get(pb, f"Pass {pb}")

    fig, axes = plt.subplots(
        len(metrics), 2,
        figsize=(32, 4 * len(metrics)),
        sharex="col"
    )
    fig.suptitle(
        f"{model}  |  {label_a}  vs  {label_b}",
        fontsize=14, y=1.002
    )

    cores = sorted(set(r["core"] for r in records))

    for row, metric in enumerate(metrics):
        for col, (recs, spans, cmap, pid) in enumerate([
            (recs_a, spans_a, cmap_a, pa),
            (recs_b, spans_b, cmap_b, pb),
        ]):
            ax = axes[row][col]

            for start, end, lkey in spans:
                color = cmap.get(lkey, "#aaaaaa")
                ax.axvspan(start, end, color=color, alpha=0.12)
                ax.axvline(start, color=color, linestyle=":", linewidth=0.7)

            for core in cores:
                data = sorted([r for r in recs if r["core"] == core],
                              key=lambda x: x["cycle"])
                if not data:
                    continue
                xs = [r["cycle"] for r in data]
                ys = [r[metric]  for r in data]
                ax.plot(xs, ys, label=f"Core {core}", linewidth=1.2)

            ax.set_ylim(-5, 105)
            ax.set_yticks(np.linspace(0, 100, 6))
            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                col_title = label_a if col == 0 else label_b
                ax.set_title(col_title, fontsize=11, pad=6)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            if row == len(metrics) - 1:
                ax.set_xlabel("Cycle (rebased to 0)", fontsize=9)
            if col == 0:
                ax.legend(loc="upper left", fontsize=7)

    # shared layer legend
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=c, alpha=0.4, label=k)
               for k, c in unified_cmap.items()]
    fig.legend(
        handles=patches,
        title="Pass.Layer",
        bbox_to_anchor=(1.01, 0.98),
        loc="upper left",
        fontsize=7,
        title_fontsize=9,
        ncol=1,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for model, fname in DEFAULT_LOGS.items():
        path = os.path.join(args.log_dir, fname)
        if not os.path.exists(path):
            print("[SKIP]", path)
            continue

        print(f"\nParsing: {path}")
        records = parse_log(path)

        passes = sorted(set(r["pass"] for r in records))
        print(f"  Found {len(passes)} pass(es): {passes}")
        layers = sorted(set(r["layer_key"] for r in records))
        print(f"  Unique pass.layer keys: {len(layers)}")

        # per-metric plots (all passes together)
        for metric in METRIC_LABELS:
            plot_metric(records, model, metric,
                        os.path.join(args.out_dir, f"{model}_{metric}.png"))

        # combined all-metrics plot
        plot_all_metrics(records, model,
                         os.path.join(args.out_dir, f"{model}_ALL_metrics.png"))

        # gantt timeline
        plot_gantt(records, model,
                   os.path.join(args.out_dir, f"{model}_gantt.png"))

        # compute vs memory
        plot_compute_vs_memory(records, model,
                               os.path.join(args.out_dir,
                                            f"{model}_compute_vs_memory.png"))

        # ── per-pass: 5 individual metric plots each ──────────────────
        PHASE_NAMES = {0: "prompt", 1: "token_gen"}
        for p in passes:
            precs = [r for r in records if r["pass"] == p]
            if not precs:
                continue
            phase_label = PHASE_NAMES.get(p, f"pass{p}")
            phase_title = ("Prompt Phase" if p == 0 else
                           "Token Generation Phase" if p == 1 else
                           f"Pass {p}")
            for metric in METRIC_LABELS:
                plot_single_metric_pass(
                    precs, model, metric, phase_label, phase_title,
                    os.path.join(args.out_dir,
                                 f"{model}_{phase_label}_{metric}.png"))
            plot_all_metrics_single_pass(
        records,
        model,
        p,
        os.path.join(args.out_dir, f"{model}_pass{p}_ALL_metrics.png")
    )

        # ── side-by-side comparison (prompt vs token_gen) ─────────────
        for i in range(len(passes) - 1):
            pa, pb = passes[i], passes[i + 1]
            plot_pass_comparison(
                records, model, pa, pb,
                os.path.join(args.out_dir,
                             f"{model}_pass{pa}_vs_pass{pb}.png"))

if __name__ == "__main__":
    main()