import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("results_per_log.csv")

# -----------------------------
# Better extraction for both Mamba and OPT
# -----------------------------
def extract_seq(log):
    log = str(log)
    # Handle both sXXXX and _XXXX patterns
    m = re.search(r'[s_](\d+)', log)
    return int(m.group(1)) if m else None

def get_model(log):
    log = str(log).lower()
    if 'mamba' in log:
        return 'Mamba'
    elif 'opt' in log:
        return 'OPT'
    return 'Unknown'

df['seq_len'] = df['log_file'].apply(extract_seq)
df['model'] = df['log_file'].apply(get_model)

# Drop rows where seq_len extraction failed
df = df.dropna(subset=['seq_len']).copy()

print("Extracted sequence lengths:", sorted(df['seq_len'].unique()))
print("Models found:", df['model'].value_counts().to_dict())

# -----------------------------
# Numeric columns (exclude grouping keys)
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
group_cols = ['model', 'seq_len']

for col in group_cols:
    if col in numeric_cols:
        numeric_cols.remove(col)

print(f"Aggregating using {len(numeric_cols)} numeric columns.")

# Fill NaNs
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Aggregate
agg_df = df.groupby(group_cols)[numeric_cols].mean().reset_index()

# -----------------------------
# Derived metrics (do all at once to avoid fragmentation warning)
# -----------------------------
derived = pd.DataFrame({
    'memory_traffic_bytes': agg_df.get('dram_total_reads', 0) + agg_df.get('dram_total_writes', 0),
    'memory_bound_ratio': agg_df['timing_total_cycles'] / agg_df['timing_total_compute_cycles'].replace(0, np.nan),
    'cycles_per_token': agg_df['timing_total_cycles'] / agg_df['seq_len'],
    'tokens_per_cycle': agg_df['seq_len'] / agg_df['timing_total_cycles'],
    'wall_time_per_token_ms': (agg_df['timing_wall_clock_seconds'] / agg_df['seq_len']) * 1000
})

agg_df = pd.concat([agg_df, derived], axis=1)

# -----------------------------
# Speedup Calculation (Mamba vs OPT)
# -----------------------------
speedup_created = False
mamba_df = agg_df[agg_df['model'] == 'Mamba'].set_index('seq_len')
opt_df   = agg_df[agg_df['model'] == 'OPT'].set_index('seq_len')

common_seq = mamba_df.index.intersection(opt_df.index)

if not common_seq.empty:
    speedup = mamba_df.loc[common_seq, 'timing_total_cycles'] / opt_df.loc[common_seq, 'timing_total_cycles']
    agg_df.loc[agg_df['model'] == 'Mamba', 'speedup_vs_opt'] = speedup.reindex(
        agg_df[agg_df['model'] == 'Mamba']['seq_len']
    ).values
    speedup_created = True
    print("\n=== Mamba Speedup vs OPT (cycles ratio <1 = Mamba faster) ===")
    print(agg_df[agg_df['model']=='Mamba'][['seq_len', 'speedup_vs_opt']].round(3))
else:
    print("\n⚠️  No common sequence lengths between Mamba and OPT.")
    print("Mamba seq:", sorted(mamba_df.index.unique()))
    print("OPT seq:  ", sorted(opt_df.index.unique()))

# -----------------------------
# Plotting - Many metrics
# -----------------------------
metrics_to_plot = [
    'timing_total_cycles', 'timing_wall_clock_seconds', 'memory_traffic_bytes',
    'memory_bound_ratio', 'cycles_per_token', 'tokens_per_cycle',
    'dram_avg_bw_utilization_pct', 'compute_avg_pe_util_pct',
    'compute_avg_systolic_util_pct', 'core_avg_pe_util_pct',
    'sram_hit_rate', 'dram_avg_row_hit_rate_pct'
]

for metric in metrics_to_plot:
    if metric not in agg_df.columns:
        continue
    plt.figure(figsize=(11, 7))
    for model_name in ['Mamba', 'OPT']:
        subset = agg_df[agg_df['model'] == model_name]
        if not subset.empty:
            plt.plot(subset['seq_len'], subset[metric], marker='o', linewidth=2.5, label=model_name)
    
    plt.xlabel("Sequence Length")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"{metric.replace('_', ' ').title()} vs Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.7)
    if agg_df['seq_len'].max() > 1000:
        plt.xscale('log')
    plt.savefig(f"{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Speedup plot
if speedup_created:
    plt.figure(figsize=(10, 6))
    speedup_data = agg_df[agg_df['model']=='Mamba'].dropna(subset=['speedup_vs_opt'])
    plt.plot(speedup_data['seq_len'], speedup_data['speedup_vs_opt'], 
             marker='o', color='red', linewidth=2.5, label='Mamba / OPT')
    plt.axhline(y=1.0, color='gray', linestyle='--', label='Equal performance')
    plt.xlabel("Sequence Length")
    plt.ylabel("Speedup Ratio (Mamba cycles / OPT cycles)")
    plt.title("Mamba Speedup vs OPT")
    plt.legend()
    plt.grid(True)
    plt.savefig("speedup_vs_opt.png", dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# Save tables
# -----------------------------
pivot = agg_df.pivot(index='seq_len', columns='model', values='timing_total_cycles')
pivot.to_latex("cycles_table.tex", float_format="%.2e")

summary_cols = ['seq_len', 'model', 'timing_total_cycles', 'timing_wall_clock_seconds',
                'memory_traffic_bytes', 'memory_bound_ratio', 'tokens_per_cycle',
                'dram_avg_bw_utilization_pct', 'compute_avg_pe_util_pct']

if 'speedup_vs_opt' in agg_df.columns:
    summary_cols.append('speedup_vs_opt')

summary = agg_df[summary_cols].round(4)
summary.to_csv("detailed_summary.csv", index=False)
summary.to_latex("detailed_summary.tex", float_format="%.2e", index=False)

print("\n✅ All plots and tables generated successfully!")
print("Check the folder for these files:")
print("• scaling plots (*.png)")
print("• speedup_vs_opt.png (if common seq lengths exist)")
print("• detailed_summary.csv")
print("• cycles_table.tex")