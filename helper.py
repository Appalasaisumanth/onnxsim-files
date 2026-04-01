import os
import json
import pandas as pd
import csv
def write_csv(rows, output_file):
    if not rows:
        return

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    fieldnames = ["file"] + sorted(k for k in all_keys if k != "file")

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def flatten_dict(d, parent_key="", sep="."):
    items = {}

    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_dict(v, new_key, sep))

    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_dict(v, new_key, sep))

    else:
        items[parent_key] = d

    return items
def reduce_heavy_sections(summary):
    s = dict(summary)

    # ── Handle per_core ─────────────────────────────
    if "per_core" in s:
        avg = s["per_core"].get("avg", {})
        
        # Replace full per_core with only avg
        s["per_core"] = {
            "avg": avg
        }

    # ── Handle L2 cache ─────────────────────────────
    if "l2_cache" in s and s["l2_cache"]:
        l2 = s["l2_cache"]

        # Remove per-bank completely
        l2.pop("per_bank", None)

        # Keep only aggregate fields (already there)
        s["l2_cache"] = l2

    return s

base_dir = "./"
files = [f for f in os.listdir(base_dir) if f.endswith("_summary.json")]

rows = []

for file in files:
    with open(os.path.join(base_dir, file)) as f:
        data = json.load(f)

    # -------------------------
    # REMOVE unwanted sections
    # -------------------------
    data.pop("per_core", None)

    if "l2_cache" in data and data["l2_cache"]:
        data["l2_cache"].pop("per_bank", None)

    # -------------------------
    # FLATTEN
    # -------------------------
    # Reduce heavy nested sections
    cleaned_summary = reduce_heavy_sections(data)

    # Flatten
    summary_flat = flatten_dict(cleaned_summary)

    # Add filename
    summary_flat["_name"] = file
    print(summary_flat["_name"])
    rows.append(summary_flat)
# print(rows)

# -------------------------
# CREATE DATAFRAME
# -------------------------
df = pd.DataFrame(rows)

# -------------------------
# SAVE
# -------------------------
write_csv(rows,"flattened_summary.csv")

print("Done. Saved as flattened_summary.csv")