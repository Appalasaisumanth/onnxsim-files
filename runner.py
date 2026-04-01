import json
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Import your parser
import parser  # make sure parser.py is in same folder

# Paths
SIMULATOR = "./build/bin/Simulator"
CONFIG = "./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json"
MODELS_JSON = "example/models_list.json"
LOG_DIR = "logs_new_mamba_without_l2"

# Parallelism (IMPORTANT: tune based on CPU/memory)
MAX_WORKERS = 4

os.makedirs(LOG_DIR, exist_ok=True)


def run_single_model(model):
    name = model["name"]

    temp_file = f"temp_{name}.json"
    log_file = os.path.join(LOG_DIR, f"{name}.log")

    try:
        print(f"🚀 Running: {name}")

        # Create temp JSON
        with open(temp_file, "w") as f:
            json.dump({"models": [model]}, f, indent=2)

        cmd = [
            SIMULATOR,
            "--config", CONFIG,
            "--models_list", temp_file
        ]

        # Run simulator
        with open(log_file, "w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            print(f"❌ Failed: {name}")
            return None

        print(f"✅ Finished: {name}")

        # ─────────────────────────────
        # PARSE IMMEDIATELY
        # ─────────────────────────────
        data = parser.parse_log(log_file)

        base = log_file.replace(".log", "")

        parsed_path = base + "_parsed.json"
        summary_path = base + "_summary.json"

        # Write parsed JSON
        with open(parsed_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Build summary
        summary = parser._build_summary_json(data)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # return {
        #     "parsed": parser.flatten_dict(data) | {"file": log_file},
        #     "summary": parser.flatten_dict(summary) | {"file": log_file}
        # }

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    with open(MODELS_JSON, "r") as f:
        models = json.load(f)["models"]

    all_parsed_rows = []
    all_summary_rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_single_model, m) for m in models]

        for future in as_completed(futures):
            result = future.result()
            if result:
                all_parsed_rows.append(result["parsed"])
                all_summary_rows.append(result["summary"])

    # ─────────────────────────────
    # WRITE CSVs
    # ─────────────────────────────
    # parser.write_csv(all_parsed_rows, "all_parsed.csv")
    # parser.write_csv(all_summary_rows, "all_summary.csv")

    # print("\n🎉 DONE")
    # print("Generated:")
    # print(" - all_parsed.csv")
    # print(" - all_summary.csv")


if __name__ == "__main__":
    main()