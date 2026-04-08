from functools import cache
import os
import subprocess
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== CONFIG =====
SIMULATOR = "./build/bin/Simulator"
CONFIG = "./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json"
OUTPUT_DIR = "opt_logs_without_l2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LENGTHS = [(1809,8192),(3617,16384),(13617,16384)]
MODELS = ["tiny"]

MAX_WORKERS = 4
# ==================


def run_simulation(model, seq,cached):
    try:
        # ---- Create CSV ----
        csv_path = os.path.join("traces", f"{model}_{seq}.csv")
        with open(csv_path, "w") as f:
            f.write("time,prompt_length,target_length,cached_length\n")
            f.write(f"0,1,{seq},{cached}\n")

        # ---- Create JSON ----
        json_path = os.path.join(OUTPUT_DIR, f"{model}_{seq}.json")
        model_json = {
            "models": [
                {
                    "name": model,
                    "trace_file": csv_path[7:],
                    "scheduler": "simple",
                    "scheduler_config": {
                        "max_batch_size": 1
                    }
                }
            ]
        }

        with open(json_path, "w") as f:
            json.dump(model_json, f, indent=2)

        # ---- Run simulator ----
        log_path = os.path.join(OUTPUT_DIR, f"{model}_s{seq}.log")

        print(f"[START] {model} seq={seq}")

        with open(log_path, "w") as log_file:
            subprocess.run(
                [
                    SIMULATOR,
                    "--config", CONFIG,
                    "--models_list", json_path,
                    "--mode", "language",
                    "trace_file", csv_path[8:]
                ],
                stdout=log_file,
                stderr=log_file
            )

        print(f"[DONE]  {model} seq={seq}")
        return f"{model}_{seq} SUCCESS"

    except Exception as e:
        return f"{model}_{seq} FAILED: {str(e)}"


if __name__ == "__main__":
    tasks = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        for model in MODELS:
            for seq,cached in SEQ_LENGTHS:
                tasks.append(executor.submit(run_simulation, model, seq,cached))

        for future in as_completed(tasks):
            print(future.result())

    print("All logs generated.")