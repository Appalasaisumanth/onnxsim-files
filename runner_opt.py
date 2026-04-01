import os
import subprocess
import json

# ===== CONFIG =====
SIMULATOR = "./build/bin/Simulator"
CONFIG = "./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json"

OUTPUT_DIR = "opt_logs_without_l2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LENGTHS = [128, 256, 512, 1024,2048]
MODELS = ["opt-125m"]

# ==================

for model in MODELS:
    for seq in SEQ_LENGTHS:

        # ---- Create CSV (generation-only) ----
        csv_path = os.path.join(OUTPUT_DIR, f"{model}_{seq}.csv")
        with open(csv_path, "w") as f:
            f.write("time,prompt_length,target_length,cached_length\n")
            f.write(f"0,1,{seq},0\n")

        # ---- Create model config ----
        json_path = os.path.join(OUTPUT_DIR, f"{model}_{seq}.json")
        model_json = {
            "models": [
                {
                    "name": model,
                    "trace_file": csv_path,
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
        log_path = os.path.join(OUTPUT_DIR, f"{model}_{seq}.log")

        print(f"Running {model} seq={seq}")

        with open(log_path, "w") as log_file:
            subprocess.run(
                [
                    SIMULATOR,
                    "--config", CONFIG,
                    "--models_list", json_path,
                    "--mode", "language"
                ],
                stdout=log_file,
                stderr=log_file
            )

print("All logs generated.")