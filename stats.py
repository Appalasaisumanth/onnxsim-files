import re
import csv
import sys
from pathlib import Path
from collections import defaultdict

# ============================================================
# Regex Patterns
# ============================================================

MODEL_START_RE = re.compile(r"Register Language Model:(\S+)")
CYCLE_RE = re.compile(r"completed in (\d+) cycles")

CORE_TILE_RE = re.compile(
    r"Core \[(\d+)\] Tile Stats \| Issued: (\d+) Running: (\d+) Finished: (\d+)"
)

ROW_STATS_RE = re.compile(
    r"Row hits: (\d+), Row misses: (\d+), Row conflicts: (\d+)"
)

MEMSYS_OTHER_RE  = re.compile(r"total_num_other_requests:\s*(\d+)")
MEMSYS_WRITE_RE  = re.compile(r"total_num_write_requests:\s*(\d+)")
MEMSYS_READ_RE   = re.compile(r"total_num_read_requests:\s*(\d+)")
MEMSYS_CYCLES_RE = re.compile(r"memory_system_cycles:\s*(\d+)")

DRAM_CH_RE = re.compile(
    r"HBM2-CH_(\d+): .*?\((\d+) reads, (\d+) writes\)"
)

# ---------------- L2 ----------------

L2_BANK_RE = re.compile(r"===== L2 BANK (\d+) =====")
L2_HIT_RE  = re.compile(r"Hits:(\d+) Misses:(\d+)")
L2_RW_RE   = re.compile(r"ReadHits:(\d+) WriteHits:(\d+)")
L2_EVICT_RE = re.compile(r"Evictions:(\d+) Writebacks:(\d+)")

# ============================================================
# Parser
# ============================================================

def parse_log(file_path):

    data = {
        "file": Path(file_path).name,
        "model": "unknown",
        "cycles": 0,
        "cores": defaultdict(int),
        "dram": defaultdict(int),
        "l2": defaultdict(int),
        "dram_channels": 0,
        "l2_banks": 0
    }

    current_bank = None

    with open(file_path, "r", errors="ignore") as f:
        for line in f:

            # ---------------- Model ----------------
            m = MODEL_START_RE.search(line)
            if m:
                data["model"] = m.group(1)

            # ---------------- Cycles (take max) ----------------
            m = CYCLE_RE.search(line)
            if m:
                cyc = int(m.group(1))
                data["cycles"] = max(data["cycles"], cyc)

            # ---------------- Core ----------------
            m = CORE_TILE_RE.search(line)
            if m:
                cid, issued, running, finished = m.groups()
                data["cores"][cid] += int(issued)

            # ---------------- DRAM Row ----------------
            m = ROW_STATS_RE.search(line)
            if m:
                hits, misses, conflicts = map(int, m.groups())
                data["dram"]["row_hits"] += hits
                data["dram"]["row_misses"] += misses
                data["dram"]["row_conflicts"] += conflicts

            # ---------------- MemSys ----------------
            m = MEMSYS_OTHER_RE.search(line)
            if m:
                data["dram"]["other"] += int(m.group(1))

            m = MEMSYS_WRITE_RE.search(line)
            if m:
                data["dram"]["writes_req"] += int(m.group(1))

            m = MEMSYS_READ_RE.search(line)
            if m:
                data["dram"]["reads_req"] += int(m.group(1))

            m = MEMSYS_CYCLES_RE.search(line)
            if m:
                data["dram"]["mem_cycles"] += int(m.group(1))

            # ---------------- Channel ----------------
            m = DRAM_CH_RE.search(line)
            if m:
                _, reads, writes = m.groups()
                data["dram"]["reads"] += int(reads)
                data["dram"]["writes"] += int(writes)
                data["dram_channels"] += 1

            # ---------------- L2 ----------------
            m = L2_BANK_RE.search(line)
            if m:
                current_bank = int(m.group(1))
                data["l2_banks"] += 1

            if current_bank is not None:

                m = L2_HIT_RE.search(line)
                if m:
                    h, ms = map(int, m.groups())
                    data["l2"]["hits"] += h
                    data["l2"]["misses"] += ms

                m = L2_RW_RE.search(line)
                if m:
                    rh, wh = map(int, m.groups())
                    data["l2"]["read_hits"] += rh
                    data["l2"]["write_hits"] += wh

                m = L2_EVICT_RE.search(line)
                if m:
                    ev, wb = map(int, m.groups())
                    data["l2"]["evictions"] += ev
                    data["l2"]["writebacks"] += wb

    # ========================================================
    # Average DRAM per channel
    # ========================================================

    if data["dram_channels"] > 0:
        for k in data["dram"]:
            data["dram"][k] /= data["dram_channels"]

    # ========================================================
    # Average L2 per bank
    # ========================================================

    if data["l2_banks"] > 0:
        for k in data["l2"]:
            data["l2"][k] /= data["l2_banks"]

    return data

# ============================================================
# CSV Writer
# ============================================================

def write_csv(models, out_file):

    headers = [
        "File","Model","Cycles","Avg_Core_Tiles",
        "DRAM_Reads","DRAM_Writes",
        "RowHits","RowMisses","RowConflicts",
        "ReadReq","WriteReq","MemCycles",
        "L2_Hits","L2_Misses",
        "L2_ReadHits","L2_WriteHits",
        "L2_Evictions","L2_Writebacks"
    ]

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for m in models:

            avg_tiles = (
                sum(m["cores"].values()) / len(m["cores"])
                if m["cores"] else 0
            )

            d = m["dram"]
            l2 = m["l2"]

            writer.writerow([
                m["file"], m["model"], m["cycles"], avg_tiles,
                d["reads"], d["writes"],
                d["row_hits"], d["row_misses"], d["row_conflicts"],
                d["reads_req"], d["writes_req"], d["mem_cycles"],
                l2["hits"], l2["misses"],
                l2["read_hits"], l2["write_hits"],
                l2["evictions"], l2["writebacks"]
            ])

# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) < 3:
        print("Usage: python parse_models.py out.csv log1 log2 ...")
        return

    out_csv = sys.argv[1]
    logs = sys.argv[2:]

    models = [parse_log(log) for log in logs]

    write_csv(models, out_csv)

    print(f"\nCSV written → {out_csv}")

if __name__ == "__main__":
    main()
