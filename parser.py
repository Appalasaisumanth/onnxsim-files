#!/usr/bin/env python3
"""
Simulator Log Parser
Parses hardware simulator logs for LLM/CNN inference.
Handles multi-channel DRAM stats, per-core SRAM stats, and L2 bank stats.
"""

import re
import sys
import json
from pathlib import Path
from collections import defaultdict
import os
import csv

def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            # Convert list → string (or you can expand if needed)
            items[new_key] = json.dumps(v)
        else:
            items[new_key] = v
    return items


def write_csv(rows, output_file):
    if not rows:
        return

    # Collect all keys across rows
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    fieldnames = sorted(all_keys)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> dict:
    with open(log_path, "r") as f:
        content = f.read()

    results = {}

    results["hardware_config"]       = _parse_hw_config(content)
    results["model_info"]            = _parse_model_info(content)
    results["memory_usage"]          = _parse_memory_usage(content)
    results["simulation_timing"]     = _parse_timing(content)
    results["per_core_stats"]        = _parse_per_core_stats(content)
    results["icnt_bandwidth"]        = _parse_icnt(content)
    results["dram_stats"]            = _parse_dram_stats(content)
    results["l2_bank_stats"]         = _parse_l2_stats(content)
    results["layer_timeline"]        = _parse_layer_timeline(content)
    results["gemm_ops"]              = _parse_gemm_ops(content)
    results["attention_ops"]         = _parse_attention_ops(content)
    results["object_size_histogram"] = _parse_object_histogram(content)
    results["summary_metrics"]       = _compute_summary(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Hardware config
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hw_config(content: str) -> dict:
    cfg = {}
    cfg["num_cores"]              = _first_int(r"num cores\s*:\s*\[(\d+)\]", content)
    cfg["dram_bandwidth_GBs"]     = _first_float(r"DRAM Bandwidth\s+(\d+)\s+GB/s", content)
    cfg["ramulator_config"]       = _first_str(r"Ramulator2 config:\s+(.+)", content)
    cfg["interconnect"]           = _first_str(r"Initialize\s+(\S+Interconnect)", content)
    cfg["tensor_parallelism"]     = _first_int(r"Tensor Parallelsim\s+(\d+)", content)
    cfg["pipeline_parallelism"]   = _first_int(r"Pipeline Parallelism\s+(\d+)", content)

    sa = re.search(
        r"\[Core 0\] Systolic Array Throughput:\s+([\d.]+)\s+GFLOPS,\s+"
        r"Spad size:\s+(\d+)\s+KB,\s+Accumulator size:\s+(\d+)\s+KB",
        content
    )
    if sa:
        cfg["systolic_throughput_GFLOPS"] = float(sa.group(1))
        cfg["spad_size_KB"]               = int(sa.group(2))
        cfg["accumulator_size_KB"]        = int(sa.group(3))

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────────────────────────────────────

def _parse_model_info(content: str) -> dict:
    m = {}
    m["name"]            = _first_str(r"Register Language Model:\s+(\S+)", content)
    m["num_layers_total"] = _first_int(r"num layer\s+(\d+)\s+num sim layer", content)
    m["num_sim_layers"]  = _first_int(r"num sim layer\s+(\d+)", content)
    m["has_l2"]          = _first_str(r"has_l2,\[(\w+)\]", content)
    m["weight_size_GB"]  = _first_float(r"Weight size:\s+([\d.]+)\s+GB", content)
    m["scheduler"]       = _first_str(
        r"(Simple Language scheduler|Language scheduler)\s+selected", content)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Memory usage
# ─────────────────────────────────────────────────────────────────────────────

def _parse_memory_usage(content: str) -> dict:
    mem = {}
    mem["total_GB"]      = _first_float(r"Total Memory Usage:\s+([\d.]+)\s+GB", content)
    mem["weight_GB"]     = _first_float(r"Weight Memory Usage:\s+([\d.]+)\s+GB", content)
    mem["kv_cache_GB"]   = _first_float(r"KV Memory Usage:\s+([\d.]+)\s+GB", content)
    mem["activation_GB"] = _first_float(r"Activation Memory Usage:\s+([\d.]+)\s+GB", content)
    return mem


# ─────────────────────────────────────────────────────────────────────────────
# Simulation timing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_timing(content: str) -> dict:
    t = {}
    t["total_cycles"]        = _first_int(r"Simulation Finished at (\d+) cycle", content)
    t["total_us"]            = _first_int(
        r"Simulation Finished at \d+ cycle\s+(\d+)\s+us", content)
    t["model_init_seconds"]  = _first_float(
        r"Model initialization time:\s+([\d.]+)\s+seconds", content)
    t["wall_clock_seconds"]  = _first_float(
        r"Simulation time:\s+([\d.]+)\s+seconds", content)
    t["request_finish_us"]   = _first_int(
        r"Request: \d+ us, Start: \d+ us, finish:(\d+)\s+us", content)
    t["avg_tiles_per_layer"] = _first_float(
        r"no\.of tiles\s*:\s*\[([\d.]+)\]", content)
    t["total_tiles"]         = _first_int(r"Total tile:\s+(\d+)", content)
    t["tiles_per_second"]    = _first_float(
        r"simulated tile per seconds\(TPS\):\s+([\d.]+)", content)
    t["total_compute_cycles"] = _first_int(r"Total compute time (\d+)", content)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Per-core stats — includes SRAM, Acc-SRAM, and core timing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sram_block(block_text: str, label: str) -> dict:
    s = {}
    s["hits"]            = _first_int(r"Hits\s*:\s*(\d+)", block_text)
    s["misses"]          = _first_int(r"Misses\s*:\s*(\d+)", block_text)
    s["hit_rate"]        = _first_float(r"Hit rate\s*:\s*([\d.]+)", block_text)
    s["read_hits"]       = _first_int(r"Read hits\s*:\s*(\d+)", block_text)
    s["write_hits"]      = _first_int(r"Write hits\s*:\s*(\d+)", block_text)
    s["prefetches"]      = _first_int(r"Prefetches\s*:\s*(\d+)", block_text)
    s["bytes_requested"] = _first_int(r"Bytes requested\s*:\s*(\d+)", block_text)
    s["bytes_received"]  = _first_int(r"Bytes received\s*:\s*(\d+)", block_text)

    total = (s["hits"] or 0) + (s["misses"] or 0)
    s["total_accesses"] = total
    if total > 0 and s["bytes_requested"] and s["bytes_received"]:
        s["prefetch_effectiveness_pct"] = round(
            (s["bytes_received"] - s["bytes_requested"]) /
            max(s["bytes_received"], 1) * 100, 4)
    return s


def _parse_per_core_stats(content: str) -> dict:
    cores = {}

    core_pattern = re.compile(
        r"(={5} SRAM stats ={5}.*?)"
        r"(={5} Acc-SRAM stats ={5}.*?)"
        r"Core \[(\d+)\] : MatMul active cycle (\d+) Vector active cycle (\d+)\s*\n"
        r".*?Memory unit idle cycle (\d+) Systolic bubble cycle (\d+) "
        r"Core idle cycle (\d+)\s*\n"
        r".*?Systolic Array Utilization\(%\) ([\d.]+) \(([\d.]+)% PE util\), "
        r"Vector Unit Utilization\(%\) ([\d.]+), Total cycle: (\d+)\s*\n"
        r".*?Tile Stats \| Issued: (\d+) Running: \d+ Finished: (\d+)\s*\n"
        r".*?ins executed \[(\d+)\] ins executed per tile ([\d.]+)\s*\n"
        r".*?Systolic Inst Issue Count : (\d+)\s*\n"
        r".*?Systolic PRELOAD Issue Count : (\d+)",
        re.DOTALL
    )

    for m in core_pattern.finditer(content):
        cid           = int(m.group(3))
        sram_text     = m.group(1)
        acc_sram_text = m.group(2)

        total_cycles  = int(m.group(12))
        matmul_cycles = int(m.group(4))
        vec_cycles    = int(m.group(5))
        mem_idle      = int(m.group(6))
        bubble_cycles = int(m.group(7))
        core_idle     = int(m.group(8))
        active_cycles = total_cycles - mem_idle - core_idle

        core_data = {
            "total_cycles":              total_cycles,
            "matmul_active_cycles":      matmul_cycles,
            "vector_active_cycles":      vec_cycles,
            "memory_unit_idle_cycles":   mem_idle,
            "systolic_bubble_cycles":    bubble_cycles,
            "core_idle_cycles":          core_idle,
            "active_cycles":             active_cycles,
            "memory_idle_pct":   round(mem_idle  / total_cycles * 100, 4) if total_cycles else 0,
            "core_idle_pct":     round(core_idle / total_cycles * 100, 4) if total_cycles else 0,
            "compute_util_pct":  round(active_cycles / total_cycles * 100, 4) if total_cycles else 0,
            "systolic_array_util_pct":   float(m.group(9)),
            "pe_util_pct":               float(m.group(10)),
            "vector_unit_util_pct":      float(m.group(11)),
            "tiles_issued":              int(m.group(13)),
            "tiles_finished":            int(m.group(14)),
            "instructions_executed":     int(m.group(15)),
            "instructions_per_tile":     float(m.group(16)),
            "systolic_inst_issue_count": int(m.group(17)),
            "systolic_preload_count":    int(m.group(18)),
            "sram":     _parse_sram_block(sram_text,     "SRAM"),
            "acc_sram": _parse_sram_block(acc_sram_text, "Acc-SRAM"),
        }
        cores[f"core_{cid}"] = core_data

    if cores:
        vals = list(cores.values())
        numeric_keys = [
            "total_cycles", "matmul_active_cycles", "vector_active_cycles",
            "memory_unit_idle_cycles", "systolic_bubble_cycles", "core_idle_cycles",
            "active_cycles", "memory_idle_pct", "core_idle_pct", "compute_util_pct",
            "systolic_array_util_pct", "pe_util_pct", "vector_unit_util_pct",
            "tiles_issued", "tiles_finished", "instructions_executed",
            "instructions_per_tile", "systolic_inst_issue_count", "systolic_preload_count",
        ]
        avg = {k: round(sum(v[k] for v in vals) / len(vals), 4) for k in numeric_keys}

        for sram_key in ["sram", "acc_sram"]:
            sram_numeric = [
                "hits", "misses", "read_hits", "write_hits",
                "prefetches", "bytes_requested", "bytes_received", "total_accesses"
            ]
            avg[sram_key] = {}
            for sk in sram_numeric:
                vals_sk = [v[sram_key][sk] for v in vals if v[sram_key].get(sk) is not None]
                avg[sram_key][sk] = round(sum(vals_sk) / len(vals_sk), 2) if vals_sk else None
            total_acc  = avg[sram_key]["total_accesses"] or 0
            total_hits = avg[sram_key]["hits"] or 0
            avg[sram_key]["hit_rate"] = round(total_hits / total_acc, 6) if total_acc else 0.0

        cores["avg_across_cores"] = avg

    return cores


# ─────────────────────────────────────────────────────────────────────────────
# ICNT bandwidth
# ─────────────────────────────────────────────────────────────────────────────

def _parse_icnt(content: str) -> dict:
    icnt = {}
    for direction in ["Core->ICNT", "Core<-ICNT", "ICNT->MEM", "ICNT<-MEM"]:
        key = direction.replace("->", "_to_").replace("<-", "_from_")
        pat = re.escape(direction) + r" request ([\d.]+)GB/Sec,(\d+)"
        m = re.search(pat, content)
        if m:
            icnt[key] = {
                "bandwidth_GBs": float(m.group(1)),
                "total_bytes":   int(m.group(2))
            }
    return icnt


# ─────────────────────────────────────────────────────────────────────────────
# DRAM / HBM channel stats
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dram_stats(content: str) -> dict:
    channel_stats = _parse_all_dram_channels(content)
    result = {}
    if channel_stats:
        result["num_channels_found"]  = len(channel_stats)
        result["per_channel_samples"] = channel_stats
        result["avg_across_channels"] = _average_dram_channels(channel_stats)

    bw = re.search(
        r"HBM2-CH_\d+: avg BW utilization (\d+)%\s+\((\d+) reads,\s+(\d+) writes\)",
        content
    )
    if bw:
        result["avg_bw_summary"] = {
            "avg_bw_utilization_pct": int(bw.group(1)),
            "total_reads":            int(bw.group(2)),
            "total_writes":           int(bw.group(3)),
        }
    return result


def _parse_all_dram_channels(content: str) -> list:
    blocks  = re.split(r"MemorySystem:", content)
    results = []
    for block in blocks[1:]:
        e = {}
        e["impl"]                     = _first_str(r"impl:\s+(\S+)", block)
        e["total_num_read_requests"]  = _first_int(r"total_num_read_requests:\s+(\d+)", block)
        e["total_num_write_requests"] = _first_int(r"total_num_write_requests:\s+(\d+)", block)
        e["total_num_other_requests"] = _first_int(r"total_num_other_requests:\s+(\d+)", block)
        e["memory_system_cycles"]     = _first_int(r"memory_system_cycles:\s+(\d+)", block)
        e["dram_impl"]                = _first_str(r"DRAM:\s*\n\s*impl:\s+(\S+)", block)
        e["addr_mapper"]              = _first_str(r"AddrMapper:\s*\n\s*impl:\s+(\S+)", block)
        e["scheduler"]                = _first_str(r"Scheduler:\s*\n\s*impl:\s+(\S+)", block)
        e["refresh_manager"]          = _first_str(r"RefreshManager:\s*\n\s*impl:\s+(\S+)", block)
        e["channel_id"]               = _first_str(r"id:\s+(Channel \d+)", block)
        results.append(e)

    row_stats = re.findall(
        r"Row hits:\s+(\d+),\s+Row misses:\s+(\d+),\s+Row conflicts:\s+(\d+)",
        content
    )
    for i, rs in enumerate(row_stats):
        if i < len(results):
            results[i]["row_hits"]      = int(rs[0])
            results[i]["row_misses"]    = int(rs[1])
            results[i]["row_conflicts"] = int(rs[2])

    return results


def _average_dram_channels(channels: list) -> dict:
    numeric_keys = [
        "total_num_read_requests", "total_num_write_requests",
        "total_num_other_requests", "memory_system_cycles",
        "row_hits", "row_misses", "row_conflicts",
    ]
    avg = {}
    for k in numeric_keys:
        vals = [c[k] for c in channels if k in c and c[k] is not None]
        avg[k] = round(sum(vals) / len(vals), 2) if vals else None

    for k in ["impl", "dram_impl", "addr_mapper", "scheduler", "refresh_manager"]:
        vals = [c[k] for c in channels if k in c and c[k] is not None]
        avg[k] = vals[0] if vals else None

    rh = avg.get("row_hits") or 0
    rm = avg.get("row_misses") or 0
    rc = avg.get("row_conflicts") or 0
    total = rh + rm + rc
    avg["row_hit_rate_pct"]      = round(rh / total * 100, 4) if total else None
    avg["row_miss_rate_pct"]     = round(rm / total * 100, 4) if total else None
    avg["row_conflict_rate_pct"] = round(rc / total * 100, 4) if total else None

    return avg


# ─────────────────────────────────────────────────────────────────────────────
# L2 bank stats
# ─────────────────────────────────────────────────────────────────────────────

def _parse_l2_stats(content: str) -> dict:
    bank_pattern = re.compile(
        r"={5} L2 BANK (\d+) \((\d+)-way, (\d+) sets\) ={5}\s*\n"
        r"(.*?)"
        r"(?=={5} L2 BANK|\Z)",
        re.DOTALL
    )

    banks = []
    for m in bank_pattern.finditer(content):
        bank_id  = int(m.group(1))
        num_ways = int(m.group(2))
        num_sets = int(m.group(3))
        body     = m.group(4)

        bank = {
            "bank_id":  bank_id,
            "num_ways": num_ways,
            "num_sets": num_sets,
        }

        bank["cycles"]         = _first_int(r"Cycles\s*:\s*(\d+)", body)
        bank["hits"]           = _first_int(r"Hits\s*:\s*(\d+)", body)
        bank["misses"]         = _first_int(r"Misses\s*:\s*(\d+)", body)
        bank["hit_rate"]       = _first_float(r"Hit rate\s*:\s*([\d.]+)", body)
        bank["read_hits"]      = _first_int(r"Read hits\s*:\s*(\d+)", body)
        bank["write_hits"]     = _first_int(r"Write hits\s*:\s*(\d+)", body)
        bank["evictions"]      = _first_int(r"Evictions\s*:\s*(\d+)", body)
        bank["conflicts"]      = _first_int(r"Conflicts\s*:\s*(\d+)", body)
        bank["writebacks"]     = _first_int(r"Writebacks\s*:\s*(\d+)", body)
        bank["pool_fallback"]  = _first_int(r"Pool fallback\s*:\s*(\d+)", body)
        bank["active_pending"] = _first_int(r"Active pending\s*:\s*(\d+)", body)
        bank["miss_q_depth"]   = _first_int(r"Miss Q depth\s*:\s*(\d+)", body)
        bank["pool_size"]      = _first_int(r"Pool size\s*:\s*(\d+)", body)

        cu = re.search(r"Cache util\s*:\s*([\d.]+)%\s+\((\d+)/(\d+) blocks\)", body)
        if cu:
            bank["cache_util_pct"]  = float(cu.group(1))
            bank["occupied_blocks"] = int(cu.group(2))
            bank["total_blocks"]    = int(cu.group(3))
            bank["free_blocks"]     = int(cu.group(3)) - int(cu.group(2))

        total = (bank["hits"] or 0) + (bank["misses"] or 0)
        bank["total_accesses"] = total
        bank["miss_rate"]      = round(1.0 - (bank["hit_rate"] or 0), 6)
        if total > 0:
            bank["read_hit_rate"]  = round((bank["read_hits"]  or 0) / total, 6)
            bank["write_hit_rate"] = round((bank["write_hits"] or 0) / total, 6)

        banks.append(bank)

    if not banks:
        return {}

    numeric_agg = [
        "hits", "misses", "read_hits", "write_hits",
        "evictions", "conflicts", "writebacks", "pool_fallback",
        "occupied_blocks", "total_blocks", "free_blocks", "total_accesses"
    ]
    agg = {}
    agg["num_banks"]    = len(banks)
    agg["num_ways"]     = banks[0]["num_ways"]
    agg["num_sets"]     = banks[0]["num_sets"]
    agg["cycles"]       = banks[0]["cycles"]
    agg["miss_q_depth"] = banks[0].get("miss_q_depth")
    agg["pool_size"]    = banks[0].get("pool_size")

    for k in numeric_agg:
        vals = [b[k] for b in banks if b.get(k) is not None]
        agg[f"total_{k}"] = sum(vals) if vals else None
        agg[f"avg_{k}"]   = round(sum(vals) / len(vals), 2) if vals else None

    total_acc  = agg["total_total_accesses"] or 0
    total_hits = agg["total_hits"] or 0
    agg["overall_hit_rate"]  = round(total_hits / total_acc, 6) if total_acc else 0.0
    agg["overall_miss_rate"] = round(1.0 - agg["overall_hit_rate"], 6)

    occ   = agg["total_occupied_blocks"] or 0
    total = agg["total_total_blocks"] or 1
    agg["overall_cache_util_pct"] = round(occ / total * 100, 4)

    hit_rates = [b["hit_rate"] for b in banks if b.get("hit_rate") is not None]
    if hit_rates:
        agg["min_bank_hit_rate"] = min(hit_rates)
        agg["max_bank_hit_rate"] = max(hit_rates)
        agg["hit_rate_variance"] = round(
            sum((r - agg["overall_hit_rate"])**2 for r in hit_rates) / len(hit_rates), 8)

    return {
        "per_bank":  banks,
        "aggregate": agg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer timeline
# ─────────────────────────────────────────────────────────────────────────────

def _parse_layer_timeline(content: str) -> list:
    finish_pat = re.compile(
        r"Layer (\S+) finish at (\d+)\s*\n"
        r"\[.*?\] Total compute time (\d+)"
    )
    layers = []
    prev_finish = 0
    for m in finish_pat.finditer(content):
        finish  = int(m.group(2))
        compute = int(m.group(3))
        entry = {
            "layer":          m.group(1),
            "finish_cycle":   finish,
            "compute_cycles": compute,
            "start_cycle":    finish - compute,
            "idle_cycles":    max(0, (finish - compute) - prev_finish),
            "type":           _classify_layer(m.group(1)),
        }
        layers.append(entry)
        prev_finish = finish
    return layers


def _classify_layer(name: str) -> str:
    if "QKVgen"    in name: return "QKV_projection"
    if "KVccat"    in name: return "KV_concat"
    if "Attention" in name: return "attention"
    if "Atccat"    in name: return "attn_concat"
    if "attn.proj" in name: return "attn_projection"
    if "attn.ln"   in name: return "attn_layernorm"
    if "ffn.fc1"   in name: return "ffn_fc1"
    if "ffn.act"   in name: return "ffn_activation"
    if "ffn.fc2"   in name: return "ffn_fc2"
    if "ffn.ln"    in name: return "ffn_layernorm"
    if "Gemm"      in name: return "gemm"
    if "conv"      in name: return "conv"
    if "Reshape"   in name: return "reshape"
    if "Mul"       in name: return "elementwise_mul"
    if "Add"       in name: return "elementwise_add"
    if "Exp"       in name: return "elementwise_exp"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# GEMM and attention op summaries
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gemm_ops(content: str) -> list:
    pat = re.compile(
        r"\[GemmWs\] Keys K = (\d+), N = (\d+), M = (\d+)\s*\n"
        r".*?\[GemmWS\]: total ([\d.]+) GFLOPs, ([\d.]+) GB\s*\n"
        r".*?\[GemmWS\]: Theoretical time\(ms\): ([\d.]+) "
        r"Compute time: ([\d.]+) Memory time: ([\d.]+)"
    )
    ops = []
    for m in pat.finditer(content):
        ct = float(m.group(7)); mt = float(m.group(8))
        ops.append({
            "K": int(m.group(1)), "N": int(m.group(2)), "M": int(m.group(3)),
            "gflops":               float(m.group(4)),
            "data_GB":              float(m.group(5)),
            "theoretical_ms":       float(m.group(6)),
            "compute_ms":           ct,
            "memory_ms":            mt,
            "bound":                "memory" if mt >= ct else "compute",
            "arithmetic_intensity": round(float(m.group(4)) / max(float(m.group(5)), 1e-9), 4),
        })
    return ops


def _parse_attention_ops(content: str) -> list:
    pat = re.compile(
        r"\[Attention\] q_len: (\d+), seq_len: (\d+), dk: (\d+), heads per tile (\d+)\s*\n"
        r".*?\[Attention\] Spad size (\d+)\s*\n"
        r".*?\[Attention\] Accum spad size (\d+)\s*\n"
        r".*?Mapping info.*?\n"
        r".*?\[Attention\] QK ([\d.]+) GFLOPs\s*\n"
        r".*?\[Attention\] total ([\d.]+) GFLOPs, ([\d.]+) GB\s*\n"
        r".*?\[Attention\] Theoretical time\(ms\): ([\d.]+) "
        r"Compute time: ([\d.]+) Memory time: ([\d.]+)"
    )
    ops = []
    for m in pat.finditer(content):
        ops.append({
            "q_len":          int(m.group(1)),
            "seq_len":        int(m.group(2)),
            "dk":             int(m.group(3)),
            "heads_per_tile": int(m.group(4)),
            "spad_size":      int(m.group(5)),
            "accum_size":     int(m.group(6)),
            "qk_gflops":      float(m.group(7)),
            "total_gflops":   float(m.group(8)),
            "data_GB":        float(m.group(9)),
            "theoretical_ms": float(m.group(10)),
            "compute_ms":     float(m.group(11)),
            "memory_ms":      float(m.group(12)),
        })
    return ops


# ─────────────────────────────────────────────────────────────────────────────
# Object size histogram
# ─────────────────────────────────────────────────────────────────────────────

def _parse_object_histogram(content: str) -> dict:
    pat = re.compile(r"(\d+) bytes : (\d+)\s+cumulative_bytes: (\d+)")
    entries = []
    for m in pat.finditer(content):
        sz  = int(m.group(1))
        cnt = int(m.group(2))
        entries.append({
            "size_bytes":       sz,
            "count":            cnt,
            "total_bytes":      sz * cnt,
            "cumulative_bytes": int(m.group(3)),
        })

    if not entries:
        return {}

    total_bytes = sum(e["total_bytes"] for e in entries)
    total_count = sum(e["count"] for e in entries)
    for e in entries:
        e["bytes_fraction_pct"] = round(
            e["total_bytes"] / total_bytes * 100, 4) if total_bytes else 0

    sorted_entries = sorted(entries, key=lambda x: -x["total_bytes"])

    return {
        "entries":              entries,
        "total_bytes":          total_bytes,
        "total_count":          total_count,
        "num_size_classes":     len(entries),
        "top_5_by_bytes":       sorted_entries[:5],
        "avg_object_size":      round(total_bytes / total_count, 2) if total_count else 0,
        "largest_object_bytes": max(e["size_bytes"] for e in entries) if entries else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary / derived metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_summary(r: dict) -> dict:
    s = {}

    gemm_total = sum(g["gflops"] for g in r.get("gemm_ops", []))
    attn_total = sum(a["total_gflops"] for a in r.get("attention_ops", []))
    s["total_compute_GFLOPs"] = round(gemm_total + attn_total, 6)

    us = (r.get("simulation_timing") or {}).get("total_us")
    if us and us > 0 and s["total_compute_GFLOPs"] > 0:
        s["effective_TFLOPS"] = round(s["total_compute_GFLOPs"] / us * 1e6 / 1e12, 8)

    t = r.get("simulation_timing") or {}
    s["total_simulated_cycles"] = t.get("total_cycles")
    s["wall_clock_seconds"]     = t.get("wall_clock_seconds")
    s["tiles_per_second"]       = t.get("tiles_per_second")
    if t.get("wall_clock_seconds") and t.get("total_cycles"):
        s["simulated_cycles_per_wall_second"] = round(
            t["total_cycles"] / t["wall_clock_seconds"], 0)

    core_vals = [v for k, v in (r.get("per_core_stats") or {}).items()
                 if k.startswith("core_")]
    if core_vals:
        s["avg_systolic_util_pct"] = round(
            sum(c["systolic_array_util_pct"] for c in core_vals) / len(core_vals), 4)
        s["avg_pe_util_pct"]       = round(
            sum(c["pe_util_pct"] for c in core_vals) / len(core_vals), 4)
        s["avg_memory_idle_pct"]   = round(
            sum(c["memory_idle_pct"] for c in core_vals) / len(core_vals), 4)
        s["avg_core_idle_pct"]     = round(
            sum(c["core_idle_pct"] for c in core_vals) / len(core_vals), 4)

        for sk in ["sram", "acc_sram"]:
            total_h = sum(c[sk]["hits"]   or 0 for c in core_vals)
            total_m = sum(c[sk]["misses"] or 0 for c in core_vals)
            total_a = total_h + total_m
            s[f"total_{sk}_hits"]     = total_h
            s[f"total_{sk}_misses"]   = total_m
            s[f"total_{sk}_hit_rate"] = round(total_h / total_a, 6) if total_a else 0.0

    bw = (r.get("dram_stats") or {}).get("avg_bw_summary", {})
    if bw:
        s["dram_avg_bw_utilization_pct"] = bw.get("avg_bw_utilization_pct")
        s["dram_total_reads"]            = bw.get("total_reads")
        s["dram_total_writes"]           = bw.get("total_writes")

    l2_agg = (r.get("l2_bank_stats") or {}).get("aggregate", {})
    if l2_agg:
        s["l2_overall_hit_rate"]       = l2_agg.get("overall_hit_rate")
        s["l2_overall_miss_rate"]      = l2_agg.get("overall_miss_rate")
        s["l2_overall_cache_util_pct"] = l2_agg.get("overall_cache_util_pct")
        s["l2_total_hits"]             = l2_agg.get("total_hits")
        s["l2_total_misses"]           = l2_agg.get("total_misses")
        s["l2_total_evictions"]        = l2_agg.get("total_evictions")
        s["l2_total_writebacks"]       = l2_agg.get("total_writebacks")
        s["l2_cycles"]                 = l2_agg.get("cycles")

    gemm_ops = r.get("gemm_ops", [])
    if gemm_ops:
        mem_bound = sum(1 for g in gemm_ops if g["bound"] == "memory")
        s["gemm_memory_bound_fraction"]      = round(mem_bound / len(gemm_ops), 4)
        s["gemm_avg_arithmetic_intensity"]   = round(
            sum(g["arithmetic_intensity"] for g in gemm_ops) / len(gemm_ops), 4)

    layer_cycles = defaultdict(int)
    layer_count  = defaultdict(int)
    for lyr in r.get("layer_timeline", []):
        layer_cycles[lyr["type"]] += lyr["compute_cycles"]
        layer_count [lyr["type"]] += 1
    s["compute_cycles_by_layer_type"] = dict(layer_cycles)
    s["layer_count_by_type"]          = dict(layer_count)

    return s


# ─────────────────────────────────────────────────────────────────────────────
# Regex helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_int(pattern, text):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None

def _first_float(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

def _first_str(pattern, text):
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary_json(data: dict) -> dict:
    m  = data.get("model_info", {})
    t  = data.get("simulation_timing", {})
    sm = data.get("summary_metrics", {})
    hw = data.get("hardware_config", {})

    # ── Per-core (ONLY AVG) ────────────────────────────────────────────────
    avg_c = data.get("per_core_stats", {}).get("avg_across_cores", {})
    per_core_rows = {}
    if avg_c:
        per_core_rows["avg"] = {
            "systolic_array_util_pct": avg_c.get("systolic_array_util_pct"),
            "pe_util_pct":             avg_c.get("pe_util_pct"),
            "memory_idle_pct":         avg_c.get("memory_idle_pct"),
            "tiles_finished":          avg_c.get("tiles_finished"),
            "sram_hit_rate":           avg_c.get("sram", {}).get("hit_rate"),
            "acc_sram_hit_rate":       avg_c.get("acc_sram", {}).get("hit_rate"),
        }

    # ── SRAM totals ────────────────────────────────────────────────────────
    sram_totals = {}
    for key, label in [("sram", "input_sram"), ("acc_sram", "acc_sram")]:
        sram_totals[label] = {
            "hits":     sm.get(f"total_{key}_hits", 0),
            "misses":   sm.get(f"total_{key}_misses", 0),
            "hit_rate": sm.get(f"total_{key}_hit_rate", 0),
        }

    # ── DRAM ──────────────────────────────────────────────────────────────
    bw       = (data.get("dram_stats") or {}).get("avg_bw_summary", {})
    dram_avg = (data.get("dram_stats") or {}).get("avg_across_channels", {})
    dram = {
        "avg_bw_utilization_pct": bw.get("avg_bw_utilization_pct"),
        "total_reads":            bw.get("total_reads"),
        "total_writes":           bw.get("total_writes"),
        "num_channels":           (data.get("dram_stats") or {}).get("num_channels_found", 0),
        "avg_row_hit_rate_pct":      dram_avg.get("row_hit_rate_pct"),
        "avg_row_miss_rate_pct":     dram_avg.get("row_miss_rate_pct"),
        "avg_row_conflict_rate_pct": dram_avg.get("row_conflict_rate_pct"),
        "avg_memory_system_cycles":  dram_avg.get("memory_system_cycles"),
    }

    # ── L2 cache (ONLY AGGREGATE) ─────────────────────────────────────────
    l2_summary = None
    l2 = data.get("l2_bank_stats", {})
    if l2:
        agg = l2.get("aggregate", {})
        l2_summary = {
            "num_banks":         agg.get("num_banks"),
            "num_ways":          agg.get("num_ways"),
            "num_sets":          agg.get("num_sets"),
            "cycles":            agg.get("cycles"),

            "overall_hit_rate":  agg.get("overall_hit_rate"),
            "overall_miss_rate": agg.get("overall_miss_rate"),
            "cache_util_pct":    agg.get("overall_cache_util_pct"),

            "total_hits":        agg.get("total_hits"),
            "total_misses":      agg.get("total_misses"),
            "total_evictions":   agg.get("total_evictions"),
            "total_writebacks":  agg.get("total_writebacks"),
            "total_conflicts":   agg.get("total_conflicts"),

            "miss_q_depth":      agg.get("miss_q_depth"),
            "pool_size":         agg.get("pool_size"),

            "avg_hits":          agg.get("avg_hits"),
            "avg_misses":        agg.get("avg_misses"),
            "avg_evictions":     agg.get("avg_evictions"),
            "avg_writebacks":    agg.get("avg_writebacks"),

            "min_bank_hit_rate": agg.get("min_bank_hit_rate"),
            "max_bank_hit_rate": agg.get("max_bank_hit_rate"),
            "hit_rate_variance": agg.get("hit_rate_variance"),
        }

    # ── Layer breakdown ───────────────────────────────────────────────────
    total_cyc = sum(sm.get("compute_cycles_by_layer_type", {}).values()) or 1
    layer_breakdown = {
        lt: {
            "cycles":  cyc,
            "pct":     round(cyc / total_cyc * 100, 2),
            "count":   sm.get("layer_count_by_type", {}).get(lt, 0),
        }
        for lt, cyc in sorted(
            sm.get("compute_cycles_by_layer_type", {}).items(),
            key=lambda x: -x[1]
        )
    }

    # ── Object histogram ──────────────────────────────────────────────────
    hist = data.get("object_size_histogram", {})
    histogram_summary = None
    if hist:
        histogram_summary = {
            "total_GB":            round(hist.get("total_bytes", 0) / 1e9, 3),
            "avg_object_size_B":   hist.get("avg_object_size", 0),
            "largest_object_B":    hist.get("largest_object_bytes", 0),
            "top_5_by_bytes": [
                {
                    "size_bytes": e["size_bytes"],
                    "count":      e["count"],
                    "total_MB":   round(e["total_bytes"] / 1e6, 1),
                    "bytes_pct":  e["bytes_fraction_pct"],
                }
                for e in (hist.get("top_5_by_bytes") or [])
            ],
        }

    # ── Assemble ──────────────────────────────────────────────────────────
    return {
        "model": {
            "name":             m.get("name"),
            "num_layers_total": m.get("num_layers_total"),
            "num_sim_layers":   m.get("num_sim_layers"),
            "weight_size_GB":   m.get("weight_size_GB"),
            "has_l2":           m.get("has_l2"),
            "scheduler":        m.get("scheduler"),
        },
        "hardware": {
            "num_cores":                  hw.get("num_cores"),
            "systolic_throughput_GFLOPS": hw.get("systolic_throughput_GFLOPS"),
            "spad_size_KB":               hw.get("spad_size_KB"),
            "accumulator_size_KB":        hw.get("accumulator_size_KB"),
            "dram_bandwidth_GBs":         hw.get("dram_bandwidth_GBs"),
            "tensor_parallelism":         hw.get("tensor_parallelism"),
            "pipeline_parallelism":       hw.get("pipeline_parallelism"),
        },
        "timing": {
            "total_cycles":         t.get("total_cycles"),
            "total_us":             t.get("total_us"),
            "total_compute_cycles": t.get("total_compute_cycles"),
            "wall_clock_seconds":   t.get("wall_clock_seconds"),
            "model_init_seconds":   t.get("model_init_seconds"),
            "total_tiles":          t.get("total_tiles"),
            "tiles_per_second":     t.get("tiles_per_second"),
            "cycles_per_wall_sec":  sm.get("simulated_cycles_per_wall_second"),
        },
        "compute": {
            "total_GFLOPs":                  sm.get("total_compute_GFLOPs"),
            "effective_TFLOPS":              sm.get("effective_TFLOPS"),
            "avg_systolic_util_pct":         sm.get("avg_systolic_util_pct"),
            "avg_pe_util_pct":               sm.get("avg_pe_util_pct"),
            "avg_memory_idle_pct":           sm.get("avg_memory_idle_pct"),
            "avg_core_idle_pct":             sm.get("avg_core_idle_pct"),
            "gemm_memory_bound_pct": (
                round(sm["gemm_memory_bound_fraction"] * 100, 1)
                if sm.get("gemm_memory_bound_fraction") is not None else None
            ),
            "gemm_avg_arithmetic_intensity": sm.get("gemm_avg_arithmetic_intensity"),
        },
        "per_core":         per_core_rows,
        "sram_totals":      sram_totals,
        "dram":             dram,
        "l2_cache":         l2_summary,
        "layer_breakdown":  layer_breakdown,
        "object_histogram": histogram_summary,
    }

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "sim.log"
    paths = os.listdir(log_path) if os.path.isdir(log_path) else [log_path]
    paths = [p for p in paths if p.endswith(".log")]

    all_parsed_rows = []
    all_summary_rows = []

    for log_file in paths:
        if not Path(log_file).exists():
            print(f"Error: file '{log_file}' not found.", file=sys.stderr)
            sys.exit(1)

        data = parse_log(log_file)

        base = log_file.replace(".log", "")

        # Write JSONs (existing)
        with open(base + "_parsed.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

        summary = _build_summary_json(data)

        with open(base + "_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # ── Flatten for CSV ──
        parsed_flat = flatten_dict(data)
        parsed_flat["file"] = log_file
        all_parsed_rows.append(parsed_flat)

        summary_flat = flatten_dict(summary)
        summary_flat["_file"] = log_file
        all_summary_rows.append(summary_flat)

    # ── Write combined CSVs ──
    write_csv(all_parsed_rows, "all_parsed.csv")
    write_csv(all_summary_rows, "all_summary.csv")

    print("CSV files generated:")
    print(" - all_parsed.csv")
    print(" - all_summary.csv")


if __name__ == "__main__":
    main()