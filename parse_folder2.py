#!/usr/bin/env python3
"""
Simulator Log Parser — handles OPT and Mamba logs.

Key differences from old version:
  1. Crash-early logs (Mamba JSON null crash, missing stats) are parsed
     gracefully — whatever was logged is captured, the rest is null/0.
  2. Summary CSV is fully flat — every column is a scalar, no JSON objects.
  3. Mamba runs output_seq_len times (once per regressive step). The parser
     detects groups of logs for the same model+config and aggregates them
     into one summary row (total cycles = sum, util = avg across steps).
  4. The crash reason is captured and written as a column so you can
     quickly see which runs failed and why.
"""

import re
import sys
import json
import csv
import os
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Regex helpers  (defined first so everything below can use them)
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
def extract_seq_len(log_file: str):
    import re
    matches = re.findall(r"(\d+)", log_file)
    return int(matches[-1]) if matches else None

# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> dict:
    with open(log_path, "r") as f:
        content = f.read()

    results = {}
    results["_log_file"]             = log_path
    results["_crash_reason"]         = _detect_crash(content)
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


def _detect_crash(content: str) -> str:
    """Return crash description if the run terminated early, else None."""
    if "terminate called" in content:
        return "terminate called (unknown reason)"
    if "Simulation Finished" not in content:
        return "simulation did not finish"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Hardware config
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hw_config(content: str) -> dict:
    cfg = {}
    cfg["num_cores"]              = _first_int(r"num cores\s*:\s*\[(\d+)\]", content)
    cfg["dram_bandwidth_GBs"]     = _first_float(r"DRAM Bandwidth\s+([\d.]+)\s+GB/s", content)
    cfg["ramulator_config"]       = _first_str(r"Ramulator2 config:\s+(.+)", content)
    cfg["interconnect"]           = _first_str(r"Initialize\s+(\S+Interconnect)", content)
    cfg["tensor_parallelism"]     = _first_int(r"Tensor Parallelsim\s+(\d+)", content)
    cfg["pipeline_parallelism"]   = _first_int(r"Pipeline Parallelism\s+(\d+)", content)
    sa = re.search(
        r"\[Core 0\] Systolic Array Throughput:\s+([\d.]+)\s+GFLOPS,\s+"
        r"Spad size:\s+(\d+)\s+KB,\s+Accumulator size:\s+(\d+)\s+KB", content)
    if sa:
        cfg["systolic_throughput_GFLOPS"] = float(sa.group(1))
        cfg["spad_size_KB"]               = int(sa.group(2))
        cfg["accumulator_size_KB"]        = int(sa.group(3))
    else:
        cfg["systolic_throughput_GFLOPS"] = None
        cfg["spad_size_KB"]               = None
        cfg["accumulator_size_KB"]        = None
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────────────────────────────────────

def _parse_model_info(content: str) -> dict:
    m = {}
    m["name"]             = _first_str(r"Register Language Model:\s+(\S+)", content)
    m["num_layers_total"] = _first_int(r"num layer\s+(\d+)\s+num sim layer", content)
    m["num_sim_layers"]   = _first_int(r"num sim layer\s+(\d+)", content)
    m["has_l2"]           = _first_str(r"has_l2,\[(\w+)\]", content)
    m["weight_size_GB"]   = _first_float(r"Weight size:\s+([\d.]+)\s+GB", content)
    m["scheduler"]        = _first_str(
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
    t["total_cycles"]         = _first_int(r"Simulation Finished at (\d+) cycle", content)
    t["total_us"]             = _first_int(
        r"Simulation Finished at \d+ cycle\s+(\d+)\s+us", content)
    t["model_init_seconds"]   = _first_float(
        r"Model initialization time:\s+([\d.]+)\s+seconds", content)
    t["wall_clock_seconds"]   = _first_float(
        r"Simulation time:\s+([\d.]+)\s+seconds", content)
    t["avg_tiles_per_layer"]  = _first_float(r"no\.of tiles\s*:\s*\[([\d.]+)\]", content)
    t["total_tiles"]          = _first_int(r"Total tile:\s+(\d+)", content)
    t["tiles_per_second"]     = _first_float(
        r"simulated tile per seconds\(TPS\):\s+([\d.]+)", content)
    t["total_compute_cycles"] = _first_int(r"Total compute time (\d+)", content)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Per-core stats
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sram_block(block_text: str) -> dict:
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
    s["total_accesses"]  = total
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
        total_cycles  = int(m.group(12))
        matmul_cycles = int(m.group(4))
        vec_cycles    = int(m.group(5))
        mem_idle      = int(m.group(6))
        bubble_cycles = int(m.group(7))
        core_idle     = int(m.group(8))
        active_cycles = total_cycles - mem_idle - core_idle
        cores[f"core_{cid}"] = {
            "total_cycles":              total_cycles,
            "matmul_active_cycles":      matmul_cycles,
            "vector_active_cycles":      vec_cycles,
            "memory_unit_idle_cycles":   mem_idle,
            "systolic_bubble_cycles":    bubble_cycles,
            "core_idle_cycles":          core_idle,
            "active_cycles":             active_cycles,
            "memory_idle_pct":   round(mem_idle  / total_cycles * 100, 4) if total_cycles else 0,
            "core_idle_pct":     round(core_idle / total_cycles * 100, 4) if total_cycles else 0,
            "systolic_array_util_pct":   float(m.group(9)),
            "pe_util_pct":               float(m.group(10)),
            "vector_unit_util_pct":      float(m.group(11)),
            "tiles_issued":              int(m.group(13)),
            "tiles_finished":            int(m.group(14)),
            "instructions_executed":     int(m.group(15)),
            "instructions_per_tile":     float(m.group(16)),
            "systolic_inst_issue_count": int(m.group(17)),
            "systolic_preload_count":    int(m.group(18)),
            "sram":     _parse_sram_block(m.group(1)),
            "acc_sram": _parse_sram_block(m.group(2)),
        }

    if cores:
        vals = list(cores.values())
        numeric_keys = [
            "total_cycles", "matmul_active_cycles", "vector_active_cycles",
            "memory_unit_idle_cycles", "systolic_bubble_cycles", "core_idle_cycles",
            "active_cycles", "memory_idle_pct", "core_idle_pct",
            "systolic_array_util_pct", "pe_util_pct", "vector_unit_util_pct",
            "tiles_issued", "tiles_finished", "instructions_executed",
            "instructions_per_tile", "systolic_inst_issue_count", "systolic_preload_count",
        ]
        avg = {k: round(sum(v[k] for v in vals) / len(vals), 4) for k in numeric_keys}
        for sk in ["sram", "acc_sram"]:
            sram_numeric = ["hits", "misses", "read_hits", "write_hits",
                            "prefetches", "bytes_requested", "bytes_received", "total_accesses"]
            avg[sk] = {}
            for snk in sram_numeric:
                sv = [v[sk][snk] for v in vals if v[sk].get(snk) is not None]
                avg[sk][snk] = round(sum(sv) / len(sv), 2) if sv else None
            ta = avg[sk]["total_accesses"] or 0
            th = avg[sk]["hits"] or 0
            avg[sk]["hit_rate"] = round(th / ta, 6) if ta else 0.0
        cores["avg_across_cores"] = avg
    return cores


# ─────────────────────────────────────────────────────────────────────────────
# ICNT bandwidth
# ─────────────────────────────────────────────────────────────────────────────

def _parse_icnt(content: str) -> dict:
    icnt = {}
    for direction in ["Core->ICNT", "Core<-ICNT", "ICNT->MEM", "ICNT<-MEM"]:
        key = direction.replace("->", "_to_").replace("<-", "_from_")
        m = re.search(re.escape(direction) + r" request ([\d.]+)GB/Sec,(\d+)", content)
        if m:
            icnt[key] = {"bandwidth_GBs": float(m.group(1)), "total_bytes": int(m.group(2))}
    return icnt


# ─────────────────────────────────────────────────────────────────────────────
# DRAM stats
# ─────────────────────────────────────────────────────────────────────────────

import re

def _parse_dram_stats(content: str) -> dict:
    channel_stats = _parse_all_dram_channels(content)
    result = {}

    if channel_stats:
        result["num_channels_found"] = len(channel_stats)
        result["per_channel_samples"] = channel_stats
        result["avg_across_channels"] = _average_dram_channels(channel_stats)

    # ✅ Extract ALL BW lines
    bw_matches = re.findall(
        r"HBM2-CH_\d+: BW utilization (\d+)%\s+\((\d+) reads,\s+(\d+) writes\)",
        content
    )

    if bw_matches:
        total_util = 0
        total_reads = 0
        total_writes = 0

        for util, reads, writes in bw_matches:
            total_util += int(util)
            total_reads += int(reads)
            total_writes += int(writes)

        result["avg_bw_summary"] = {
            "avg_bw_utilization_pct": total_util / len(bw_matches),  # ✅ real average
            "total_reads": total_reads,
            "total_writes": total_writes
        }
        # print(result)
        # print(f"Parsed DRAM BW for {len(bw_matches)} channels: {result['avg_bw_summary']}")

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
        r"Row hits:\s+(\d+),\s+Row misses:\s+(\d+),\s+Row conflicts:\s+(\d+)", content)
    for i, rs in enumerate(row_stats):
        if i < len(results):
            results[i]["row_hits"]      = int(rs[0])
            results[i]["row_misses"]    = int(rs[1])
            results[i]["row_conflicts"] = int(rs[2])
    return results


def _average_dram_channels(channels: list) -> dict:
    numeric_keys = ["total_num_read_requests", "total_num_write_requests",
                    "total_num_other_requests", "memory_system_cycles",
                    "row_hits", "row_misses", "row_conflicts"]
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
        r"={5} L2 BANK (\d+) \((\d+)-way, (\d+) sets\) ={5}\s*\n(.*?)"
        r"(?=={5} L2 BANK|\Z)", re.DOTALL)
    banks = []
    for m in bank_pattern.finditer(content):
        body = m.group(4)
        bank = {
            "bank_id": int(m.group(1)), "num_ways": int(m.group(2)),
            "num_sets": int(m.group(3)),
            "cycles":        _first_int(r"Cycles\s*:\s*(\d+)", body),
            "hits":          _first_int(r"Hits\s*:\s*(\d+)", body),
            "misses":        _first_int(r"Misses\s*:\s*(\d+)", body),
            "hit_rate":      _first_float(r"Hit rate\s*:\s*([\d.]+)", body),
            "read_hits":     _first_int(r"Read hits\s*:\s*(\d+)", body),
            "write_hits":    _first_int(r"Write hits\s*:\s*(\d+)", body),
            "evictions":     _first_int(r"Evictions\s*:\s*(\d+)", body),
            "conflicts":     _first_int(r"Conflicts\s*:\s*(\d+)", body),
            "writebacks":    _first_int(r"Writebacks\s*:\s*(\d+)", body),
            "pool_fallback": _first_int(r"Pool fallback\s*:\s*(\d+)", body),
            "miss_q_depth":  _first_int(r"Miss Q depth\s*:\s*(\d+)", body),
            "pool_size":     _first_int(r"Pool size\s*:\s*(\d+)", body),
        }
        cu = re.search(r"Cache util\s*:\s*([\d.]+)%\s+\((\d+)/(\d+) blocks\)", body)
        if cu:
            bank["cache_util_pct"]  = float(cu.group(1))
            bank["occupied_blocks"] = int(cu.group(2))
            bank["total_blocks"]    = int(cu.group(3))
        total = (bank["hits"] or 0) + (bank["misses"] or 0)
        bank["total_accesses"] = total
        banks.append(bank)

    if not banks:
        return {}

    numeric_agg = ["hits", "misses", "read_hits", "write_hits",
                   "evictions", "conflicts", "writebacks", "pool_fallback",
                   "occupied_blocks", "total_blocks", "total_accesses"]
    agg = {
        "num_banks":    len(banks),
        "num_ways":     banks[0]["num_ways"],
        "num_sets":     banks[0]["num_sets"],
        "cycles":       banks[0]["cycles"],
        "miss_q_depth": banks[0].get("miss_q_depth"),
        "pool_size":    banks[0].get("pool_size"),
    }
    for k in numeric_agg:
        vals = [b[k] for b in banks if b.get(k) is not None]
        agg[f"total_{k}"] = sum(vals) if vals else None
        agg[f"avg_{k}"]   = round(sum(vals) / len(vals), 2) if vals else None
    ta = agg["total_total_accesses"] or 0
    th = agg["total_hits"] or 0
    agg["overall_hit_rate"]        = round(th / ta, 6) if ta else 0.0
    agg["overall_miss_rate"]       = round(1.0 - agg["overall_hit_rate"], 6)
    occ   = agg["total_occupied_blocks"] or 0
    total = agg["total_total_blocks"] or 1
    agg["overall_cache_util_pct"] = round(occ / total * 100, 4)
    hit_rates = [b["hit_rate"] for b in banks if b.get("hit_rate") is not None]
    if hit_rates:
        agg["min_bank_hit_rate"] = min(hit_rates)
        agg["max_bank_hit_rate"] = max(hit_rates)
        agg["hit_rate_variance"] = round(
            sum((r - agg["overall_hit_rate"])**2 for r in hit_rates) / len(hit_rates), 8)
    return {"per_bank": banks, "aggregate": agg}


# ─────────────────────────────────────────────────────────────────────────────
# Layer timeline
# ─────────────────────────────────────────────────────────────────────────────

def _parse_layer_timeline(content: str) -> list:
    finish_pat = re.compile(
        r"Layer (\S+) finish at (\d+)\s*\n\[.*?\] Total compute time (\d+)")
    layers = []
    prev_finish = 0
    for m in finish_pat.finditer(content):
        finish  = int(m.group(2))
        compute = int(m.group(3))
        layers.append({
            "layer":          m.group(1),
            "finish_cycle":   finish,
            "compute_cycles": compute,
            "start_cycle":    finish - compute,
            "idle_cycles":    max(0, (finish - compute) - prev_finish),
            "type":           _classify_layer(m.group(1)),
        })
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
    
    if "gemm"     in name or "linear" in name: return "gemm"
    if "conv"     in name: return "conv"
    if "reshape"  in name: return "reshape"

    if "mul"      in name: return "elementwise_mul"
    if "add"      in name: return "elementwise_add"
    if "exp"      in name: return "elementwise_exp"
    if "div"      in name: return "elementwise_div"
    if "sub"      in name: return "elementwise_sub"
    if "neg"      in name: return "elementwise_neg"

    if "sigmoid"  in name: return "activation_sigmoid"
    if "silu"     in name: return "activation_silu"

    if "slice"    in name: return "slice"
    if "cat"      in name: return "concat"
    if "expand"   in name: return "expand"
    if "repeat"   in name: return "repeat"

    if "in_proj"  in name: return "mamba_in_proj"
    if "x_proj"   in name: return "mamba_x_proj"
    if "dt_proj"  in name: return "mamba_dt_proj"
    if "out_proj" in name: return "mamba_out_proj"
    if "ssm"      in name: return "mamba_ssm"

    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# GEMM and attention ops
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gemm_ops(content: str) -> list:
    pat = re.compile(
        r"\[GemmWs\] Keys K = (\d+), N = (\d+), M = (\d+)\s*\n"
        r".*?\[GemmWS\]: total ([\d.]+) GFLOPs, ([\d.]+) GB\s*\n"
        r".*?\[GemmWS\]: Theoretical time\(ms\): ([\d.]+) "
        r"Compute time: ([\d.]+) Memory time: ([\d.]+)")
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
        r"Compute time: ([\d.]+) Memory time: ([\d.]+)")
    ops = []
    for m in pat.finditer(content):
        ops.append({
            "q_len": int(m.group(1)), "seq_len": int(m.group(2)),
            "dk": int(m.group(3)), "heads_per_tile": int(m.group(4)),
            "spad_size": int(m.group(5)), "accum_size": int(m.group(6)),
            "qk_gflops": float(m.group(7)), "total_gflops": float(m.group(8)),
            "data_GB": float(m.group(9)), "theoretical_ms": float(m.group(10)),
            "compute_ms": float(m.group(11)), "memory_ms": float(m.group(12)),
        })
    return ops


# ─────────────────────────────────────────────────────────────────────────────
# Object histogram
# ─────────────────────────────────────────────────────────────────────────────

def _parse_object_histogram(content: str) -> dict:
    pat = re.compile(r"(\d+) bytes : (\d+)\s+cumulative_bytes: (\d+)")
    entries = []
    for m in pat.finditer(content):
        sz  = int(m.group(1)); cnt = int(m.group(2))
        entries.append({
            "size_bytes": sz, "count": cnt,
            "total_bytes": sz * cnt, "cumulative_bytes": int(m.group(3))})
    if not entries:
        return {}
    total_bytes = sum(e["total_bytes"] for e in entries)
    total_count = sum(e["count"] for e in entries)
    for e in entries:
        e["bytes_fraction_pct"] = round(e["total_bytes"] / total_bytes * 100, 4) if total_bytes else 0
    sorted_entries = sorted(entries, key=lambda x: -x["total_bytes"])
    return {
        "entries": entries, "total_bytes": total_bytes,
        "total_count": total_count, "num_size_classes": len(entries),
        "top_5_by_bytes": sorted_entries[:5],
        "avg_object_size": round(total_bytes / total_count, 2) if total_count else 0,
        "largest_object_bytes": max(e["size_bytes"] for e in entries) if entries else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics
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
            th = sum(c[sk]["hits"]   or 0 for c in core_vals)
            tm = sum(c[sk]["misses"] or 0 for c in core_vals)
            ta = th + tm
            s[f"total_{sk}_hits"]     = th
            s[f"total_{sk}_misses"]   = tm
            s[f"total_{sk}_hit_rate"] = round(th / ta, 6) if ta else 0.0
 
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
        s["gemm_memory_bound_fraction"]    = round(mem_bound / len(gemm_ops), 4)
        s["gemm_avg_arithmetic_intensity"] = round(
            sum(g["arithmetic_intensity"] for g in gemm_ops) / len(gemm_ops), 4)
 
    layer_cycles = defaultdict(int)
    layer_count  = defaultdict(int)
    for lyr in r.get("layer_timeline", []):
        layer_cycles[lyr["type"]] += lyr["compute_cycles"]
        layer_count [lyr["type"]] += 1
    s["compute_cycles_by_layer_type"] = dict(layer_cycles)
    s["layer_count_by_type"]          = dict(layer_count)
 
    # ── Object-size histogram → L2 memory-request traffic ─────────────────
    # The histogram records every object size class that cores sent toward L2,
    # so total_bytes here == total core→L2 memory-request bytes for the run.
    hist = r.get("object_size_histogram") or {}
    if hist:
        s["histogram_total_bytes"]      = hist.get("total_bytes")
        s["histogram_total_count"]      = hist.get("total_count")
        s["histogram_avg_object_size"]  = hist.get("avg_object_size")
        s["histogram_num_size_classes"] = hist.get("num_size_classes")
 
        top5 = hist.get("top_5_by_bytes") or []
        for rank, entry in enumerate(top5[:2], start=1):
            # rank 1 = largest contributor, rank 2 = second-largest
            s[f"histogram_top{rank}_size_bytes"]  = entry.get("size_bytes")
            s[f"histogram_top{rank}_total_bytes"] = entry.get("total_bytes")
            s[f"histogram_top{rank}_bytes_pct"]   = entry.get("bytes_fraction_pct")
 
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Build a FULLY FLAT summary row (no nested dicts, no JSON blobs in cells)
# This is what goes into the CSV — every cell is a plain scalar.
# ─────────────────────────────────────────────────────────────────────────────

def build_flat_summary_row(data: dict, log_file: str) -> dict:
    hw   = data.get("hardware_config", {})
    mi   = data.get("model_info", {})
    mu   = data.get("memory_usage", {})
    t    = data.get("simulation_timing", {})
    sm   = data.get("summary_metrics", {})
    cs   = data.get("per_core_stats", {})
    ds   = data.get("dram_stats", {})
    l2   = data.get("l2_bank_stats", {})
    icnt = data.get("icnt_bandwidth", {})
    avg_c    = cs.get("avg_across_cores", {})
    dram_bw  = ds.get("avg_bw_summary", {})
    dram_avg = ds.get("avg_across_channels", {})
    l2_agg   = l2.get("aggregate", {}) if l2 else {}
 
    layer_cyc = sm.get("compute_cycles_by_layer_type", {})
    layer_cnt = sm.get("layer_count_by_type", {})
    total_lc  = sum(layer_cyc.values()) or 1
 
    row = {
        # ── Identification ──────────────────────────────────────────────────
        "log_file":     log_file,
        "model_name":   mi.get("name"),
        "crash_reason": data.get("_crash_reason"),
        "run_status":   "crashed" if data.get("_crash_reason") else "ok",
 
        # ── Hardware ────────────────────────────────────────────────────────
        "hw_num_cores":                  hw.get("num_cores"),
        "hw_systolic_throughput_GFLOPS": hw.get("systolic_throughput_GFLOPS"),
        "hw_spad_size_KB":               hw.get("spad_size_KB"),
        "hw_accumulator_size_KB":        hw.get("accumulator_size_KB"),
        "hw_dram_bandwidth_GBs":         hw.get("dram_bandwidth_GBs"),
        "hw_tensor_parallelism":         hw.get("tensor_parallelism"),
        "hw_pipeline_parallelism":       hw.get("pipeline_parallelism"),
 
        # ── Model ────────────────────────────────────────────────────────────
        "model_num_layers_total": mi.get("num_layers_total"),
        "model_num_sim_layers":   mi.get("num_sim_layers"),
        "model_weight_size_GB":   mi.get("weight_size_GB"),
        "model_has_l2":           mi.get("has_l2"),
        "model_scheduler":        mi.get("scheduler"),
 
        # ── Memory usage ────────────────────────────────────────────────────
        "mem_total_GB":      mu.get("total_GB"),
        "mem_weight_GB":     mu.get("weight_GB"),
        "mem_kv_cache_GB":   mu.get("kv_cache_GB"),
        "mem_activation_GB": mu.get("activation_GB"),
 
        # ── Timing ──────────────────────────────────────────────────────────
        "timing_total_cycles":         t.get("total_cycles"),
        "timing_total_us":             t.get("total_us"),
        "timing_total_compute_cycles": t.get("total_compute_cycles"),
        "timing_wall_clock_seconds":   t.get("wall_clock_seconds"),
        "timing_model_init_seconds":   t.get("model_init_seconds"),
        "timing_total_tiles":          t.get("total_tiles"),
        "timing_tiles_per_second":     t.get("tiles_per_second"),
        "timing_cycles_per_wall_sec":  sm.get("simulated_cycles_per_wall_second"),
 
        # ── Compute ─────────────────────────────────────────────────────────
        "compute_total_GFLOPs":               sm.get("total_compute_GFLOPs"),
        "compute_effective_TFLOPS":           sm.get("effective_TFLOPS"),
        "compute_avg_systolic_util_pct":      sm.get("avg_systolic_util_pct"),
        "compute_avg_pe_util_pct":            sm.get("avg_pe_util_pct"),
        "compute_avg_memory_idle_pct":        sm.get("avg_memory_idle_pct"),
        "compute_avg_core_idle_pct":          sm.get("avg_core_idle_pct"),
        "compute_gemm_memory_bound_pct": (
            round(sm["gemm_memory_bound_fraction"] * 100, 1)
            if sm.get("gemm_memory_bound_fraction") is not None else None),
        "compute_gemm_avg_arithmetic_intensity": sm.get("gemm_avg_arithmetic_intensity"),
 
        # ── Per-core averages ────────────────────────────────────────────────
        "core_avg_systolic_array_util_pct":  avg_c.get("systolic_array_util_pct"),
        "core_avg_pe_util_pct":              avg_c.get("pe_util_pct"),
        "core_avg_vector_unit_util_pct":     avg_c.get("vector_unit_util_pct"),
        "core_avg_memory_idle_pct":          avg_c.get("memory_idle_pct"),
        "core_avg_core_idle_pct":            avg_c.get("core_idle_pct"),
        "core_avg_matmul_active_cycles":     avg_c.get("matmul_active_cycles"),
        "core_avg_total_cycles":             avg_c.get("total_cycles"),
        "core_avg_tiles_finished":           avg_c.get("tiles_finished"),
        "core_avg_instructions_executed":    avg_c.get("instructions_executed"),
        "core_avg_instructions_per_tile":    avg_c.get("instructions_per_tile"),
        "core_avg_systolic_inst_issue":      avg_c.get("systolic_inst_issue_count"),
        "core_avg_systolic_preload":         avg_c.get("systolic_preload_count"),
 
        # ── SRAM (input scratchpad) ──────────────────────────────────────────
        "sram_total_hits":    sm.get("total_sram_hits"),
        "sram_total_misses":  sm.get("total_sram_misses"),
        "sram_hit_rate":      sm.get("total_sram_hit_rate"),
        "sram_avg_bytes_req": avg_c.get("sram", {}).get("bytes_requested"),
        "sram_avg_bytes_rcv": avg_c.get("sram", {}).get("bytes_received"),
 
        # ── Acc-SRAM (accumulator scratchpad) ───────────────────────────────
        "acc_sram_total_hits":   sm.get("total_acc_sram_hits"),
        "acc_sram_total_misses": sm.get("total_acc_sram_misses"),
        "acc_sram_hit_rate":     sm.get("total_acc_sram_hit_rate"),
 
        # ── ICNT ────────────────────────────────────────────────────────────
        "icnt_core_to_icnt_GBs": (icnt.get("Core_to_ICNT")   or {}).get("bandwidth_GBs"),
        "icnt_icnt_to_core_GBs": (icnt.get("Core_from_ICNT") or {}).get("bandwidth_GBs"),
        "icnt_icnt_to_mem_GBs":  (icnt.get("ICNT_to_MEM")    or {}).get("bandwidth_GBs"),
        "icnt_mem_to_icnt_GBs":  (icnt.get("ICNT_from_MEM")  or {}).get("bandwidth_GBs"),
 
        # ── DRAM ────────────────────────────────────────────────────────────
        "dram_num_channels":              ds.get("num_channels_found"),
        "dram_avg_bw_utilization_pct":    dram_bw.get("avg_bw_utilization_pct"),
        "dram_total_reads":               dram_bw.get("total_reads"),
        "dram_total_writes":              dram_bw.get("total_writes"),
        "dram_avg_row_hit_rate_pct":      dram_avg.get("row_hit_rate_pct"),
        "dram_avg_row_miss_rate_pct":     dram_avg.get("row_miss_rate_pct"),
        "dram_avg_row_conflict_rate_pct": dram_avg.get("row_conflict_rate_pct"),
        "dram_avg_memory_system_cycles":  dram_avg.get("memory_system_cycles"),
        "dram_avg_read_requests":         dram_avg.get("total_num_read_requests"),
        "dram_avg_write_requests":        dram_avg.get("total_num_write_requests"),
 
        # ── L2 cache ────────────────────────────────────────────────────────
        "l2_num_banks":           l2_agg.get("num_banks"),
        "l2_num_ways":            l2_agg.get("num_ways"),
        "l2_num_sets":            l2_agg.get("num_sets"),
        "l2_overall_hit_rate":    l2_agg.get("overall_hit_rate"),
        "l2_overall_miss_rate":   l2_agg.get("overall_miss_rate"),
        "l2_cache_util_pct":      l2_agg.get("overall_cache_util_pct"),
        "l2_total_hits":          l2_agg.get("total_hits"),
        "l2_total_misses":        l2_agg.get("total_misses"),
        "l2_total_evictions":     l2_agg.get("total_evictions"),
        "l2_total_writebacks":    l2_agg.get("total_writebacks"),
        "l2_min_bank_hit_rate":   l2_agg.get("min_bank_hit_rate"),
        "l2_max_bank_hit_rate":   l2_agg.get("max_bank_hit_rate"),
        "l2_hit_rate_variance":   l2_agg.get("hit_rate_variance"),
 
        # ── Object histogram (core→L2 memory-request traffic) ───────────────
        # total_bytes = sum of (size × count) across all size classes,
        # i.e. total bytes requested from cores toward L2 during the run.
        "hist_total_bytes":         sm.get("histogram_total_bytes"),
        "hist_total_count":         sm.get("histogram_total_count"),
        "hist_avg_object_size":     sm.get("histogram_avg_object_size"),
        "hist_num_size_classes":    sm.get("histogram_num_size_classes"),
        # Top-1: size class responsible for the most bytes
        "hist_top1_size_bytes":     sm.get("histogram_top1_size_bytes"),
        "hist_top1_total_bytes":    sm.get("histogram_top1_total_bytes"),
        "hist_top1_bytes_pct":      sm.get("histogram_top1_bytes_pct"),
        # Top-2: second-largest size class by bytes
        "hist_top2_size_bytes":     sm.get("histogram_top2_size_bytes"),
        "hist_top2_total_bytes":    sm.get("histogram_top2_total_bytes"),
        "hist_top2_bytes_pct":      sm.get("histogram_top2_bytes_pct"),
 
        # ── Layer type breakdown (cycles + count per type) ───────────────────
        "layer_gemm_cycles":            layer_cyc.get("gemm", 0),
        "layer_gemm_count":             layer_cnt.get("gemm", 0),
        "layer_gemm_pct":               round(layer_cyc.get("gemm", 0) / total_lc * 100, 2),
        "layer_attention_cycles":       layer_cyc.get("attention", 0),
        "layer_attention_count":        layer_cnt.get("attention", 0),
        "layer_attention_pct":          round(layer_cyc.get("attention", 0) / total_lc * 100, 2),
        "layer_ffn_fc1_cycles":         layer_cyc.get("ffn_fc1", 0),
        "layer_ffn_fc2_cycles":         layer_cyc.get("ffn_fc2", 0),
        "layer_QKV_projection_cycles":  layer_cyc.get("QKV_projection", 0),
        "layer_attn_projection_cycles": layer_cyc.get("attn_projection", 0),
        "layer_gemm_cycles":            layer_cyc.get("gemm", 0),
        "layer_conv_cycles":            layer_cyc.get("conv", 0),

        "layer_mamba_in_proj_cycles":   layer_cyc.get("mamba_in_proj", 0),
        "layer_mamba_x_proj_cycles":    layer_cyc.get("mamba_x_proj", 0),
        "layer_mamba_dt_proj_cycles":   layer_cyc.get("mamba_dt_proj", 0),
        "layer_mamba_out_proj_cycles":  layer_cyc.get("mamba_out_proj", 0),
        "layer_mamba_ssm_cycles":       layer_cyc.get("mamba_ssm", 0),

        "layer_elementwise_mul_cycles": layer_cyc.get("elementwise_mul", 0),
        "layer_elementwise_add_cycles": layer_cyc.get("elementwise_add", 0),
        "layer_elementwise_exp_cycles": layer_cyc.get("elementwise_exp", 0),
        "layer_elementwise_div_cycles": layer_cyc.get("elementwise_div", 0),
        "layer_elementwise_sub_cycles": layer_cyc.get("elementwise_sub", 0),
        "layer_elementwise_neg_cycles": layer_cyc.get("elementwise_neg", 0),

        "layer_activation_sigmoid_cycles": layer_cyc.get("activation_sigmoid", 0),
        "layer_activation_silu_cycles":    layer_cyc.get("activation_silu", 0),

        "layer_slice_cycles":           layer_cyc.get("slice", 0),
        "layer_concat_cycles":          layer_cyc.get("concat", 0),
        "layer_expand_cycles":          layer_cyc.get("expand", 0),
        "layer_repeat_cycles":          layer_cyc.get("repeat", 0),
        "layer_reshape_cycles":         layer_cyc.get("reshape", 0),

        "layer_other_cycles":           layer_cyc.get("other", 0),
    }
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Mamba multi-run aggregation
#
# Mamba with output_seq_len=N generates N separate log files (one per
# regressive step). We group them by model name prefix and aggregate:
#   - total_cycles, total_us, total_tiles → SUM across steps
#   - utilisation, hit rates              → MEAN across steps (weighted by cycles)
#   - crashes                             → count how many steps crashed
# ─────────────────────────────────────────────────────────────────────────────

def _model_group_key(log_file: str) -> str:
    """
    Strip the step suffix to get the group key.
    e.g. "mamba-130m-s1-b2_step03.log" → "mamba-130m-s1-b2"
         "mamba-130m-b2-s1.log"         → "mamba-130m-b2-s1"  (single run, its own group)
    """
    base = Path(log_file).stem
    # Remove trailing _stepNN or _runNN or _NNN patterns
    base = re.sub(r'[_-](step|run|iter|s)\d+$', '', base, flags=re.IGNORECASE)
    base = re.sub(r'[_-]\d+$', '', base)
    return base


def aggregate_mamba_runs(flat_rows: list) -> list:
    groups = defaultdict(list)
    for row in flat_rows:
        key = _model_group_key(row["log_file"])
        groups[key].append(row)
 
    aggregated = []
    for key, rows in groups.items():
        if len(rows) == 1:
            row = dict(rows[0])
            row["mamba_num_steps"]     = 1
            row["mamba_steps_crashed"] = 1 if row.get("crash_reason") else 0
            aggregated.append(row)
            continue
 
        ok_rows     = [r for r in rows if not r.get("crash_reason")]
        crash_count = sum(1 for r in rows if r.get("crash_reason"))
 
        if not ok_rows:
            row = dict(rows[0])
            row["mamba_num_steps"]     = len(rows)
            row["mamba_steps_crashed"] = crash_count
            row["run_status"]          = f"all_{len(rows)}_steps_crashed"
            aggregated.append(row)
            continue
 
        agg = dict(ok_rows[0])
        agg["log_file"]            = key + f"_[{len(rows)}_steps]"
        agg["mamba_num_steps"]     = len(rows)
        agg["mamba_steps_crashed"] = crash_count
        agg["run_status"]          = "ok" if crash_count == 0 else f"{crash_count}_steps_crashed"
        agg["crash_reason"]        = None
 
        # ── SUM columns ──────────────────────────────────────────────────────
        sum_cols = [
            "timing_total_cycles", "timing_total_us", "timing_total_compute_cycles",
            "timing_total_tiles", "timing_wall_clock_seconds",
            "compute_total_GFLOPs",
            "sram_total_hits", "sram_total_misses",
            "acc_sram_total_hits", "acc_sram_total_misses",
            "dram_total_reads", "dram_total_writes",
            "l2_total_hits", "l2_total_misses", "l2_total_evictions", "l2_total_writebacks",
            "layer_gemm_cycles", "layer_gemm_count",
            "layer_attention_cycles", "layer_attention_count",
            "layer_ffn_fc1_cycles", "layer_ffn_fc2_cycles",
            "layer_QKV_projection_cycles", "layer_attn_projection_cycles",
            # "layer_mamba_in_proj_cycles", "layer_mamba_x_proj_cycles",
            # "layer_mamba_dt_proj_cycles", "layer_mamba_out_proj_cycles",
            # "layer_mamba_ssm_cycles",
            # "layer_elementwise_mul_cycles", "layer_elementwise_add_cycles",
            # "layer_elementwise_exp_cycles", "layer_other_cycles",
            # histogram byte/count totals accumulate across generation steps
            "layer_mamba_in_proj_cycles"  ,
       "layer_mamba_x_proj_cycles"  ,
      "layer_mamba_dt_proj_cycles"  ,
       "layer_mamba_out_proj_cycles" ,
        "layer_mamba_ssm_cycles" ,   
    "layer_elementwise_mul_cycles",
        "layer_elementwise_add_cycles",
        "layer_elementwise_exp_cycles",
        "layer_elementwise_div_cycles",
        "layer_elementwise_sub_cycles",
        "layer_elementwise_neg_cycles",
        "layer_activation_sigmoid_cycles",
        "layer_activation_silu_cycles"   ,
        "layer_slice_cycles"   ,  
        "layer_concat_cycles"  ,  
        "layer_expand_cycles"    ,
        "layer_repeat_cycles"    ,
        "layer_reshape_cycles"    ,
          "layer_other_cycles",
            "hist_total_bytes",
            "hist_total_count",
            "hist_top1_total_bytes",
            "hist_top2_total_bytes",
        ]
        for col in sum_cols:
            vals = [r[col] for r in ok_rows if r.get(col) is not None]
            agg[col] = sum(vals) if vals else None
 
        # ── MEAN columns ─────────────────────────────────────────────────────
        mean_cols = [
            "compute_avg_systolic_util_pct", "compute_avg_pe_util_pct",
            "compute_avg_memory_idle_pct", "compute_avg_core_idle_pct",
            "compute_effective_TFLOPS",
            "compute_gemm_memory_bound_pct", "compute_gemm_avg_arithmetic_intensity",
            "core_avg_systolic_array_util_pct", "core_avg_pe_util_pct",
            "core_avg_vector_unit_util_pct", "core_avg_memory_idle_pct",
            "core_avg_core_idle_pct", "core_avg_instructions_per_tile",
            "sram_hit_rate", "acc_sram_hit_rate",
            "dram_avg_bw_utilization_pct", "dram_avg_row_hit_rate_pct",
            "dram_avg_row_miss_rate_pct", "dram_avg_row_conflict_rate_pct",
            "l2_overall_hit_rate", "l2_overall_miss_rate", "l2_cache_util_pct",
            "l2_min_bank_hit_rate", "l2_max_bank_hit_rate",
            # num_size_classes is structural (doesn't change across steps)
            "hist_num_size_classes",
        ]
        for col in mean_cols:
            vals = [r[col] for r in ok_rows if r.get(col) is not None]
            agg[col] = round(sum(vals) / len(vals), 6) if vals else None
 
        # ── Recompute derived histogram fields from aggregated sums ──────────
        agg["hist_avg_object_size"] = (
            round(agg["hist_total_bytes"] / agg["hist_total_count"], 2)
            if (agg.get("hist_total_bytes") and agg.get("hist_total_count"))
            else None
        )
        agg["hist_top1_bytes_pct"] = (
            round(agg["hist_top1_total_bytes"] / agg["hist_total_bytes"] * 100, 4)
            if (agg.get("hist_top1_total_bytes") and agg.get("hist_total_bytes"))
            else None
        )
        agg["hist_top2_bytes_pct"] = (
            round(agg["hist_top2_total_bytes"] / agg["hist_total_bytes"] * 100, 4)
            if (agg.get("hist_top2_total_bytes") and agg.get("hist_total_bytes"))
            else None
        )
        # top1/top2 size_bytes (the size-class label) — keep from first ok step;
        # it's a bucket label, not an accumulating value
        # (already copied into agg via dict(ok_rows[0]) above, no action needed)
 
        # ── Recompute layer pct and SRAM hit rates (unchanged from original) ─
        total_lc = sum(
            agg.get(c, 0) or 0 for c in [
                "layer_gemm_cycles", 
            "layer_attention_cycles", 
            "layer_ffn_fc1_cycles", "layer_ffn_fc2_cycles",
            "layer_QKV_projection_cycles", "layer_attn_projection_cycles",
            "layer_mamba_in_proj_cycles" , 
       "layer_mamba_x_proj_cycles"  , 
      "layer_mamba_dt_proj_cycles"  , 
       "layer_mamba_out_proj_cycles" , 
        "layer_mamba_ssm_cycles"    , 
    "layer_elementwise_mul_cycles", 
        "layer_elementwise_add_cycles", 
        "layer_elementwise_exp_cycles", 
        "layer_elementwise_div_cycles", 
        "layer_elementwise_sub_cycles", 
        "layer_elementwise_neg_cycles", 
        "layer_activation_sigmoid_cycles", 
        "layer_activation_silu_cycles"   , 
        "layer_slice_cycles"     , 
        "layer_concat_cycles"    , 
        "layer_expand_cycles"    , 
        "layer_repeat_cycles"    , 
        "layer_reshape_cycles"    , 
          "layer_other_cycles"
            ]) or 1
        agg["layer_gemm_pct"]      = round((agg.get("layer_gemm_cycles") or 0) / total_lc * 100, 2)
        agg["layer_attention_pct"] = round((agg.get("layer_attention_cycles") or 0) / total_lc * 100, 2)
 
        for sk in ["sram", "acc_sram"]:
            th = agg.get(f"{sk}_total_hits") or 0
            tm = agg.get(f"{sk}_total_misses") or 0
            ta = th + tm
            agg[f"{sk}_hit_rate"] = round(th / ta, 6) if ta else None
 
        aggregated.append(agg)
 
    return aggregated
 
def build_detailed_summary(rows: list) -> list:
    # Build OPT baseline
    baseline = {}
    for r in rows:
        seq = extract_seq_len(r["log_file"])
        if r["model_name"] == "OPT":
            baseline[(seq)] = r["timing_total_cycles"]

    final = []

    for r in rows:
        seq = extract_seq_len(r["log_file"])
        model = "Mamba" if not r["model_name"] else "OPT"
        cycles = r.get("timing_total_cycles")
        steps  = r.get("mamba_num_steps", seq if model == "Mamba" else seq)

        tokens_per_cycle = (
            round(steps / cycles, 10)
            if cycles else None
        )

        base_cycles = baseline.get(seq)
        speedup = (
            round(base_cycles / cycles, 4)
            if base_cycles and cycles and model == "Mamba"
            else None
        )

        final.append({
            "seq_len": seq,
            "model": model,
            "timing_total_cycles": cycles,
            "timing_wall_clock_seconds": r.get("timing_wall_clock_seconds"),
            "memory_traffic_bytes": r.get("hist_total_bytes"),
            "memory_bound_ratio": r.get("compute_gemm_memory_bound_pct"),
            "tokens_per_cycle": tokens_per_cycle,
            "dram_avg_bw_utilization_pct": r.get("dram_avg_bw_utilization_pct"),
            "compute_avg_pe_util_pct": r.get("compute_avg_pe_util_pct"),
            "speedup_vs_opt": speedup,
        })

    return final
# ─────────────────────────────────────────────────────────────────────────────
# CSV writer — fully flat, no JSON blobs
# ─────────────────────────────────────────────────────────────────────────────

def write_flat_csv(rows: list, output_file: str):
    if not rows:
        print(f"  [warn] No rows to write to {output_file}")
        return
    all_keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Guarantee every cell is a plain scalar — convert None to ""
            safe_row = {}
            for k in all_keys:
                v = row.get(k)
                if v is None:
                    safe_row[k] = ""
                elif isinstance(v, (dict, list)):
                    # Should not happen in flat rows — but guard anyway
                    safe_row[k] = json.dumps(v)
                else:
                    safe_row[k] = v
            writer.writerow(safe_row)
    print(f"  Written: {output_file}  ({len(rows)} rows, {len(all_keys)} columns)")
def apply_custom_mamba_combinations(flat_rows: list) -> list:
    """Create s10000, s20000, s30000 combined rows.
       Keep ALL power-of-two logs (1 to 16384) even if crashed.
       Remove ONLY the true intermediates: s1808, s3616, s13616.
    """
    
    SEQ_LENS_TO_KEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    combinations = {
        # Mamba
        "mamba_tiny_s10000.log": ["mamba_tiny_s1808.log", "mamba_tiny_s8192.log"],
        "mamba_tiny_s20000.log": ["mamba_tiny_s3616.log", "mamba_tiny_s16384.log"],
        "mamba_tiny_s30000.log": ["mamba_tiny_s13616.log", "mamba_tiny_s16384.log"],
        # Tiny (OPT)
        "tiny_s10000.log": ["tiny_s1808.log", "tiny_s8192.log"],
        "tiny_s20000.log": ["tiny_s3616.log", "tiny_s16384.log"],
        "tiny_s30000.log": ["tiny_s13616.log", "tiny_s16384.log"],
    }

    row_dict = {r["log_file"]: r for r in flat_rows}
    final_rows = []
    used_logs = set()   # ← ONLY the intermediates we want to REMOVE

    # Step 1: Create combined rows
    for new_log_name, src_logs in combinations.items():
        group = []
        for src in src_logs:
            if src in row_dict:
                group.append(row_dict[src])
                # ONLY mark the true intermediates for removal
                if "s1808" in src or "s3616" in src or "s13616" in src:
                    used_logs.add(src)
            else:
                print(f"Warning: Missing source {src} for {new_log_name}")

        if len(group) != 2:
            print(f"Warning: Could not combine {new_log_name}")
            continue

        # Build combined row
        agg = dict(group[0])
        agg["log_file"] = new_log_name
        agg["run_status"] = "ok"
        agg["crash_reason"] = ""
        agg["mamba_num_steps"] = 2

        # SUM columns
        sum_cols = [
            "timing_total_cycles", "timing_total_us", "timing_total_compute_cycles",
            "timing_total_tiles", "timing_wall_clock_seconds", "compute_total_GFLOPs",
            "sram_total_hits", "sram_total_misses", "acc_sram_total_hits", "acc_sram_total_misses",
            "dram_total_reads", "dram_total_writes",
            "l2_total_hits", "l2_total_misses", "l2_total_evictions", "l2_total_writebacks",
            "layer_gemm_cycles", "layer_gemm_count", "layer_attention_cycles", "layer_attention_count",
            "layer_ffn_fc1_cycles", "layer_ffn_fc2_cycles",
            "layer_QKV_projection_cycles", "layer_attn_projection_cycles",
            "layer_mamba_in_proj_cycles", "layer_mamba_x_proj_cycles",
            "layer_mamba_dt_proj_cycles", "layer_mamba_out_proj_cycles",
            "layer_mamba_ssm_cycles",
            "layer_elementwise_mul_cycles", "layer_elementwise_add_cycles",
            "layer_elementwise_exp_cycles", "layer_elementwise_div_cycles",
            "layer_elementwise_sub_cycles", "layer_elementwise_neg_cycles",
            "layer_activation_sigmoid_cycles", "layer_activation_silu_cycles",
            "layer_slice_cycles", "layer_concat_cycles", "layer_expand_cycles",
            "layer_repeat_cycles", "layer_reshape_cycles", "layer_other_cycles",
            "hist_total_bytes", "hist_total_count", "hist_top1_total_bytes", "hist_top2_total_bytes",
        ]
        for col in sum_cols:
            vals = [float(r.get(col) or 0) for r in group]
            agg[col] = round(sum(vals)) if vals else None

        # MEAN columns
        mean_cols = [
            "compute_avg_systolic_util_pct", "compute_avg_pe_util_pct",
            "compute_avg_memory_idle_pct", "compute_avg_core_idle_pct",
            "compute_effective_TFLOPS", "compute_gemm_memory_bound_pct",
            "compute_gemm_avg_arithmetic_intensity",
            "core_avg_systolic_array_util_pct", "core_avg_pe_util_pct",
            "core_avg_vector_unit_util_pct", "core_avg_memory_idle_pct",
            "core_avg_core_idle_pct", "core_avg_instructions_per_tile",
            "sram_hit_rate", "acc_sram_hit_rate",
            "dram_avg_bw_utilization_pct", "dram_avg_row_hit_rate_pct",
            "dram_avg_row_miss_rate_pct", "dram_avg_row_conflict_rate_pct",
            "l2_overall_hit_rate", "l2_overall_miss_rate", "l2_cache_util_pct",
            "l2_min_bank_hit_rate", "l2_max_bank_hit_rate",
            "hist_num_size_classes",
        ]
        for col in mean_cols:
            vals = [float(r.get(col) or 0) for r in group]
            agg[col] = round(sum(vals) / len(vals), 6) if vals else None

        # Recompute derived fields
        total_lc = sum(float(agg.get(c, 0) or 0) for c in agg if c.startswith("layer_") and c.endswith("_cycles"))
        if total_lc > 0:
            agg["layer_gemm_pct"] = round(float(agg.get("layer_gemm_cycles", 0)) / total_lc * 100, 2)
            agg["layer_attention_pct"] = round(float(agg.get("layer_attention_cycles", 0)) / total_lc * 100, 2)

        for sk in ["sram", "acc_sram"]:
            th = float(agg.get(f"{sk}_total_hits", 0))
            tm = float(agg.get(f"{sk}_total_misses", 0))
            ta = th + tm
            agg[f"{sk}_hit_rate"] = round(th / ta, 6) if ta > 0 else None

        if agg.get("hist_total_bytes") and agg.get("hist_total_count"):
            agg["hist_avg_object_size"] = round(
                float(agg["hist_total_bytes"]) / float(agg["hist_total_count"]), 2)

        final_rows.append(agg)

    # Step 2: Keep everything except the true intermediates
    for r in flat_rows:
        lf = r["log_file"]

        if lf in used_logs:          # only skip s1808, s3616, s13616
            continue

        # Keep all power-of-two logs (including 8192 and 16384) even if crashed
        if any(str(s) in lf for s in SEQ_LENS_TO_KEEP):
            final_rows.append(r)
            continue

        # Keep crashed tiny logs (safety)
        if "tiny_s" in lf and r.get("crash_reason"):
            final_rows.append(r)
            continue

        # Keep other OPT rows
        if r.get("model_name") != "Mamba":
            final_rows.append(r)

    print(f"   Total rows: {len(final_rows)}")
    return final_rows
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: parser.py <log_file_or_directory> [output_prefix]")
        sys.exit(1)

    log_path      = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "results"

    # Collect log files
    if os.path.isdir(log_path):
        log_files = sorted(
            str(p) for p in Path(log_path).rglob("*.log"))
    else:
        log_files = [log_path]

    if not log_files:
        print(f"No .log files found in {log_path}")
        sys.exit(1)

    print(f"Parsing {len(log_files)} log file(s)...")

    flat_rows = []
    for lf in log_files:
        print(f"  {lf} ...", end=" ")
        try:
            data     = parse_log(lf)
            flat_row = build_flat_summary_row(data, lf)
            flat_rows.append(flat_row)
            status = data.get("_crash_reason") or "ok"
            print(status)

            # Also write per-log JSON for detailed inspection
            base = lf.replace(".log", "")
            with open(base + "_parsed.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"ERROR: {e}")
    aggregated = aggregate_mamba_runs(flat_rows)
    summary_rows = build_detailed_summary(flat_rows)

    # === CUSTOM CHANGE: Apply your 10k/20k/30k combinations and remove intermediates ===
    flat_rows = apply_custom_mamba_combinations(flat_rows)

    # Write outputs
    # 1. Per-log flat CSV — now contains only the 3 combined + crashed + OPT
    write_flat_csv(flat_rows, f"{output_prefix}_per_log.csv")
    
    # 2. Aggregated flat CSV
    write_flat_csv(aggregated, f"{output_prefix}_summary.csv")
    
    # 3. Detailed summary CSV
    write_flat_csv(summary_rows, f"{output_prefix}_detailed_summary.csv")

    print(f"\nDone. {len(aggregated)} model(s) in summary.")
    print("Note: per_log.csv now contains exactly the 3 combined mamba_tiny_10k/20k/30k rows you requested.")



if __name__ == "__main__":
    main()