import math

# Config: 128x128 systolic array, 4 cores
# spad_size = 32768 KB, accum_size = 4096 KB
# precision = 2 bytes (fp16)

dim          = 128        # core_height = core_width
spad_kb      = 32768
accum_kb     = 4096
precision    = 2          # bytes
num_cores    = 4

spad_bytes   = spad_kb  * 1024
accum_bytes  = accum_kb * 1024

# Max rows available for double buffering
max_spad_rows  = (spad_bytes  // (dim * precision * 2))   # /2 for double buffer
max_acc_rows   = (accum_bytes // (dim * 4        * 2))    # /2 for double buffer, fp32 accum

def ceil_div(a, b):
    return (a + b - 1) // b

def pad(x, d):
    return ceil_div(x, d) * d

def gemm_mapping(N, C, M):
    """
    Reproduce the simulator's gemm_mapping() logic from Mapping.cpp.
    Returns (tile_out_N, tile_out_C, tile_out_M, tile_in_N, tile_in_C, tile_in_M)
    """
    dim_I = N   # batch / output rows
    dim_J = M   # output cols
    dim_K = C   # inner (reduction)

    dim_I_p = pad(dim_I, dim)
    dim_J_p = pad(dim_J, dim)
    dim_K_p = pad(dim_K, dim)

    db_partitions_rows   = max_spad_rows // 2
    db_mats_in_partition = db_partitions_rows // dim
    db_mats_in_acc       = max_acc_rows // dim

    db_max_tile_i_j = max(1, int(math.sqrt(db_mats_in_acc)))
    db_max_tile_k   = max(1, db_mats_in_partition // db_max_tile_i_j)

    tile_I = min(dim_I_p // dim, ceil_div(dim_I, db_max_tile_i_j * dim))
    tile_J = min(dim_J_p // dim, ceil_div(dim_J, db_max_tile_i_j * dim))
    tile_K = min(dim_K_p // dim, ceil_div(dim_K, db_max_tile_k   * dim))

    tile_I = max(1, tile_I)
    tile_J = max(1, tile_J)
    tile_K = max(1, tile_K)

    # ensure enough tiles for all cores
    num_tiles = tile_I * tile_J
    if num_tiles < num_cores:
        increase = ceil_div(num_cores, num_tiles)
        if dim_J >= dim_I:
            tile_J *= increase
        else:
            tile_I *= increase
        num_tiles = tile_I * tile_J

    if num_tiles % num_cores != 0:
        rem = num_tiles % num_cores
        if dim_J >= dim_I:
            tile_J += rem
        else:
            tile_I += rem

    inner_I = ceil_div(dim_I_p, tile_I)
    inner_J = ceil_div(dim_J_p, tile_J)
    inner_K = ceil_div(dim_K_p, tile_K)

    # align down to dim boundary
    inner_I = max(dim, inner_I - (inner_I % dim)) if inner_I % dim != 0 else inner_I
    inner_J = max(dim, inner_J - (inner_J % dim)) if inner_J % dim != 0 else inner_J
    inner_K = max(dim, inner_K - (inner_K % dim)) if inner_K % dim != 0 else inner_K

    # recompute tile counts after alignment
    tile_I = ceil_div(dim_I, inner_I)
    tile_J = ceil_div(dim_J, inner_J)
    tile_K = ceil_div(dim_K, inner_K)

    return (tile_I, tile_K, tile_J,   # outer loop: N, C, M
            inner_I, inner_K, inner_J) # inner loop: N, C, M

def mapping_line(N, C, M):
    oN, oC, oM, iN, iC, iM = gemm_mapping(N, C, M)
    return (f"[T] N{N} C{C} M{M} - "
            f"[O] N{oN} C{oC} M{oM} - "
            f"[I] N{iN} C{iC} M{iM}")

# ── Mamba-130M single block Gemm layers ──────────────────────────────────────
# Key: N = output rows (batch),  C = input features,  M = output features
# Gemm.cpp: key.N = _output_shape[Ndim] * _batch_size
#           key.C = _weight_shape[Cdim_w]   (K dim)
#           key.M = _weight_shape[Mdim]     (output features)
#
# All weights are stored as [M, C] with transB=1, so:
#   Cdim_w = 1  (when transB=1, Cdim_w=1)
#   Mdim   = 0  (when transB=1, Mdim=0)

layers = [
    # name,              N,  C,     M
    ("in_proj",          1,  768,   3072),
    ("conv_proj",        1,  6144,  1536),
    ("x_proj",           1,  1536,  80  ),
    ("dt_proj",          1,  48,    1536),
    ("ssm_readout",      1,  16,    1536),
    ("out_proj",         1,  1536,  768 ),
    
]

print("=" * 70)
print("Mamba-130M Single Block — Mapping File")
print("Config: 128x128 systolic, 4 cores, 32768KB spad, 4096KB accum, fp16")
print("=" * 70)

lines = []
for name, N, C, M in layers:
    line = mapping_line(N, C, M)
    lines.append(line)
    oN, oC, oM, iN, iC, iM = gemm_mapping(N, C, M)
    spad_needed  = (iN + iM) * iC * precision
    accum_needed =  iN * iM * 4   # fp32
    print(f"\n  {name:15s}  N={N} C={C} M={M}")
    print(f"    {line}")
    print(f"    spad needed: {spad_needed/1024:.1f} KB  "
          f"(limit: {spad_kb//2} KB)  "
          f"{'OK' if spad_needed <= spad_bytes//2 else 'OVERFLOW'}")
    print(f"    accum needed: {accum_needed/1024:.1f} KB  "
          f"(limit: {accum_kb//2} KB)  "
          f"{'OK' if accum_needed <= accum_bytes//2 else 'OVERFLOW'}")

# write mapping file
out_path = "models/mamba-130m-single/mamba-130m-single.mapping"
with open(out_path, "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"\n\nWritten -> {out_path}")
print("\nFinal mapping file contents:")
print("-" * 70)
for line in lines:
    print(line)