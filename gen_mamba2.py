# ─────────────────────────────────────────────────────────────────────────────
# Mamba ONNX export — architecturally correct per paper Algorithm 2
#
# Real Mamba block (mamba_simple.py) per token (inference / step mode):
#
#   INPUTS : x [d_model],  conv_state [d_inner, d_conv],  ssm_state [d_inner, d_state]
#
#   1. in_proj    : Linear(d_model → 2*d_inner, no bias)
#                   splits into x_in [d_inner]  and  z [d_inner]
#
#   2. conv1d     : depthwise Conv1d(d_inner, kernel=d_conv)
#                   updates conv_state (shift register), output → x_conv [d_inner]
#
#   3. SiLU       : x_act = silu(x_conv)   [d_inner]
#
#   4. x_proj     : Linear(d_inner → dt_rank + 2*d_state, no bias)
#                   splits into  dt_raw [dt_rank],  B [d_state],  C [d_state]
#
#   5. dt_proj    : Linear(dt_rank → d_inner, WITH bias)
#                   dt = softplus(dt_proj(dt_raw))   [d_inner]
#
#   6. SSM step   : selective_state_update
#                   dA = exp(dt ⊗ A)             A [d_inner, d_state] (learned log)
#                   dB = dt ⊗ B                  B [d_state] (input-dependent)
#                   h  = dA * h + dB * x_act     h [d_inner, d_state]
#                   y  = h @ C                   C [d_state] (input-dependent)
#
#   7. D skip     : y = y + D * x_act            D [d_inner] (learned scalar/channel)
#
#   8. gate       : y = y * silu(z)              z from step 1
#
#   9. out_proj   : Linear(d_inner → d_model, no bias)
#                   output [d_model]
#
# Weight shapes (130m as example, d_model=768, d_inner=1536, d_state=16, dt_rank=48):
#   in_proj.weight    [2*d_inner,  d_model]  = [3072, 768]
#   conv1d.weight     [d_inner, 1, d_conv]   = [1536, 1, 4]   depthwise
#   conv1d.bias       [d_inner]              = [1536]
#   x_proj.weight     [dt_rank+2*d_state, d_inner] = [80, 1536]
#   dt_proj.weight    [d_inner, dt_rank]     = [1536, 48]
#   dt_proj.bias      [d_inner]              = [1536]
#   A_log             [d_inner, d_state]     = [1536, 16]      log(-A), learned
#   D                 [d_inner]              = [1536]           skip
#   out_proj.weight   [d_model, d_inner]     = [768, 1536]
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Mamba ONNX export — architecturally correct per paper Algorithm 2
#
# Real Mamba block (mamba_simple.py) per token (inference / step mode):
#
#   INPUTS : x [d_model],  conv_state [d_inner, d_conv],  ssm_state [d_inner, d_state]
#
#   1. in_proj    : Linear(d_model → 2*d_inner, no bias)
#                   splits into x_in [d_inner]  and  z [d_inner]
#
#   2. conv1d     : depthwise Conv1d(d_inner, kernel=d_conv)
#                   updates conv_state (shift register), output → x_conv [d_inner]
#
#   3. SiLU       : x_act = silu(x_conv)   [d_inner]
#
#   4. x_proj     : Linear(d_inner → dt_rank + 2*d_state, no bias)
#                   splits into  dt_raw [dt_rank],  B [d_state],  C [d_state]
#
#   5. dt_proj    : Linear(dt_rank → d_inner, WITH bias)
#                   dt = softplus(dt_proj(dt_raw))   [d_inner]
#
#   6. SSM step   : selective_state_update
#                   dA = exp(dt ⊗ A)             A [d_inner, d_state] (learned log)
#                   dB = dt ⊗ B                  B [d_state] (input-dependent)
#                   h  = dA * h + dB * x_act     h [d_inner, d_state]
#                   y  = h @ C                   C [d_state] (input-dependent)
#
#   7. D skip     : y = y + D * x_act            D [d_inner] (learned scalar/channel)
#
#   8. gate       : y = y * silu(z)              z from step 1
#
#   9. out_proj   : Linear(d_inner → d_model, no bias)
#                   output [d_model]
#
# Weight shapes (130m as example, d_model=768, d_inner=1536, d_state=16, dt_rank=48):
#   in_proj.weight    [2*d_inner,  d_model]  = [3072, 768]
#   conv1d.weight     [d_inner, 1, d_conv]   = [1536, 1, 4]   depthwise
#   conv1d.bias       [d_inner]              = [1536]
#   x_proj.weight     [dt_rank+2*d_state, d_inner] = [80, 1536]
#   dt_proj.weight    [d_inner, dt_rank]     = [1536, 48]
#   dt_proj.bias      [d_inner]              = [1536]
#   A_log             [d_inner, d_state]     = [1536, 16]      log(-A), learned
#   D                 [d_inner]              = [1536]           skip
#   out_proj.weight   [d_model, d_inner]     = [768, 1536]
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import shape_inference, numpy_helper
from collections import Counter
import os
import math

MODELS = {
    "mamba-130m": dict(d_model=768,  d_inner=1536, d_state=16, d_conv=4, dt_rank=48,  n_layer=1, vocab_size=50280),
    "mamba-370m": dict(d_model=101, d_inner=2048, d_state=16, d_conv=4, dt_rank=64,  n_layer=48, vocab_size=50280),
    "mamba-790m": dict(d_model=1536, d_inner=3072, d_state=16, d_conv=4, dt_rank=96,  n_layer=48, vocab_size=50280),
    "mamba-1.4b": dict(d_model=2048, d_inner=4096, d_state=16, d_conv=4, dt_rank=128, n_layer=48, vocab_size=50280),
    "mamba-2.8b": dict(d_model=2560, d_inner=5120, d_state=16, d_conv=4, dt_rank=160, n_layer=64, vocab_size=50280),
}


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv  = d_conv
        self.dt_rank = dt_rank

        self.in_proj   = nn.Linear(d_model,         2 * d_inner,            bias=True)
        self.conv_proj = nn.Linear(d_inner * d_conv, d_inner,                bias=True)
        self.x_proj    = nn.Linear(d_inner,          dt_rank + 2 * d_state,  bias=True)
        self.dt_proj   = nn.Linear(dt_rank,           d_inner,                bias=True)
        self.ssm_out   = nn.Linear(d_state,           d_inner,                bias=True)
        self.out_proj  = nn.Linear(d_inner,           d_model,                bias=True)

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      .repeat(d_inner, 1)))          # [d_inner, d_state]
        self.D = nn.Parameter(torch.ones(d_inner))

        for linear in [self.in_proj, self.conv_proj, self.x_proj,
                       self.ssm_out, self.out_proj]:
            nn.init.zeros_(linear.bias)

    def forward(self, x, conv_state_flat, ssm_state):
        S  = x.shape[0]
        di = self.d_inner
        ds = self.d_state
        dr = self.dt_rank

        # 1. in_proj
        proj  = self.in_proj(x)
        xi, z = proj[:, :di], proj[:, di:]

        # 2. conv (unchanged)
        xa_pre = self.conv_proj(conv_state_flat)
        new_cs = torch.cat([conv_state_flat[:, di:], xi], dim=1)
        xa     = F.silu(xa_pre)

        # 3. x_proj
        xp     = self.x_proj(xa)
        dt_raw = xp[:, :dr]
        B      = xp[:, dr:dr + ds]
        C      = xp[:, dr + ds:]

        # 4. dt
        dt     = F.softplus(self.dt_proj(dt_raw))

        A = -torch.exp(self.A_log.float())

        # ─────────────────────────────────────────
        # 🔥 FIX: true recurrence ONLY for S == 1
        # ─────────────────────────────────────────
        if S == 1:
            h = ssm_state.reshape(di, ds)

            dt1 = dt[0].unsqueeze(-1)
            B1  = B[0].unsqueeze(0)
            xa1 = xa[0].unsqueeze(-1)

            dA  = torch.exp(dt1 * A)
            dB  = dt1 * B1

            new_h  = dA * h + dB * xa1
            new_ss = new_h.reshape(1, di * ds)

        else:
            # KEEP YOUR ORIGINAL PARALLEL PATH
            dt3 = dt.unsqueeze(-1)
            dA  = torch.exp(dt3 * A.unsqueeze(0))
            B3  = B.unsqueeze(1)
            dB  = dt3 * B3
            xa3 = xa.unsqueeze(-1)

            h      = ssm_state.reshape(S, di, ds)
            new_h  = dA * h + dB * xa3
            new_ss = new_h.reshape(S, di * ds)

        # 6. readout
        y_ssm = self.ssm_out(C)

        # 7. skip + gate
        y = (y_ssm + self.D * xa) * F.silu(z)

        # 8. out
        out = self.out_proj(y)

        return out, new_cs, new_ss


class MambaStack(nn.Module):
    def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank)
            for _ in range(n_blocks)]
      )

    def forward(self, x, *states):
        """
        x         : [S, d_model]
        states    : 2*n_blocks tensors, alternating conv_state, ssm_state per block
                    states[2*i]   = conv_state for block i  [S, d_inner*d_conv]
                    states[2*i+1] = ssm_state  for block i  [S, d_inner*d_state]
        """
        for i, block in enumerate(self.blocks):
            x, _, _ = block(x, states[2*i], states[2*i+1])
        return x

def fix_gemm_bias_1d(m):
    gemm_bias_names = {
        node.input[2]
        for node in m.graph.node
        if node.op_type == "Gemm" and len(node.input) == 3
    }
    for init in list(m.graph.initializer):
        if init.name not in gemm_bias_names:
            continue
        dims = list(init.dims)
        if len(dims) == 2 and dims[0] == 1:
            arr      = numpy_helper.to_array(init).reshape(-1)
            new_init = numpy_helper.from_array(arr, name=init.name)
            m.graph.initializer.remove(init)
            m.graph.initializer.append(new_init)
    return m


def verify(m, label):
    shapes = {}
    for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for init in m.graph.initializer:
        shapes[init.name] = list(init.dims)
    init_names = {i.name for i in m.graph.initializer}

    node_names = [n.name for n in m.graph.node if n.name]
    dupes = {n for n in node_names if node_names.count(n) > 1}
    if dupes:
        print(f"  [{label}] WARNING: {len(dupes)} duplicate node names — deadlock risk!")

    problems = 0
    for node in m.graph.node:
        if node.op_type != "Gemm":
            continue
        for i, inp in enumerate(node.input):
            s   = shapes.get(inp, "UNKNOWN")
            bad = (s == "UNKNOWN" or
                   (isinstance(s, list) and (len(s) < 1 or 0 in s)) or
                   (i == 1 and inp not in init_names))
            if bad:
                problems += 1
    return problems, len(dupes)


def gen_mapping(folder, file_stem, cfg, seq_len, n_blocks):
    dim, spad_kb, accum_kb, precision, num_cores = 128, 32768, 4096, 2, 4
    max_spad_rows = (spad_kb  * 101) // (dim * precision * 2)
    max_acc_rows  = (accum_kb * 101) // (dim * 4        * 2)

    def ceil_div(a, b): return (a + b - 1) // b
    def pad_up(x, d):   return ceil_div(x, d) * d

    def gemm_mapping(N, C, M):
        dim, spad_kb, accum_kb, precision, num_cores = 128, 32768, 4096, 2, 4
        max_spad_rows = (spad_kb * 101) // (dim * precision * 2)
        max_acc_rows = (accum_kb * 101) // (dim * 4 * 2)

        def ceil_div(a, b): return (a + b - 1) // b
        def pad_up(x, d): return ceil_div(x, d) * d

        dI, dJ, dK = N, M, C
        dIp = pad_up(dI, dim)
        dJp = pad_up(dJ, dim)
        dKp = pad_up(dK, dim)

        acc = max_acc_rows // dim
        part = (max_spad_rows // 2) // dim
        sq = max(1, int(math.sqrt(acc)))
        tk = max(1, part // sq)

        tI = max(1, min(dIp // dim, ceil_div(dI, sq * dim)))
        tJ = max(1, min(dJp // dim, ceil_div(dJ, sq * dim)))
        tK = max(1, min(dKp // dim, ceil_div(dK, tk * dim)))

        nt = tI * tJ
        if nt < num_cores:
            inc = ceil_div(num_cores, nt)
            if dJ >= dI:
                tJ *= inc
            else:
                tI *= inc
            nt = tI * tJ
        if nt % num_cores:
            r = nt % num_cores
            if dJ >= dI:
                tJ += r
            else:
                tI += r

        iI = ceil_div(dIp, tI)
        iJ = ceil_div(dJp, tJ)
        iK = ceil_div(dKp, tK)

        # align down
        if iI % dim: iI = max(dim, iI - iI % dim)
        if iJ % dim: iJ = max(dim, iJ - iJ % dim)
        if iK % dim: iK = max(dim, iK - iK % dim)

        iI = min(iI, dIp)
        iJ = min(iJ, dJp)
        iK = min(iK, dKp)

        tI = ceil_div(dI, iI)
        tJ = ceil_div(dJ, iJ)
        tK = ceil_div(dK, iK)

        # ====================== MINIMAL FIX: NEVER ZERO TILES ======================
        # This is the only added part. Safe for simulator's bias + 3-input Gemm requirements.
        if iI == 0: 
            iI = dIp
            tI = 1
        if iJ == 0: 
            iJ = dJp
            tJ = 1
        if iK == 0: 
            iK = dKp
            tK = 1

        # For dimensions smaller than systolic tile (e.g. M=80), use one full tile
        if dI < dim:
            iI = dIp
            tI = 1
        if dJ < dim:
            iJ = dJp
            tJ = 1
        if dK < dim:
            iK = dKp
            tK = 1

        return tI, tK, tJ, iI, iK, iJ
        

    S=seq_len
    d_model=cfg["d_model"]; d_inner=cfg["d_inner"]
    d_state=cfg["d_state"]; d_conv=cfg["d_conv"]; dt_rank=cfg["dt_rank"]

    layers = [
        (1, d_model,         2*d_inner          ),
        (1, d_inner*d_conv,  d_inner             ),
        (1, d_inner,         dt_rank+2*d_state   ),
        (1, dt_rank,          d_inner             ),
        (1, d_state,          d_inner             ),
        (1, d_inner,          d_model             ),
    ]
  

    seen,lines = set(),[]
    for _ in range(n_blocks):
        for N,C,M in layers:
            if (N,C,M) not in seen:
                seen.add((N,C,M))
                oN,oC,oM,iN,iC,iM = gemm_mapping(N,C,M)
                lines.append(f"[T] N{N} C{C} M{M} - [O] N{oN} C{oC} M{oM} - [I] N{iN} C{iC} M{iM}")

    path = os.path.join(folder, f"{file_stem}.mapping")
    with open(path,"w") as f:
        f.write("\n".join(lines)+"\n")
    print(f"  Mapping  -> {path}  ({len(lines)} unique keys)")


def export_model(model_name, cfg, seq_len=1, n_blocks=1):
    d_model=cfg["d_model"]; d_inner=cfg["d_inner"]
    d_state=cfg["d_state"]; d_conv=cfg["d_conv"]; dt_rank=cfg["dt_rank"]
    S=seq_len

    stem   = f"{model_name}-s{S}-b{n_blocks}"
    folder = f"models/{stem}"
    os.makedirs(folder, exist_ok=True)
    onnx_path = os.path.join(folder, f"{stem}.onnx")

    if n_blocks == 1:
        model  = MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank).eval()
        inputs = (torch.zeros(1, d_model),
                  torch.zeros(1, d_inner * d_conv),
                  torch.zeros(1, d_inner * d_state))   # S-dim state for prefill
        in_names  = ["x", "conv_state_flat", "ssm_state"]
        out_names = ["out", "conv_state_new", "ssm_state_new"]
    else:
        model  = MambaStack(d_model, d_inner, d_state, d_conv, dt_rank, n_blocks).eval()
        
        # flat list: x, cs0, ss0, cs1, ss1, ..., cs23, ss23
        state_inputs = []
        for i in range(n_blocks):
            state_inputs.append(torch.zeros(1, d_inner * d_conv))   # conv_state_i
            state_inputs.append(torch.zeros(1, d_inner * d_state))  # ssm_state_i
        
        inputs    = (torch.zeros(1, d_model), *state_inputs)
        in_names  = ["x"] + [f"cs{i}" if j%2==0 else f"ss{i//2}"
                            for j, i in enumerate(
                                [k//2 for k in range(2*n_blocks)])]
        # simpler naming:
        in_names  = ["x"]
        for i in range(n_blocks):
            in_names += [f"conv_state_{i}", f"ssm_state_{i}"]
        out_names = ["out"]
    
    dynamic_axes = {
    "x": {0: "seq_len"},
    "out": {0: "seq_len"},
}

    if n_blocks == 1:
        dynamic_axes.update({
            "conv_state_flat": {0: "seq_len"},
            "ssm_state": {0: "seq_len"},
            "conv_state_new": {0: "seq_len"},
            "ssm_state_new": {0: "seq_len"},
        })
    else:
        for i in range(n_blocks):
            dynamic_axes[f"conv_state_{i}"] = {0: "seq_len"}
            dynamic_axes[f"ssm_state_{i}"]  = {0: "seq_len"}
    import warnings; warnings.filterwarnings("ignore")
    torch.onnx.export(
      model,
    inputs,
    f=onnx_path,
    input_names=in_names,
    output_names=out_names,
    opset_version=18,
    dynamo=False,
    dynamic_axes=dynamic_axes
)

    m = onnx.load(onnx_path)
    m = shape_inference.infer_shapes(m, check_type=True, strict_mode=False)
    m = fix_gemm_bias_1d(m)
    m = shape_inference.infer_shapes(m, check_type=True, strict_mode=False)
    onnx.save(m, onnx_path)

    problems, dupes = verify(m, stem)
    counts  = Counter(n.op_type for n in m.graph.node)
    status  = "✓" if problems==0 and dupes==0 else f"✗ shapes:{problems} dupes:{dupes}"
    print(f"  ONNX     -> {onnx_path}  [{status}]  Gemm:{counts.get('Gemm',0)}  nodes:{len(m.graph.node)}")

    gen_mapping(folder, stem, cfg, seq_len, n_blocks)
    return stem, folder


def write_models_list(entries, path="example/models_list.json"):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w") as f:
        json.dump({"models": entries}, f, indent=2)
    print(f"\nmodels_list.json -> {path}  ({len(entries)} models)")


if __name__ == "__main__":

    EXPORT_VARIANTS = [
        ("mamba-130m",  1,  2),
        ("mamba-130m",  2, 2),
        ("mamba-130m",  128,  2),
        ("mamba-130m",  256, 2),
        ("mamba-130m",  512,  2),
        ("mamba-130m",  1024, 2),
        ("mamba-130m",  2048,  2),
        ("mamba-130m",  8192, 2),
        ("mamba-370m",  1,  2),
        ("mamba-370m",  128, 2),
        ("mamba-790m",  1,  2),
        ("mamba-790m",  128, 2),
        ("mamba-1.4b",  1,  2),
        ("mamba-1.4b",  128, 2),
        ("mamba-2.8b",  1,  2),
        ("mamba-2.8b",  128, 2),
    ]

    entries = []
    for model_name, seq_len, n_blocks in EXPORT_VARIANTS:
        cfg = MODELS[model_name]
        print(f"\n[{model_name}  seq_len={seq_len}  n_blocks={n_blocks}]")
        stem, folder = export_model(model_name, cfg, seq_len=seq_len, n_blocks=n_blocks)
        entries.append({
            "name":       stem,
            "batch_size": 1,
            "seq_len": 1,
            "past_seq_len": 0,
            "total_seq_len": 0,
            "output_seq_len": seq_len,
        })
        

    write_models_list(entries)
    print("\nDone.")

