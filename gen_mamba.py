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
        """
        x              : [S, d_model]
        conv_state_flat: [S, d_inner * d_conv]
        ssm_state      : [S, d_inner * d_state]

        For S=1  → decode/autoregressive step
        For S>1  → prefill (all tokens parallel, no recurrence across tokens)
        """
        S  = x.shape[0]
        di = self.d_inner
        ds = self.d_state
        dr = self.dt_rank

        # 1. in_proj  Gemm [S, d_model] → [S, 2*d_inner]
        proj  = self.in_proj(x)
        xi, z = proj[:, :di], proj[:, di:]           # each [S, d_inner]

        # 2. conv_proj  Gemm [S, d_inner*d_conv] → [S, d_inner]
        xa_pre = self.conv_proj(conv_state_flat)
        new_cs = torch.cat([conv_state_flat[:, di:], xi], dim=1)  # [S, d_inner*d_conv]
        xa     = F.silu(xa_pre)                                    # [S, d_inner]

        # 3. x_proj  Gemm [S, d_inner] → [S, dt_rank+2*d_state]
        xp     = self.x_proj(xa)
        dt_raw = xp[:, :dr]                           # [S, dt_rank]
        B      = xp[:, dr:dr + ds]                    # [S, d_state]
        C      = xp[:, dr + ds:]                      # [S, d_state]

        # 4. dt_proj  Gemm [S, dt_rank] → [S, d_inner]
        dt     = F.softplus(self.dt_proj(dt_raw))     # [S, d_inner]

        # 5. SSM state update — all shapes keep S dimension
        #    A_log: [d_inner, d_state]
        #    dt:    [S, d_inner]
        #    B:     [S, d_state]
        #    xa:    [S, d_inner]
        #    h:     [S, d_inner, d_state]  (one state per token position)
        A   = -torch.exp(self.A_log.float())          # [d_inner, d_state]

        # expand dt: [S, d_inner] → [S, d_inner, 1] for broadcasting
        dt3 = dt.unsqueeze(-1)                        # [S, d_inner, 1]

        # dA: [S, d_inner, d_state]
        dA  = torch.exp(dt3 * A.unsqueeze(0))         # A: [1, d_inner, d_state]

        # dB: [S, d_inner, d_state]
        B3  = B.unsqueeze(1)                          # [S, 1, d_state]
        dB  = dt3 * B3                                # [S, d_inner, d_state]

        # xa expanded: [S, d_inner, 1]
        xa3 = xa.unsqueeze(-1)                        # [S, d_inner, 1]

        # h: [S, d_inner, d_state]
        h      = ssm_state.reshape(S, di, ds)
        new_h  = dA * h + dB * xa3                   # [S, d_inner, d_state]
        new_ss = new_h.reshape(S, di * ds)            # [S, d_inner*d_state]

        # 6. SSM readout  Gemm [S, d_state] → [S, d_inner]
        y_ssm  = self.ssm_out(C)                      # [S, d_inner]

        # 7. D skip + gate
        y      = (y_ssm + self.D * xa) * F.silu(z)   # [S, d_inner]

        # 8. out_proj  Gemm [S, d_inner] → [S, d_model]
        out    = self.out_proj(y)

        return out, new_cs, new_ss


class MambaStack(nn.Module):
    def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank)
            for _ in range(n_blocks)
        ])

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
        dI, dJ, dK = N, M, C
        dIp = pad_up(dI, dim)
        dJp = pad_up(dJ, dim)
        dKp = pad_up(dK, dim)

        acc  = max_acc_rows // dim
        part = (max_spad_rows // 2) // dim
        sq   = max(1, int(math.sqrt(acc)))
        tk   = max(1, part // sq)

        tI = max(1, min(dIp // dim, ceil_div(dI, sq * dim)))
        tJ = max(1, min(dJp // dim, ceil_div(dJ, sq * dim)))
        tK = max(1, min(dKp // dim, ceil_div(dK, tk * dim)))

        nt = tI * tJ
        if nt < num_cores:
            inc = ceil_div(num_cores, nt)
            if dJ >= dI: tJ *= inc
            else:        tI *= inc
            nt = tI * tJ
        if nt % num_cores:
            r = nt % num_cores
            if dJ >= dI: tJ += r
            else:        tI += r

        iI = ceil_div(dIp, tI)
        iJ = ceil_div(dJp, tJ)
        iK = ceil_div(dKp, tK)

        # align down to dim boundary
        if iI % dim: iI = max(dim, iI - iI % dim)
        if iJ % dim: iJ = max(dim, iJ - iJ % dim)
        if iK % dim: iK = max(dim, iK - iK % dim)

        # ── CLAMP AFTER ALIGNMENT ──────────────────────────────────────────
        # inner tile must not exceed actual problem size.
        # Use pad_up so we round up to nearest dim multiple ≤ actual padded size.
        iI = min(iI, dIp)
        iJ = min(iJ, dJp)
        iK = min(iK, dKp)

        # recompute outer tile counts
        tI = ceil_div(dI, iI)
        tJ = ceil_div(dJ, iJ)
        tK = ceil_div(dK, iK)

        return tI, tK, tJ, iI, iK, iJ


        

    S=seq_len
    d_model=cfg["d_model"]; d_inner=cfg["d_inner"]
    d_state=cfg["d_state"]; d_conv=cfg["d_conv"]; dt_rank=cfg["dt_rank"]

    layers = [
        (S, d_model,         2*d_inner          ),
        (S, d_inner*d_conv,  d_inner             ),
        (S, d_inner,         dt_rank+2*d_state   ),
        (S, dt_rank,          d_inner             ),
        (S, d_state,          d_inner             ),
        (S, d_inner,          d_model             ),
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
        inputs = (torch.zeros(S, d_model),
                  torch.zeros(S, d_inner * d_conv),
                  torch.zeros(S, d_inner * d_state))   # S-dim state for prefill
        in_names  = ["x", "conv_state_flat", "ssm_state"]
        out_names = ["out", "conv_state_new", "ssm_state_new"]
    else:
        model  = MambaStack(d_model, d_inner, d_state, d_conv, dt_rank, n_blocks).eval()
        
        # flat list: x, cs0, ss0, cs1, ss1, ..., cs23, ss23
        state_inputs = []
        for i in range(n_blocks):
            state_inputs.append(torch.zeros(S, d_inner * d_conv))   # conv_state_i
            state_inputs.append(torch.zeros(S, d_inner * d_state))  # ssm_state_i
        
        inputs    = (torch.zeros(S, d_model), *state_inputs)
        in_names  = ["x"] + [f"cs{i}" if j%2==0 else f"ss{i//2}"
                            for j, i in enumerate(
                                [k//2 for k in range(2*n_blocks)])]
        # simpler naming:
        in_names  = ["x"]
        for i in range(n_blocks):
            in_names += [f"conv_state_{i}", f"ssm_state_{i}"]
        out_names = ["out"]

    import warnings; warnings.filterwarnings("ignore")
    torch.onnx.export(
        model, inputs, f=onnx_path,
        input_names=in_names, output_names=out_names,
        opset_version=18, dynamo=False,
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
        ("mamba-130m",  128, 2),
    ]

    entries = []
    for model_name, seq_len, n_blocks in EXPORT_VARIANTS:
        cfg = MODELS[model_name]
        print(f"\n[{model_name}  seq_len={seq_len}  n_blocks={n_blocks}]")
        stem, folder = export_model(model_name, cfg, seq_len=seq_len, n_blocks=n_blocks)
        entries.append({
            "name":       stem,
            "batch_size": 1,
            "seq_len":seq_len
        })

    write_models_list(entries)
    print("\nDone.")




    '''
    The ONNX graph is not what you expect. The issue is in your Python export — your `MambaBlock` is **not implementing the real Mamba algorithm**. Let me break down what's wrong:

## What the ONNX nodes reveal

The pattern repeating twice (for 2 blocks) is:
```
Gemm → Slice → Gemm → Sigmoid → Mul → Gemm → Slice → Gemm → Mul → Add → Sigmoid → Mul → Mul → Gemm
```

That's only **4 Gemms per block** but you should have **6 Gemms**. More critically, there's **no SSM state update** (no `Exp`, no elementwise `dA`/`dB` computation, no `[S, d_inner, d_state]` tensor ops).

## Root cause in your Python code

```python
# Step 6 — SSM readout is WRONG
y_ssm = self.ssm_out(C)   # [S, d_state] → [S, d_inner]
```

Your `ssm_out` is a `Linear(d_state → d_inner)` applied only to `C`, **completely bypassing the actual SSM recurrence** (`dA`, `dB`, `h` update). The real computation:

```python
# What you have (WRONG):
y_ssm = self.ssm_out(C)   # just a linear on C, ignores h entirely

# What it should be:
A   = -torch.exp(self.A_log.float())           # [d_inner, d_state]
dA  = torch.exp(dt.unsqueeze(-1) * A)          # [S, d_inner, d_state]
dB  = dt.unsqueeze(-1) * B.unsqueeze(1)        # [S, d_inner, d_state]
h   = ssm_state.reshape(S, d_inner, d_state)
new_h = dA * h + dB * xa.unsqueeze(-1)         # [S, d_inner, d_state]
y_ssm = (new_h * C.unsqueeze(1)).sum(-1)       # [S, d_inner]  ← h @ C contraction
```

## Also wrong: `conv_proj` 

```python
# You have (WRONG):
self.conv_proj = nn.Linear(d_inner * d_conv, d_inner, bias=True)
xa_pre = self.conv_proj(conv_state_flat)   # flattened linear, not depthwise conv

# Should be depthwise conv1d — approximated for ONNX as:
# per-channel multiply+sum over d_conv kernel positions
```

## Corrected MambaBlock forward

```python
def forward(self, x, conv_state_flat, ssm_state):
    S  = x.shape[0]
    di = self.d_inner
    ds = self.d_state
    dr = self.dt_rank

    # 1. in_proj: [S, d_model] → [S, 2*d_inner]
    proj = self.in_proj(x)
    xi, z = proj[:, :di], proj[:, di:]

    # 2. depthwise conv (approximated as grouped linear over conv window)
    #    conv_state: [S, d_inner * d_conv], shift and apply
    new_cs = torch.cat([conv_state_flat[:, di:], xi], dim=1)  # shift register
    #    reshape to [S, d_inner, d_conv], apply per-channel weights
    cs_3d   = new_cs.reshape(S, di, self.d_conv)              # [S, d_inner, d_conv]
    w_conv  = self.conv_proj.weight.reshape(di, self.d_conv)  # [d_inner, d_conv]
    x_conv  = (cs_3d * w_conv.unsqueeze(0)).sum(-1)           # [S, d_inner]
    x_conv  = x_conv + self.conv_proj.bias
    xa      = F.silu(x_conv)                                  # [S, d_inner]

    # 3. x_proj: [S, d_inner] → [S, dt_rank + 2*d_state]
    xp     = self.x_proj(xa)
    dt_raw = xp[:, :dr]
    B      = xp[:, dr:dr+ds]                                  # [S, d_state]
    C      = xp[:, dr+ds:]                                    # [S, d_state]

    # 4. dt_proj: [S, dt_rank] → [S, d_inner]
    dt = F.softplus(self.dt_proj(dt_raw))                     # [S, d_inner]

    # 5. SSM state update — THE CRITICAL MISSING PART
    A   = -torch.exp(self.A_log.float())                      # [d_inner, d_state]
    dt3 = dt.unsqueeze(-1)                                    # [S, d_inner, 1]
    dA  = torch.exp(dt3 * A.unsqueeze(0))                     # [S, d_inner, d_state]
    dB  = dt3 * B.unsqueeze(1)                                # [S, d_inner, d_state]
    h   = ssm_state.reshape(S, di, ds)                        # [S, d_inner, d_state]
    new_h  = dA * h + dB * xa.unsqueeze(-1)                  # [S, d_inner, d_state]
    new_ss = new_h.reshape(S, di * ds)

    # 6. SSM readout: contract over d_state dimension
    y_ssm = (new_h * C.unsqueeze(1)).sum(-1)                 # [S, d_inner]

    # 7. D skip + gate
    y = (y_ssm + self.D * xa) * F.silu(z)                   # [S, d_inner]

    # 8. out_proj: [S, d_inner] → [S, d_model]
    out = self.out_proj(y)

    return out, new_cs, new_ss
```

## Expected ONNX nodes after fix

Per block you should now see:
```
Gemm (in_proj)
Slice ×2
Reshape + Mul + ReduceSum (depthwise conv)
SiLU
Gemm (x_proj)
Slice ×3
Gemm (dt_proj)
Softplus
Exp (A)
Mul, Exp (dA)
Mul (dB)
Mul, Add (h update)        ← [S, d_inner, d_state] ops — the missing memory traffic
Mul, ReduceSum (y = h@C)
Mul (D skip)
SiLU, Mul (gate)
Gemm (out_proj)
```

The `[S, d_inner, d_state]` tensors in the SSM update are `8192 × 1536 × 16 = 201M` elements per block — **this is where your 10x memory footprint difference was coming from.**
❌ But root cause is deeper:
You broke recurrence → turned Mamba into parallel model
🔥 One-line takeaway

👉 Mamba is only O(N) if you run it sequentially — your implementation made it O(N × d_state × d_inner × S)
    '''