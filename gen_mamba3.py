
# ─────────────────────────────────────────────────────────────────────────────
# Mamba ONNX export — no Conv1d, only ops your OperationFactory handles
#
# Your OperationFactory supports:
#   Gemm/FusedGemm/MatMul  → GemmWS
#   Mul, Add, Sub, Div     → BinaryOp
#   Exp, Silu/SiLU,
#   Softplus               → UnaryOp
#   Reshape, Transpose,
#   Unsqueeze, Squeeze,
#   Slice, Concat, Gather,
#   Expand, Constant,
#   ConstantOfShape,
#   ReduceSum, Identity    → ShapeOp
#   MambaBlock             → your custom op
#
# Conv1d replacement (no Conv op in factory):
#   conv_state [1, d_inner, d_conv] is a shift-register.
#   new_state  = Concat([conv_state[:,:,1:], x_in.unsqueeze(-1)], dim=2)
#   x_conv     = (new_state[0] * conv_weight).sum(-1) + conv_bias
#              = Mul  →  ReduceSum  →  Add
#   All three ops are in your factory. ✓
#
# Export strategy:
#   • Always S=1. Simulator drives the generation loop.
#   • State shapes: [1, d_inner, d_conv] and [1, d_inner, d_state].
#   • No dynamic axes (S is always 1).
#   • Mapping generated for S=1 GEMM shapes only.
#   • "model_type": "mamba" written to models_list.json so C++ enables state-carry.
# ─────────────────────────────────────────────────────────────────────────────
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import shape_inference, numpy_helper
from collections import Counter
import os
import math
 
# ── Model configurations ──────────────────────────────────────────────────────
MODELS = {
    "mamba-130m": dict(d_model=768,  d_inner=1536, d_state=16, d_conv=4, dt_rank=48,  n_layer=24, vocab_size=50280),
    "mamba-370m": dict(d_model=1024, d_inner=2048, d_state=16, d_conv=4, dt_rank=64,  n_layer=48, vocab_size=50280),
    "mamba-790m": dict(d_model=1536, d_inner=3072, d_state=16, d_conv=4, dt_rank=96,  n_layer=48, vocab_size=50280),
    "mamba-1.4b": dict(d_model=2048, d_inner=4096, d_state=16, d_conv=4, dt_rank=128, n_layer=48, vocab_size=50280),
    "mamba-2.8b": dict(d_model=2560, d_inner=5120, d_state=16, d_conv=4, dt_rank=160, n_layer=64, vocab_size=50280),
}
 
 
class MambaBlockStep(nn.Module):
    """
    Single autoregressive step, S=1.
    Every op reduces to something your OperationFactory already handles.
    NO nn.Conv1d anywhere.
 
    Inputs:
        x           [1, d_model]
        conv_state  [1, d_inner, d_conv]   shift-register (last d_conv tokens)
        ssm_state   [1, d_inner, d_state]  recurrent hidden state h
 
    Outputs:
        out             [1, d_model]
        conv_state_new  [1, d_inner, d_conv]
        ssm_state_new   [1, d_inner, d_state]
    """
    def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv  = d_conv
        self.dt_rank = dt_rank
 
        # All four become Gemm nodes in ONNX
        self.in_proj  = nn.Linear(d_model, 2 * d_inner,           bias=True)
        self.x_proj   = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=True)
        self.dt_proj  = nn.Linear(dt_rank, d_inner,                bias=True)
        self.out_proj = nn.Linear(d_inner, d_model,                bias=True)
 
        # Conv1d weights as plain parameters (NOT nn.Conv1d).
        # Used as: (new_state[0] * conv_weight).sum(-1) + conv_bias
        # → Mul, ReduceSum, Add  — all in BinaryOp / ShapeOp / BinaryOp
        self.conv_weight = nn.Parameter(torch.randn(d_inner, d_conv) * 0.02)
        self.conv_bias   = nn.Parameter(torch.zeros(d_inner))
 
        # SSM parameters
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      .unsqueeze(0).expand(d_inner, -1))   # [d_inner, d_state]
        )
        self.D = nn.Parameter(torch.ones(d_inner))         # [d_inner]
 
    def forward(self, x, conv_state, ssm_state):
        di = self.d_inner
        ds = self.d_state
        dr = self.dt_rank
 
        # ── 1. in_proj → Gemm ────────────────────────────────────────────────
        proj = self.in_proj(x)          # [1, 2*d_inner]
        x_in = proj[:, :di]             # [1, d_inner]
        z    = proj[:, di:]             # [1, d_inner]
 
        # ── 2. Conv1d via shift-register (NO Conv op) ─────────────────────────
        # Concat → Slice (already done by slicing dim=2 from 1:) → Mul → ReduceSum → Add
        x_in_3d   = x_in.unsqueeze(2)                                    # [1, d_inner, 1]
        new_state = torch.cat([conv_state[:, :, 1:], x_in_3d], dim=2)   # [1, d_inner, d_conv]
        #   Concat  → your factory: Concat (ShapeOp)  ✓
        #   new_state[0] * conv_weight → Mul (BinaryOp)  ✓
        #   .sum(-1)  → ReduceSum (ShapeOp)  ✓
        #   + conv_bias → Add (BinaryOp)  ✓
        x_conv = (new_state[0] * self.conv_weight).sum(dim=-1) + self.conv_bias
        x_conv = x_conv.unsqueeze(0)    # [1, d_inner]
 
        # ── 3. SiLU → UnaryOp ────────────────────────────────────────────────
        x_act = F.silu(x_conv)          # [1, d_inner]
 
        # ── 4. x_proj → Gemm ─────────────────────────────────────────────────
        xp     = self.x_proj(x_act)                  # [1, dt_rank+2*d_state]
        dt_raw = xp[:, :dr]                          # [1, dt_rank]
        B      = xp[:, dr:dr + ds]                   # [1, d_state]
        C      = xp[:, dr + ds:]                     # [1, d_state]
 
        # ── 5. dt_proj + Softplus → Gemm + UnaryOp ───────────────────────────
        dt = F.softplus(self.dt_proj(dt_raw))        # [1, d_inner]
 
        # ── 6. SSM step → Exp + Mul + Add ─────────────────────────────────────
        #   A_log stored, A = -exp(A_log) is folded as constant → Mul + Exp
        A   = -torch.exp(self.A_log.float())         # [d_inner, d_state]  (constant)
        h   = ssm_state[0]                           # [d_inner, d_state]
 
        dt1 = dt[0].unsqueeze(-1)                    # [d_inner, 1]
        B1  = B[0].unsqueeze(0)                      # [1, d_state]
        xa1 = x_act[0].unsqueeze(-1)                 # [d_inner, 1]
 
        dA      = torch.exp(dt1 * A)                 # Mul + Exp  ✓
        dB      = dt1 * B1                           # Mul        ✓
        new_h   = dA * h + dB * xa1                  # Mul + Add  ✓
        new_ssm = new_h.unsqueeze(0)                 # [1, d_inner, d_state]
 
        # ── 7. Readout → Mul + ReduceSum ──────────────────────────────────────
        y_ssm = (new_h * C[0].unsqueeze(0)).sum(dim=-1).unsqueeze(0)  # [1, d_inner]
 
        # ── 8. D skip + gate → Mul + Add + SiLU ──────────────────────────────
        y = (y_ssm + self.D * x_act) * F.silu(z)    # [1, d_inner]
 
        # ── 9. out_proj → Gemm ────────────────────────────────────────────────
        out = self.out_proj(y)                       # [1, d_model]
 
        return out, new_state, new_ssm
 
 
class MambaStackStep(nn.Module):
    """N MambaBlockStep blocks chained for a single autoregressive step."""
    def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlockStep(d_model, d_inner, d_state, d_conv, dt_rank)
            for _ in range(n_blocks)
        ])
 
    def forward(self, x, *states):
        # states: conv_0, ssm_0, conv_1, ssm_1, …
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_conv, new_ssm = block(x, states[2 * i], states[2 * i + 1])
            new_states.extend([new_conv, new_ssm])
        return (x, *new_states)
 
 
# ── ONNX post-processing ──────────────────────────────────────────────────────
def fix_gemm_bias_1d(m):
    """Reshape [1, N] bias initialisers to [N] so Gemm nodes are valid."""
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
 
 
# Ops handled by OperationFactory (used to warn on unknown ops)
KNOWN_OPS = {
    "Gemm", "MatMul", "Conv", "MaxPool", "GlobalAveragePool",
    "AdaptiveAveragePool", "AveragePool", "Flatten", "Attention",
    "Cast", "EmbedLayerNormalization", "LayerNormalization",
    "SkipLayerNormalization", "SkipLayerNorm", "BiasGelu", "FastGelu",
    "ReorderOutput", "MambaBlock",
    "Reshape", "Transpose", "Unsqueeze", "Squeeze", "Slice",
    "Expand", "Gather", "Constant", "ConstantOfShape", "Concat",
    "ReduceSum", "Identity",
    "Relu", "Erf", "Tanh", "Sigmoid", "Neg", "Abs", "Sqrt",
    "Exp", "Silu", "SiLU", "Softplus", "Gelu",
    "Add", "Sub", "Mul", "Div", "Equal", "Greater", "Where",
}
 
 
def verify(m, label):
    shapes = {}
    for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for init in m.graph.initializer:
        shapes[init.name] = list(init.dims)
    init_names = {i.name for i in m.graph.initializer}
 
    node_names = [n.name for n in m.graph.node if n.name]
    dupes      = {n for n in node_names if node_names.count(n) > 1}
 
    op_counts = Counter(n.op_type for n in m.graph.node)
    unknown   = {op for op in op_counts if op not in KNOWN_OPS}
    if unknown:
        print(f"  [{label}] ⚠ UNKNOWN OPS (Dummy fallback): {unknown}")
    # Conv must NOT appear — we have no Conv1d in factory
    if "Conv" in op_counts:
        print(f"  [{label}] ✗ ERROR: Conv op found — simulator will mishandle it!")
 
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
 
    return problems, len(dupes), op_counts
 
 
# ── Tile mapping (S=1 GEMM shapes only) ──────────────────────────────────────
def gen_mapping(folder, file_stem, cfg, n_blocks):
    dim       = 128
    spad_kb   = 32768
    accum_kb  = 4096
    precision = 2
    num_cores = 4
 
    def ceil_div(a, b): return (a + b - 1) // b
    def pad_up(x, d):   return ceil_div(x, d) * d
 
    def gemm_mapping(N, C, M):
        max_spad_rows = (spad_kb  * 1024) // (dim * precision * 2)
        max_acc_rows  = (accum_kb * 1024) // (dim * 4         * 2)
        dIp = pad_up(N, dim);  dJp = pad_up(M, dim);  dKp = pad_up(C, dim)
 
        acc  = max_acc_rows // dim
        part = (max_spad_rows // 2) // dim
        sq   = max(1, int(math.sqrt(acc)))
        tk   = max(1, part // sq)
 
        tI = max(1, min(dIp // dim, ceil_div(N, sq * dim)))
        tJ = max(1, min(dJp // dim, ceil_div(M, sq * dim)))
        tK = max(1, min(dKp // dim, ceil_div(C, tk * dim)))
 
        nt = tI * tJ
        if nt < num_cores:
            inc = ceil_div(num_cores, nt)
            if M >= N: tJ *= inc
            else:      tI *= inc
            nt = tI * tJ
        if nt % num_cores:
            r = nt % num_cores
            if M >= N: tJ += r
            else:      tI += r
 
        iI = ceil_div(dIp, tI);  iJ = ceil_div(dJp, tJ);  iK = ceil_div(dKp, tK)
        if iI % dim: iI = max(dim, iI - iI % dim)
        if iJ % dim: iJ = max(dim, iJ - iJ % dim)
        if iK % dim: iK = max(dim, iK - iK % dim)
        iI = min(iI, dIp);  iJ = min(iJ, dJp);  iK = min(iK, dKp)
        tI = ceil_div(N, iI);  tJ = ceil_div(M, iJ);  tK = ceil_div(C, iK)
 
        # Non-zero tile guarantee
        if iI == 0: iI = dIp; tI = 1
        if iJ == 0: iJ = dJp; tJ = 1
        if iK == 0: iK = dKp; tK = 1
        # Sub-systolic-tile dimensions: one full tile
        if N < dim: iI = dIp; tI = 1
        if M < dim: iJ = dJp; tJ = 1
        if C < dim: iK = dKp; tK = 1
 
        return tI, tK, tJ, iI, iK, iJ
 
    d_model = cfg["d_model"];  d_inner = cfg["d_inner"]
    d_state = cfg["d_state"];  dt_rank = cfg["dt_rank"]
 
    # Only GEMM layers. Conv1d → Mul+ReduceSum has no GEMM mapping entry.
    layers = [
        (1, d_model,               2 * d_inner,          "in_proj"),
        (1, d_inner,               dt_rank + 2 * d_state, "x_proj"),
        (1, dt_rank,               d_inner,              "dt_proj"),
        (1, d_inner,               d_model,              "out_proj"),
    ]
 
    seen, lines = set(), []
    for _ in range(n_blocks):
        for N, C, M, label in layers:
            if (N, C, M) not in seen:
                seen.add((N, C, M))
                oN, oC, oM, iN, iC, iM = gemm_mapping(N, C, M)
                lines.append(
                    f"[T] N{N} C{C} M{M} - [O] N{oN} C{oC} M{oM} - [I] N{iN} C{iC} M{iM}"
                    f"  # {label}"
                )
 
    #path = os.path.join(folder, f"{file_stem}.mapping")
    # with open(path, "w") as f:
    #     f.write("\n".join(lines) + "\n")
    # print(f"  Mapping  -> {path}  ({len(lines)} unique GEMM keys)")
 
 
# ── Export one model ──────────────────────────────────────────────────────────
def export_model(model_name, cfg, n_blocks=1):
    d_model = cfg["d_model"];  d_inner = cfg["d_inner"]
    d_state = cfg["d_state"];  d_conv  = cfg["d_conv"];  dt_rank = cfg["dt_rank"]
 
    stem      = f"{model_name}-s1-b{n_blocks}"
    folder    = f"models/{stem}"
    os.makedirs(folder, exist_ok=True)
    onnx_path = os.path.join(folder, f"{stem}.onnx")
 
    if n_blocks == 1:
        model = MambaBlockStep(d_model, d_inner, d_state, d_conv, dt_rank).eval()
        inputs    = (
            torch.zeros(1, d_model),
            torch.zeros(1, d_inner, d_conv),
            torch.zeros(1, d_inner, d_state),
        )
        in_names  = ["x", "conv_state", "ssm_state"]
        out_names = ["out", "conv_state_new", "ssm_state_new"]
    else:
        model     = MambaStackStep(d_model, d_inner, d_state, d_conv, dt_rank, n_blocks).eval()
        state_in  = []
        for i in range(n_blocks):
            state_in.append(torch.zeros(1, d_inner, d_conv))
            state_in.append(torch.zeros(1, d_inner, d_state))
        inputs    = (torch.zeros(1, d_model), *state_in)
        in_names  = ["x"]
        out_names = ["out"]
        for i in range(n_blocks):
            in_names  += [f"conv_state_{i}", f"ssm_state_{i}"]
            out_names += [f"conv_state_new_{i}", f"ssm_state_new_{i}"]
 
    import warnings; warnings.filterwarnings("ignore")
    torch.onnx.export(
        model, inputs,
        f=onnx_path,
        input_names=in_names,
        output_names=out_names,
        opset_version=18,
        dynamo=False,
        do_constant_folding=True,
    )
 
    m = onnx.load(onnx_path)
    m = shape_inference.infer_shapes(m, check_type=True, strict_mode=False)
    m = fix_gemm_bias_1d(m)
    m = shape_inference.infer_shapes(m, check_type=True, strict_mode=False)
    onnx.save(m, onnx_path)
 
    problems, dupes, op_counts = verify(m, stem)
    status = "✓" if problems == 0 and dupes == 0 else f"✗ shapes:{problems} dupes:{dupes}"
    op_str = "  ".join(f"{k}:{v}" for k, v in sorted(op_counts.items()))
 
    print(f"  ONNX     -> {onnx_path}  [{status}]")
    print(f"  Ops      :  {op_str}")
 
    conv_kb = (d_inner * d_conv  * 2) / 1024
    ssm_kb  = (d_inner * d_state * 2) / 1024
    print(f"  State    :  conv={conv_kb:.1f}KB  ssm={ssm_kb:.1f}KB  (float16, per block)")
 
    gen_mapping(folder, stem, cfg, n_blocks)
    return stem, folder
def write_models_list(entries, path="example/models_list.json"):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"models": entries}, f, indent=2)
    print(f"\nmodels_list.json -> {path}  ({len(entries)} models)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # All exports at S=1, varying model size and block count.
    # The original script exported S=1,2,128,256,...,8192 — those larger
    # sequence lengths make no sense for autoregressive ONNX export because:
    #   (a) Mamba has no KV cache; it IS the recurrent state.
    #   (b) The simulator loop handles sequence generation, not the ONNX graph.
    #   (c) Exporting S=8192 inflated state tensors by 8192x and tiling by S.
    EXPORT_VARIANTS = [
        # (model_name,   n_blocks,seq_len)
        ("mamba-130m",   2, 1),   # single block — use for per-layer sim
        ("mamba-130m",   2, 2),   # 2-block stack — matches your original n_blocks=2
         ("mamba-130m",   2, 128),   # single block — use for per-layer sim
        ("mamba-130m",   2, 256),
         ("mamba-130m",   2, 512),   # single block — use for per-layer sim
        ("mamba-130m",   2, 1024),
        ("mamba-130m",   2, 2048)
    ]

    entries = []
    for model_name, n_blocks, seq_len in EXPORT_VARIANTS:
        cfg = MODELS[model_name]
        print(f"\n[{model_name}  n_blocks={n_blocks}]")
        stem, folder = export_model(model_name, cfg, n_blocks=n_blocks)
        entries.append({
            "name":            stem,
            "batch_size":      1,
            "seq_len":         1,          # always 1 — single autoregressive step
            "past_seq_len":    0,
            "total_seq_len":   0,
            "output_seq_len":  seq_len,   # total tokens to generate
        })

    write_models_list(entries)
    print("\nDone.")