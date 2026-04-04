# ─────────────────────────────────────────────────────────────────────────────
# Mamba ONNX export — architecturally correct per paper Algorithm 2
#  + HuggingFace modeling_mamba.py (document 2)
#  + Custom MambaBlock (document 1)
#  + Diagram (image): λ→A_t→diag, Δ_t, B_t, C_t, Conv1D, Linear, h_t SSM recurrence
#
# KEY FIXES vs original gen_mamba.py:
#  1. MambaBlock now faithfully follows the real architecture:
#       in_proj  : d_model → 2*d_inner  (no bias, like HF)
#       conv1d   : true depthwise Conv1d(d_inner, kernel=d_conv) — NOT a flat linear
#       x_proj   : d_inner → dt_rank + 2*d_state  (no bias)
#       dt_proj  : dt_rank → d_inner  (WITH bias)
#       A_log    : [d_inner, d_state]  (log of negative eigenvalues)
#       D        : [d_inner]  (skip connection)
#       out_proj : d_inner → d_model  (no bias)
#  2. SSM recurrence uses ZOH discretization (exact, as in diagram + paper):
#       Ā = exp(Δ ⊗ A)
#       B̄ = (Ā - 1) / A  * B   (ZOH B-bar)
#       h = Ā * h + B̄ * x
#       y = h @ C
#  3. Gate applied AFTER SSM: y = (y + D*x) * silu(z)
#  4. Parameter count for mamba-130m is calculated and printed exactly.
# ─────────────────────────────────────────────────────────────────────────────

# import os
# import math
# import json
# from collections import Counter

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import onnx
# from onnx import shape_inference, numpy_helper


# # ─── Model configs ────────────────────────────────────────────────────────────
# MODELS = {
#     # d_model, d_inner(=expand*d_model), d_state, d_conv, dt_rank, n_layer, vocab_size
#     "mamba-130m": dict(d_model=768,  d_inner=1536, d_state=16, d_conv=4, dt_rank=48,  n_layer=24, vocab_size=50280),
#     "mamba-370m": dict(d_model=1024, d_inner=2048, d_state=16, d_conv=4, dt_rank=64,  n_layer=48, vocab_size=50280),
#     "mamba-790m": dict(d_model=1536, d_inner=3072, d_state=16, d_conv=4, dt_rank=96,  n_layer=48, vocab_size=50280),
#     "mamba-1.4b": dict(d_model=2048, d_inner=4096, d_state=16, d_conv=4, dt_rank=128, n_layer=48, vocab_size=50280),
#     "mamba-2.8b": dict(d_model=2560, d_inner=5120, d_state=16, d_conv=4, dt_rank=160, n_layer=64, vocab_size=50280),
# }


# # ─── Parameter counter ────────────────────────────────────────────────────────
def count_mamba_130m_params(cfg):
    """
    Exact parameter breakdown for mamba-130m following the real architecture.

    Weight shapes per MambaMixer (from HF modeling_mamba.py + paper):
      in_proj.weight    : [2*d_inner, d_model]         no bias
      conv1d.weight     : [d_inner, 1, d_conv]         depthwise
      conv1d.bias       : [d_inner]
      x_proj.weight     : [dt_rank + 2*d_state, d_inner]  no bias
      dt_proj.weight    : [d_inner, dt_rank]           WITH bias
      dt_proj.bias      : [d_inner]
      A_log             : [d_inner, d_state]           (parameter, not a weight matrix)
      D                 : [d_inner]
      out_proj.weight   : [d_model, d_inner]           no bias

    Non-mixer params:
      embedding         : [vocab_size, d_model]
      norm per layer    : [d_model]  × n_layer   (RMSNorm, weight only)
      final norm_f      : [d_model]
      lm_head           : tied to embedding → 0 extra params
    """
    d  = cfg["d_model"]
    di = cfg["d_inner"]
    ds = cfg["d_state"]
    dc = cfg["d_conv"]
    dr = cfg["dt_rank"]
    L  = cfg["n_layer"]
    V  = cfg["vocab_size"]

    # ── per-layer MambaMixer ──────────────────────────────────────────────────
    in_proj      = 2 * di * d          # [2*d_inner, d_model]
    conv1d_w     = di * 1 * dc         # [d_inner, 1, d_conv]  depthwise
    conv1d_b     = di                  # bias
    x_proj       = (dr + 2*ds) * di   # [dt_rank + 2*d_state, d_inner]
    dt_proj_w    = di * dr             # [d_inner, dt_rank]
    dt_proj_b    = di                  # bias
    A_log        = di * ds             # [d_inner, d_state]
    D            = di                  # skip scalar per channel
    out_proj     = d * di              # [d_model, d_inner]

    per_mixer = in_proj + conv1d_w + conv1d_b + x_proj + dt_proj_w + dt_proj_b + A_log + D + out_proj

    # ── per-layer RMSNorm (weight only, no bias) ──────────────────────────────
    per_norm = d

    per_layer = per_mixer + per_norm

    # ── non-layer params ──────────────────────────────────────────────────────
    embedding  = V * d   # lm_head is tied → count once
    final_norm = d

    total = L * per_layer + embedding + final_norm

    print("=" * 64)
    print(f"  Mamba-130M exact parameter count")
    print(f"  d_model={d}  d_inner={di}  d_state={ds}  d_conv={dc}")
    print(f"  dt_rank={dr}  n_layer={L}  vocab_size={V}")
    print("=" * 64)
    print(f"  Per-MambaMixer breakdown:")
    print(f"    in_proj.weight      : {in_proj:>12,}   [{2*di} × {d}]")
    print(f"    conv1d.weight       : {conv1d_w:>12,}   [{di} × 1 × {dc}]  depthwise")
    print(f"    conv1d.bias         : {conv1d_b:>12,}   [{di}]")
    print(f"    x_proj.weight       : {x_proj:>12,}   [{dr+2*ds} × {di}]")
    print(f"    dt_proj.weight      : {dt_proj_w:>12,}   [{di} × {dr}]")
    print(f"    dt_proj.bias        : {dt_proj_b:>12,}   [{di}]")
    print(f"    A_log               : {A_log:>12,}   [{di} × {ds}]")
    print(f"    D (skip)            : {D:>12,}   [{di}]")
    print(f"    out_proj.weight     : {out_proj:>12,}   [{d} × {di}]")
    print(f"    ─────────────────────────────────")
    print(f"    subtotal per mixer  : {per_mixer:>12,}")
    print(f"    RMSNorm weight      : {per_norm:>12,}")
    print(f"    total per layer     : {per_layer:>12,}")
    print(f"")
    print(f"  Global params:")
    print(f"    {L} layers           : {L*per_layer:>12,}")
    print(f"    embedding (=lm_head): {embedding:>12,}   [{V} × {d}]  (tied)")
    print(f"    final RMSNorm       : {final_norm:>12,}")
    print(f"    ─────────────────────────────────")
    print(f"    TOTAL               : {total:>12,}  ({total/1e6:.2f} M)")
    print("=" * 64)
    return total


# # ─── Architecturally correct MambaBlock for ONNX export ───────────────────────
# class MambaBlock(nn.Module):
#     """
#     Single Mamba mixer block (inference / step mode, one token at a time).

#     Follows exactly:
#       - Paper Algorithm 2 (ZOH discretization)
#       - HF modeling_mamba.py  (slow_forward path)
#       - Custom impl (document 1) eigenvalue / recurrence structure
#       - Diagram:  λ→A_t, Δ_t→Ā, B_t→B̄, h_{t-1}→h_t=Āh+B̄x, C_t·h_t→y

#     Inputs  (all [1, *]):
#       x              : [1, d_model]
#       conv_state     : [1, d_inner * d_conv]   shift-register, oldest first
#       ssm_state      : [1, d_inner * d_state]

#     Outputs:
#       out            : [1, d_model]
#       conv_state_new : [1, d_inner * d_conv]
#       ssm_state_new  : [1, d_inner * d_state]
#     """

#     def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank):
#         super().__init__()
#         self.d_model = d_model
#         self.d_inner = d_inner
#         self.d_state = d_state
#         self.d_conv  = d_conv
#         self.dt_rank = dt_rank

#         # 1. Input projection: d_model → x_branch + gate
#         self.in_proj  = nn.Linear(d_model, 2 * d_inner, bias=False)

#         # 2. Depthwise conv (stored as weight [d_inner, d_conv] for step-mode)
#         #    In full training mode this is Conv1d; here we keep it as a parameter
#         #    so the ONNX graph is a simple matmul-equivalent for hardware mapping.
#         #    conv_state is the shift register [1, d_inner*d_conv], newest on right.
#         self.conv1d_weight = nn.Parameter(torch.empty(d_inner, d_conv))  # [di, dc]
#         self.conv1d_bias   = nn.Parameter(torch.zeros(d_inner))

#         # 3. Selective projection: d_inner → dt_raw, B, C
#         self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

#         # 4. dt projection: dt_rank → d_inner  (WITH bias, as in paper)
#         self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

#         # 5. SSM eigenvalues: A_log = log(-A), shape [d_inner, d_state]
#         #    Initialized as in HF: A[i,j] = j+1  (1-indexed), A_log = log(A)
#         A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0) \
#                   .expand(d_inner, -1).contiguous()                # [di, ds]
#         self.A_log = nn.Parameter(torch.log(A))                    # log(-A), A<0

#         # 6. Skip-connection scalar per channel
#         self.D = nn.Parameter(torch.ones(d_inner))

#         # 7. Output projection: d_inner → d_model
#         self.out_proj = nn.Linear(d_inner, d_model, bias=False)

#         self._init_weights()

#     def _init_weights(self):
#         nn.init.kaiming_uniform_(self.in_proj.weight,  a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.conv1d_weight,   a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.x_proj.weight,   a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))

#         # dt_proj bias init: inverse-softplus of uniform random dt in [dt_min, dt_max]
#         dt_min, dt_max = 0.001, 0.1
#         dt = torch.exp(
#             torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=1e-4)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))   # softplus^{-1}(dt)
#         with torch.no_grad():
#             self.dt_proj.bias.copy_(inv_dt)

#     def forward(self, x, conv_state, ssm_state):
#         di, ds, dc = self.d_inner, self.d_state, self.d_conv

#         # ── 1. projection ─────────────────────────
#         proj = self.in_proj(x)          # [2*di]
#         x_in = proj[:di]                # [di]
#         z    = proj[di:]                # [di]

#         # ── 2. conv (shift register, FLAT) ────────
#         conv_new = torch.cat([conv_state[di:], x_in])   # [di*dc]

#         x_conv = torch.matmul(conv_new, self.conv_weight) + self.conv_bias  # [di]
#         x_act  = F.silu(x_conv)

#         # ── 3. selective projection ──────────────
#         ssm = self.x_proj(x_act)        # [dt_rank + 2*ds]

#         dt_raw = ssm[:self.dt_rank]
#         Bt     = ssm[self.dt_rank:self.dt_rank + ds]
#         Ct     = ssm[self.dt_rank + ds:]

#         dt = F.softplus(self.dt_proj(dt_raw))   # [di]

#         # ── 4. SSM (fully flat) ────────────────
#         dt_exp = dt.repeat_interleave(ds)       # [di*ds]

#         A = -torch.exp(self.A_log)              # [di*ds]

#         dA = torch.exp(dt_exp * A)              # [di*ds]

#         Bt_exp = Bt.repeat_interleave(di)       # [di*ds]

#         dB = (dA - 1.0) / (A + 1e-5) * Bt_exp

#         x_rep = x_act.repeat_interleave(ds)

#         new_h = dA * ssm_state + dB * x_rep     # [di*ds]

#         # ── 5. output ─────────────────────────
#         Ct_exp = Ct.repeat_interleave(di)

#         y = new_h * Ct_exp
#         y = y.view(di, ds).sum(dim=1)           # [di]

#         y = (y + self.D * x_act) * F.silu(z)

#         out = self.out_proj(y)                  # [d_model]

#         return out, conv_new, new_h


# # ─── Multi-block stack for ONNX export ────────────────────────────────────────
# class MambaStack(nn.Module):
#     def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank, n_blocks):
#         super().__init__()
#         self.n_blocks = n_blocks
#         self.blocks = nn.ModuleList([
#             MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank)
#             for _ in range(n_blocks)
#         ])

#     def forward(self, x, *states):
#         """
#         x      : [1, d_model]
#         states : 2*n_blocks tensors alternating (conv_state_i, ssm_state_i)
#         """
#         for i, block in enumerate(self.blocks):
#             x, _, _ = block(x, states[2*i], states[2*i+1])
#         return x


# # ─── ONNX helpers ─────────────────────────────────────────────────────────────
# def fix_gemm_bias_1d(m):
#     gemm_bias_names = {
#         node.input[2]
#         for node in m.graph.node
#         if node.op_type == "Gemm" and len(node.input) == 3
#     }
#     for init in list(m.graph.initializer):
#         if init.name not in gemm_bias_names:
#             continue
#         dims = list(init.dims)
#         if len(dims) == 2 and dims[0] == 1:
#             arr      = numpy_helper.to_array(init).reshape(-1)
#             new_init = numpy_helper.from_array(arr, name=init.name)
#             m.graph.initializer.remove(init)
#             m.graph.initializer.append(new_init)
#     return m


# def verify(m, label):
#     shapes = {}
#     for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
#         shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
#     for init in m.graph.initializer:
#         shapes[init.name] = list(init.dims)
#     init_names = {i.name for i in m.graph.initializer}

#     node_names = [n.name for n in m.graph.node if n.name]
#     dupes = {n for n in node_names if node_names.count(n) > 1}

#     problems = 0
#     for node in m.graph.node:
#         if node.op_type != "Gemm":
#             continue
#         for i, inp in enumerate(node.input):
#             s   = shapes.get(inp, "UNKNOWN")
#             bad = (s == "UNKNOWN" or
#                    (isinstance(s, list) and (len(s) < 1 or 0 in s)) or
#                    (i == 1 and inp not in init_names))
#             if bad:
#                 problems += 1
#     return problems, len(dupes)


# # # ─── Mapping file generator (unchanged logic, correct layer shapes) ────────────
# # def gen_mapping(folder, file_stem, cfg, seq_len, n_blocks):
# #     dim, spad_kb, accum_kb, precision, num_cores = 128, 32768, 4096, 2, 4

# #     def ceil_div(a, b): return (a + b - 1) // b
# #     def pad_up(x, d):   return ceil_div(x, d) * d

# #     def gemm_mapping(N, C, M):
# #         max_spad_rows = (spad_kb  * 1024) // (dim * precision * 2)
# #         max_acc_rows  = (accum_kb * 1024) // (dim * 4        * 2)

# #         dI, dJ, dK = N, M, C
# #         dIp = pad_up(dI, dim); dJp = pad_up(dJ, dim); dKp = pad_up(dK, dim)

# #         acc  = max_acc_rows // dim
# #         part = (max_spad_rows // 2) // dim
# #         sq   = max(1, int(math.sqrt(acc)))
# #         tk   = max(1, part // sq)

# #         tI = max(1, min(dIp // dim, ceil_div(dI, sq * dim)))
# #         tJ = max(1, min(dJp // dim, ceil_div(dJ, sq * dim)))
# #         tK = max(1, min(dKp // dim, ceil_div(dK, tk * dim)))

# #         nt = tI * tJ
# #         if nt < num_cores:
# #             inc = ceil_div(num_cores, nt)
# #             tJ = tJ * inc if dJ >= dI else tI * inc
# #             if dJ >= dI: tJ *= inc
# #             else:        tI *= inc
# #             nt = tI * tJ
# #         if nt % num_cores:
# #             r = nt % num_cores
# #             if dJ >= dI: tJ += r
# #             else:        tI += r

# #         iI = ceil_div(dIp, tI); iJ = ceil_div(dJp, tJ); iK = ceil_div(dKp, tK)
# #         if iI % dim: iI = max(dim, iI - iI % dim)
# #         if iJ % dim: iJ = max(dim, iJ - iJ % dim)
# #         if iK % dim: iK = max(dim, iK - iK % dim)
# #         iI = min(iI, dIp); iJ = min(iJ, dJp); iK = min(iK, dKp)

# #         tI = ceil_div(dI, iI); tJ = ceil_div(dJ, iJ); tK = ceil_div(dK, iK)

# #         for val, padded, ti in [(iI, dIp, tI), (iJ, dJp, tJ), (iK, dKp, tK)]:
# #             pass  # safe check below
# #         if iI == 0: iI = dIp; tI = 1
# #         if iJ == 0: iJ = dJp; tJ = 1
# #         if iK == 0: iK = dKp; tK = 1
# #         if dI < dim: iI = dIp; tI = 1
# #         if dJ < dim: iJ = dJp; tJ = 1
# #         if dK < dim: iK = dKp; tK = 1

# #         return tI, tK, tJ, iI, iK, iJ

# #     d  = cfg["d_model"]; di = cfg["d_inner"]
# #     ds = cfg["d_state"]; dc = cfg["d_conv"]; dr = cfg["dt_rank"]

# #     # Correct GEMM shapes matching the fixed MambaBlock
# #     layers = [
# #         (seq_len, d,          2 * di         ),   # in_proj
# #         (seq_len, di,         dr + 2*ds      ),   # x_proj
# #         (seq_len, dr,         di             ),   # dt_proj
# #         (seq_len, di,         d              ),   # out_proj
# #     ]

# #     seen, lines = set(), []
# #     for _ in range(n_blocks):
# #         for N, C, M in layers:
# #             if (N, C, M) not in seen:
# #                 seen.add((N, C, M))
# #                 oN, oC, oM, iN, iC, iM = gemm_mapping(N, C, M)
# #                 lines.append(
# #                     f"[T] N{N} C{C} M{M} - [O] N{oN} C{oC} M{oM} - [I] N{iN} C{iC} M{iM}"
# #                 )

# #     path = os.path.join(folder, f"{file_stem}.mapping")
# #     with open(path, "w") as f:
# #         f.write("\n".join(lines) + "\n")
# #     print(f"  Mapping  -> {path}  ({len(lines)} unique keys)")
# def gen_mapping(folder, file_stem, cfg, seq_len, n_blocks):
#     dim, spad_kb, accum_kb, precision, num_cores = 128, 32768, 4096, 2, 4

#     def ceil_div(a, b): return (a + b - 1) // b
#     def pad_up(x, d):   return ceil_div(x, d) * d

#     def gemm_mapping(N, C, M):
#         max_spad_rows = (spad_kb  * 1024) // (dim * precision * 2)
#         max_acc_rows  = (accum_kb * 1024) // (dim * 4        * 2)

#         dI, dJ, dK = N, M, C
#         dIp = pad_up(dI, dim); dJp = pad_up(dJ, dim); dKp = pad_up(dK, dim)

#         acc  = max_acc_rows // dim
#         part = (max_spad_rows // 2) // dim
#         sq   = max(1, int(math.sqrt(acc)))
#         tk   = max(1, part // sq)

#         tI = max(1, min(dIp // dim, ceil_div(dI, sq * dim)))
#         tJ = max(1, min(dJp // dim, ceil_div(dJ, sq * dim)))
#         tK = max(1, min(dKp // dim, ceil_div(dK, tk * dim)))

#         nt = tI * tJ
#         if nt < num_cores:
#             inc = ceil_div(num_cores, nt)
#             if dJ >= dI: tJ *= inc
#             else:        tI *= inc
#             nt = tI * tJ

#         if nt % num_cores:
#             r = nt % num_cores
#             if dJ >= dI: tJ += r
#             else:        tI += r

#         iI = ceil_div(dIp, tI); iJ = ceil_div(dJp, tJ); iK = ceil_div(dKp, tK)

#         if iI % dim: iI = max(dim, iI - iI % dim)
#         if iJ % dim: iJ = max(dim, iJ - iJ % dim)
#         if iK % dim: iK = max(dim, iK - iK % dim)

#         iI = min(iI, dIp); iJ = min(iJ, dJp); iK = min(iK, dKp)

#         tI = ceil_div(dI, iI); tJ = ceil_div(dJ, iJ); tK = ceil_div(dK, iK)

#         if iI == 0: iI = dIp; tI = 1
#         if iJ == 0: iJ = dJp; tJ = 1
#         if iK == 0: iK = dKp; tK = 1

#         if dI < dim: iI = dIp; tI = 1
#         if dJ < dim: iJ = dJp; tJ = 1
#         if dK < dim: iK = dKp; tK = 1

#         return tI, tK, tJ, iI, iK, iJ

#     d  = cfg["d_model"]
#     di = cfg["d_inner"]
#     ds = cfg["d_state"]
#     dc = cfg["d_conv"]
#     dr = cfg["dt_rank"]

#     # GEMM layers
#     layers = [
#         ( d,          2 * di),
#         ( di,         dr + 2*ds),
#         ( dr,         di),
#         ( di,         d),
#     ]

#     seen, lines = set(), []

#     # GEMM entries
#     for _ in range(n_blocks):
#         for N, C, M in layers:
#             if (N, C, M) not in seen:
#                 seen.add((N, C, M))
#                 oN, oC, oM, iN, iC, iM = gemm_mapping(N, C, M)
#                 lines.append(
#                     f"[T] N{N} C{C} M{M} - [O] N{oN} C{oC} M{oM} - [I] N{iN} C{iC} M{iM}"
#                 )

#     N = 1
#     C = di          # input channels
#     M = di          # depthwise → same as input
#     S = 1           # height kernel
#     R = dc          # width kernel (Conv1D)
#     Q = 1           # height
#     P = seq_len     # width (sequence)

#     lines.append(
#         f"[T] N{N} C{C} M{M} S{S} R{R} Q{Q} P{P} - "
#         f"[O] N{N} C1 M{M} P{P} Q{Q} S1 R1 - "
#         f"[I] N{N} C{C} M1 P{P} Q{Q} S{S} R{R}"
#     )

#     path = os.path.join(folder, f"{file_stem}.mapping")
#     with open(path, "w") as f:
#         f.write("\n".join(lines) + "\n")
# import onnx
# from onnx import shape_inference, numpy_helper

# def flatten_to_2d(shape):
#     if len(shape) <= 2:
#         return shape
#     N = 1
#     for d in shape[:-1]:
#         N *= d
#     return [N, shape[-1]]

# def fix_gemm_to_2d(model):
#     graph = model.graph
#     value_info = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}

#     def get_shape(name):
#         if name not in value_info:
#             return None
#         return [d.dim_value for d in value_info[name].type.tensor_type.shape.dim]

#     for node in graph.node:
#         if node.op_type != "Gemm":
#             continue

#         A, B = node.input[0], node.input[1]

#         shape_A = get_shape(A)
#         shape_B = get_shape(B)
#         print(f"Checking Gemm node '{node.name}': A={shape_A}, B={shape_B}")

#         if shape_A is None or shape_B is None:
#             continue
#         from onnx import helper, TensorProto

# def flatten_to_2d(shape):
#     if len(shape) <= 2:
#         return shape
#     N = 1
#     for d in shape[:-1]:
#         N *= d
#     return [N, shape[-1]]

# def get_shape_map(graph):
#     shape_map = {}

#     def extract(vi):
#         return [d.dim_value for d in vi.type.tensor_type.shape.dim]

#     for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
#         shape_map[vi.name] = extract(vi)

#     for init in graph.initializer:
#         shape_map[init.name] = list(init.dims)

#     return shape_map


# def insert_reshape(graph, input_name, new_shape, tag):
#     shape_name = input_name + "_shape_" + tag
#     output_name = input_name + "_reshaped_" + tag

#     shape_tensor = helper.make_tensor(
#         name=shape_name,
#         data_type=TensorProto.INT64,
#         dims=[len(new_shape)],
#         vals=new_shape
#     )

#     reshape_node = helper.make_node(
#         "Reshape",
#         inputs=[input_name, shape_name],
#         outputs=[output_name],
#         name=input_name + "_reshape_" + tag
#     )

#     graph.initializer.append(shape_tensor)
#     graph.node.append(reshape_node)

#     return output_name


# def fix_gemm_to_2d(model):
#     graph = model.graph
#     shape_map = get_shape_map(graph)
#     init_names = {i.name for i in graph.initializer}

#     for node in graph.node:
#         if node.op_type != "Gemm":
#             continue

#         A, B = node.input[0], node.input[1]

#         shape_A = shape_map.get(A)
#         shape_B = shape_map.get(B)

#         print(f"[BEFORE] {node.name}: A={shape_A}, B={shape_B}")

#         # ---- FIX A ----
#         if shape_A is not None:
#             new_A_shape = flatten_to_2d(shape_A)
#             if shape_A != new_A_shape:
#                 new_A = insert_reshape(graph, A, new_A_shape, "A")
#                 node.input[0] = new_A
#                 print(f"  A FIXED: {shape_A} → {new_A_shape}")

#         # ---- FIX B ----
#         if B not in init_names:
#             print(f"  ⚠️ WARNING: {node.name} B is NOT initializer")

#         if shape_B is not None:
#             new_B_shape = flatten_to_2d(shape_B)
#             if shape_B != new_B_shape:
#                 new_B = insert_reshape(graph, B, new_B_shape, "B")
#                 node.input[1] = new_B
#                 print(f"  B FIXED: {shape_B} → {new_B_shape}")

#     return model
# def build_shape_dict(m):
#     shape_dict = {}

#     # inputs / outputs / intermediates
#     for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
#         dims = []
#         for d in vi.type.tensor_type.shape.dim:
#             dims.append(d.dim_value)
#         shape_dict[vi.name] = dims

#     # initializers (weights)
#     for init in m.graph.initializer:
#         shape_dict[init.name] = list(init.dims)

#     return shape_dict
# import onnx
# from onnx import helper, numpy_helper

# def replace_gemm_with_matmul(model):
#     new_nodes = []
#     new_initializers = list(model.graph.initializer)

#     for node in model.graph.node:
#         if node.op_type == "Gemm":
#             A = node.input[0]
#             W = node.input[1]
#             B = node.input[2] if len(node.input) > 2 else None
#             out = node.output[0]

#             # Create transpose node for weights
#             W_T_name = W + "_T"
#             transpose_node = helper.make_node(
#                 "Transpose",
#                 inputs=[W],
#                 outputs=[W_T_name],
#                 name=node.name + "_transpose",
#                 perm=[1, 0]
#             )

#             # MatMul
#             matmul_out = out + "_matmul"
#             matmul_node = helper.make_node(
#                 "MatMul",
#                 inputs=[A, W_T_name],
#                 outputs=[matmul_out],
#                 name=node.name + "_matmul"
#             )

#             new_nodes.append(transpose_node)
#             new_nodes.append(matmul_node)

#             # Add bias if exists
#             if B:
#                 add_node = helper.make_node(
#                     "Add",
#                     inputs=[matmul_out, B],
#                     outputs=[out],
#                     name=node.name + "_add"
#                 )
#                 new_nodes.append(add_node)
#             else:
#                 # no bias case
#                 identity_node = helper.make_node(
#                     "Identity",
#                     inputs=[matmul_out],
#                     outputs=[out],
#                     name=node.name + "_identity"
#                 )
#                 new_nodes.append(identity_node)

#         else:
#             new_nodes.append(node)

#     model.graph.ClearField("node")
#     model.graph.node.extend(new_nodes)

#     return model
# def fix_zero_dims(model):
#     # Fix ValueInfo (tensor shapes)
#     for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
#         if value.type.tensor_type.HasField("shape"):
#             for dim in value.type.tensor_type.shape.dim:
#                 if dim.dim_value == 0:
#                     dim.dim_value = 1

#     # Fix initializers if needed
#     for init in model.graph.initializer:
#         shape = list(init.dims)
#         new_shape = [1 if d == 0 else d for d in shape]
#         if shape != new_shape:
#             arr = numpy_helper.to_array(init)
#             arr = arr.reshape(new_shape)
#             new_init = numpy_helper.from_array(arr, init.name)
#             model.graph.initializer.remove(init)
#             model.graph.initializer.append(new_init)

#     return model

# # ─── Export ───────────────────────────────────────────────────────────────────
# def export_model(model_name, cfg, seq_len=1, n_blocks=1):
#     d_model = cfg["d_model"]; d_inner = cfg["d_inner"]
#     d_state = cfg["d_state"]; d_conv  = cfg["d_conv"]; dt_rank = cfg["dt_rank"]

#     stem   = f"{model_name}-b{n_blocks}-s{seq_len}"
#     folder = f"models/{stem}"
#     os.makedirs(folder, exist_ok=True)
#     onnx_path = os.path.join(folder, f"{stem}.onnx")

#     if n_blocks == 1:
#         model  = MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank).eval()
#         inputs = (
#             torch.zeros( d_model),
#             torch.zeros( d_inner * d_conv),
#             torch.zeros(d_inner * d_state),
#         )
#         in_names  = ["x", "conv_state_flat", "ssm_state_flat"]
#         out_names = ["out", "conv_state_new", "ssm_state_new"]
#     else:
#         model = MambaStack(d_model, d_inner, d_state, d_conv, dt_rank, n_blocks).eval()
#         state_inputs = []
#         for i in range(n_blocks):
#             state_inputs.append(torch.zeros( d_inner * d_conv))
#             state_inputs.append(torch.zeros( d_inner * d_state))
#         inputs   = (torch.zeros(1, d_model), *state_inputs)
#         in_names = ["x"] + [
#             f"conv_state_{i}" if j % 2 == 0 else f"ssm_state_{j // 2}"
#             for j, i in enumerate(range(2 * n_blocks))
#         ]
#         in_names = ["x"]
#         for i in range(n_blocks):
#             in_names += [f"conv_state_{i}", f"ssm_state_{i}"]
#         out_names = ["out"]

#     dynamic_axes = {}
#     if n_blocks == 1:
#         dynamic_axes.update({
#             "conv_state_flat": {0: "batch"},
#             "ssm_state_flat":  {0: "batch"},
#             "conv_state_new":  {0: "batch"},
#             "ssm_state_new":   {0: "batch"},
#         })
#     else:
#         for i in range(n_blocks):
#             dynamic_axes[f"conv_state_{i}"] = {0: "batch"}
#             dynamic_axes[f"ssm_state_{i}"]  = {0: "batch"}

#     import warnings; warnings.filterwarnings("ignore")
#     torch.onnx.export(
#         model, inputs, f=onnx_path,
#         input_names=in_names, output_names=out_names,
#         opset_version=18, dynamo=False,
#         dynamic_axes=dynamic_axes,
#     )

#     m = onnx.load(onnx_path)
#     m = shape_inference.infer_shapes(m, check_type=True, strict_mode=True)
#     m = fix_gemm_bias_1d(m)
#     m = fix_gemm_to_2d(m)
#     m = fix_zero_dims(m)
#     m = replace_gemm_with_matmul(m)
#     m = shape_inference.infer_shapes(m, check_type=True, strict_mode=True)
#     shape_dict = build_shape_dict(m)

#     print("\n=== NODE SHAPES ===")
#     for node in m.graph.node:
#         print(f"\n{node.op_type}  ({node.name})")

#         for inp in node.input:
#             print(f"  IN  {inp:40s} -> {shape_dict.get(inp, 'UNKNOWN')}")

#         for out in node.output:
#             print(f"  OUT {out:40s} -> {shape_dict.get(out, 'UNKNOWN')}")
#     print("\n=== FINAL GEMM CHECK ===")
#     onnx.save(m, onnx_path)

#     problems, dupes = verify(m, stem)
#     counts = Counter(n.op_type for n in m.graph.node)
#     status = "✓" if problems == 0 and dupes == 0 else f"✗ shapes:{problems} dupes:{dupes}"
#     print(
#         f"  ONNX     -> {onnx_path}  [{status}]"
#         f"  Gemm:{counts.get('MatMul', 0)}  nodes:{len(m.graph.node)}"
#     )
#     gen_mapping(folder, stem, cfg, seq_len, n_blocks)
#     return stem, folder


# def write_models_list(entries, path="example/models_list.json"):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w") as f:
#         json.dump({"models": entries}, f, indent=2)
#     print(f"\nmodels_list.json -> {path}  ({len(entries)} models)")


# # ─── Main ─────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":

#     # ── 1. Print exact parameter count for mamba-130m ─────────────────────────
#     #total = count_mamba_130m_params(MODELS["mamba-130m"])

#     # ── 2. Export ONNX variants ───────────────────────────────────────────────
#     EXPORT_VARIANTS = [
#         ("mamba-130m", 1,    2),
#         # ("mamba-130m", 2,    2),
#         # ("mamba-130m", 128,  2),
#         # ("mamba-130m", 256,  2),
#         # ("mamba-130m", 512,  2),
#         # ("mamba-130m", 1024, 2),
#         # ("mamba-130m", 2048, 2),
#     ]

#     entries = []
#     for model_name, seq_len, n_blocks in EXPORT_VARIANTS:
#         cfg = MODELS[model_name]
#         print(f"\n[{model_name}  seq_len={seq_len}  n_blocks={n_blocks}]")
#         stem, folder = export_model(model_name, cfg, seq_len=seq_len, n_blocks=n_blocks)
#         entries.append({
#             "name":          stem,
#             "batch_size":    1,
#             "seq_len":       1,
#             "past_seq_len":  0,
#             "total_seq_len": 0,
#             "output_seq_len": seq_len,
#         })

#     write_models_list(entries)
#     print("\nDone.")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS = {
    "mamba-tiny": dict(
     d_model=256,
    d_inner=256,    
    d_state=8,
    d_conv=2,
    dt_rank=16,
    n_layer=1,
    vocab_size=10000
),
    "mamba-130m": dict(
        d_model=768,
        d_inner=1536,
        d_state=16,
        d_conv=4,
        dt_rank=48,
        n_layer=24
    )
}

# ─────────────────────────────────────────────
# CLEAN MAMBA BLOCK (PURE 2D)
# ─────────────────────────────────────────────
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank):
        super().__init__()
        self.di, self.ds, self.dr, self.dc = d_inner, d_state, dt_rank, d_conv

        self.in_proj   = nn.Linear(d_model,              2*d_inner,          bias=False)
        self.conv_proj = nn.Linear(d_inner*d_conv,        d_inner,            bias=False)
        # self.x_proj    = nn.Linear(d_inner,               dt_rank+2*d_inner*d_state, bias=False)
        # self.dt_proj   = nn.Linear(dt_rank,               d_inner*d_state,   bias=True)
        # self.ssm_out   = nn.Linear(d_inner*d_state,       d_inner,            bias=False)
        self.out_proj  = nn.Linear(d_inner,               d_model,            bias=False)

        self.D = nn.Parameter(torch.ones(1, d_inner))
        # self.A_log = nn.Parameter(
        # torch.log(torch.arange(1, d_state+1, dtype=torch.float32)
        #           .repeat(d_inner, 1)
        #           .reshape(1, d_inner*d_state))     )     # [1, di*ds] ✓)
        # __init__ — restore small x_proj and ssm_out
        self.x_proj = nn.Linear(d_inner, dt_rank + 2*d_state, bias=False)  # M=80, back to ref
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)               # M=1536, back to ref
        self.ssm_out = nn.Linear(d_state, d_inner, bias=False)              # M=1536, tiny

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state+1, dtype=torch.float32)
                    .unsqueeze(0).expand(d_inner, -1)
                    .reshape(1, d_inner*d_state))                          # [1, di*ds]
        )
            
    def forward(self, x, conv_state, ssm_state):
        # 1. in_proj
        proj = self.in_proj(x)
        x_in = proj[:, :self.di]
        z    = proj[:, self.di:]

        # 2. conv
        conv_new = torch.cat([conv_state[:, self.di:], x_in], dim=1)
        x_act    = F.silu(self.conv_proj(conv_new))         # [1, di]

        # 3. x_proj — small output M=80, same as ref
        ssm  = self.x_proj(x_act)                          # [1, dr+2*ds]
        dt_r = ssm[:, :self.dr]                            # [1, dr]
        Bv   = ssm[:, self.dr:self.dr+self.ds]             # [1, ds]
        Cv   = ssm[:, self.dr+self.ds:]                    # [1, ds]

        # 4. dt_proj — [1,dr] -> [1,di], same as ref
        dt = F.softplus(self.dt_proj(dt_r))                # [1, di]

        # 5. SSM state update
        # dt [1,di] -> [1,di*ds]: repeat ds times (ds-major layout)
        # x_act [1,di] -> [1,di*ds]: same repeat
        # Both use .repeat(1, ds) — valid ONNX Tile op, no interleave
        dt_flat  = dt.repeat(1, self.ds)                   # [1, di*ds]
        xa_flat  = x_act.repeat(1, self.ds)               # [1, di*ds]

        # Bv [1,ds] -> [1,di*ds]: repeat di times
        Bv_flat  = Bv.repeat(1, self.di)                  # [1, di*ds]

        A_flat   = -torch.exp(self.A_log.float())          # [1, di*ds]

        dA    = torch.exp(dt_flat * A_flat)                # [1, di*ds]
        dB    = (dA - 1.0) / A_flat * Bv_flat             # [1, di*ds]
        h_new = dA * ssm_state + dB * xa_flat             # [1, di*ds]

        # 6. readout — tiny GEMM 1×ds×di
        y_ssm = self.ssm_out(Cv)                           # [1, di]

        # 7. gate
        y   = (y_ssm + self.D * x_act) * F.silu(z)
        out = self.out_proj(y)

        return out, conv_new, h_new

import os, math

def gen_mapping(folder, file_stem, cfg, seq_len=1, n_blocks=2):
    dim, spad_kb, accum_kb, precision, num_cores = 128, 32768, 4096, 2, 4

    def ceil_div(a, b): return (a + b - 1) // b
    def pad_up(x, d):   return ceil_div(x, d) * d

    def gemm_mapping(N, C, M):
        dI, dJ, dK = N, M, C
        dIp = pad_up(dI, dim); dJp = pad_up(dJ, dim); dKp = pad_up(dK, dim)

        max_spad_rows = (spad_kb * 1024) // (dim * precision * 2)
        max_acc_rows  = (accum_kb * 1024) // (dim * 4 * 2)

        acc   = max(1, max_acc_rows  // dim)
        part  = max(1, (max_spad_rows // 2) // dim)
        sq    = max(1, int(math.sqrt(acc)))
        tk    = max(1, part // sq)

        tI = max(1, min(dIp // dim, ceil_div(dI, sq * dim)))
        tJ = max(1, min(dJp // dim, ceil_div(dJ, sq * dim)))
        tK = max(1, min(dKp // dim, ceil_div(dK, tk * dim)))

        nt = tI * tJ
        if nt < num_cores:
            inc = ceil_div(num_cores, nt)
            tJ = tJ * inc if dJ >= dI else tI * inc

        iI = max(dim, (dIp // tI) // dim * dim)
        iJ = max(dim, (dJp // tJ) // dim * dim)
        iK = max(dim, (dKp // tK) // dim * dim)

        # clamp small dims — avoid zero tiles
        if dI < dim: iI = dIp; tI = 1
        if dJ < dim: iJ = dJp; tJ = 1
        if dK < dim: iK = dKp; tK = 1

        tI = ceil_div(dI, iI)
        tJ = ceil_div(dJ, iJ)
        tK = ceil_div(dK, iK)

        return tI, tK, tJ, iI, iK, iJ

    d  = cfg["d_model"]
    di = cfg["d_inner"]
    ds = cfg["d_state"]
    dr = cfg["dt_rank"]
    dc = cfg["d_conv"]
    d_model=d
    d_inner=di
    d_state=ds
    d_conv=dc
    
    layers = [
    (1, d_model,        2*d_inner      ),  # in_proj    C=768,   M=3072
    (1, d_inner*d_conv, d_inner        ),  # conv_proj  C=6144,  M=1536
    (1, d_inner,        dr + 2*d_state ),  # x_proj     C=1536,  M=80   ← small again
    (1, dr,             d_inner        ),  # dt_proj    C=48,    M=1536
    (1, d_state,        d_inner        ),  # ssm_out    C=16,    M=1536
    (1, d_inner,        d_model        ),  # out_proj   C=1536,  M=768
]

    seen, lines = set(), []
    for _ in range(n_blocks):
        for N, C, M in layers:
            if (N, C, M) not in seen:
                seen.add((N, C, M))
                oN, oC, oM, iN, iC, iM = gemm_mapping(N, C, M)
                lines.append(
                    f"[T] N{N} C{C} M{M} - [O] N{oN} C{oC} M{oM} - [I] N{iN} C{iC} M{iM}"
                )

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{file_stem}.mapping")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Mapping -> {path}  ({len(lines)} unique keys)")
# ─────────────────────────────────────────────
# EXPORT (CLEAN)
# ─────────────────────────────────────────────
def export():
    cfg = MODELS["mamba-tiny"]
    count_mamba_130m_params(cfg)
    model = MambaBlock(
        cfg["d_model"],
        cfg["d_inner"],
        cfg["d_state"],
        cfg["d_conv"],
        cfg["dt_rank"]
    ).eval()

    x    = torch.zeros(1, cfg["d_model"])
    conv = torch.zeros(1, cfg["d_inner"] * cfg["d_conv"])
    ssm  = torch.zeros(1, cfg["d_inner"] * cfg["d_state"])

    os.makedirs("models", exist_ok=True)

    folder = "models/mamba_tiny"
    name = "mamba_tiny"
    os.makedirs(folder, exist_ok=True)


    onnx_path = os.path.join(folder, f"{name}.onnx")

    torch.onnx.export(
        model,
        (x, conv, ssm),
        onnx_path,
        input_names=["x", "conv_state", "ssm_state"],
        output_names=["out", "conv_new", "ssm_new"],
        opset_version=18
    )

    print(f"✅ ONNX saved: {onnx_path}")

    # generate mapping
    gen_mapping(folder, name, cfg)


if __name__ == "__main__":
    export()