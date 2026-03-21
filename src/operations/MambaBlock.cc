#include "MambaBlock.h"
#include "../Model.h"
#include "../Tensor.h"

// ─────────────────────────────────────────────────────────────────────────────
// Inputs (12 total):
//   0  x            [1, d_model]
//   1  conv_state   [1, d_inner, d_conv]
//   2  ssm_state    [1, d_inner, d_state]
//   3  w_in         [2*d_inner, d_model]
//   4  w_conv       [d_inner, 1, d_conv]
//   5  b_conv       [d_inner]
//   6  w_xproj      [dt_rank+2*d_state, d_inner]
//   7  w_dt         [d_inner, dt_rank]
//   8  b_dt         [d_inner]
//   9  A_log        [d_inner, d_state]
//  10  D            [d_inner]
//  11  w_out        [d_model, d_inner]
//
// Outputs (3 total):
//   0  y              [1, d_model]
//   1  conv_state_new [1, d_inner, d_conv]
//   2  ssm_state_new  [1, d_inner, d_state]
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
#include "MambaBlock.h"
#include "../Model.h"
#include "../Tensor.h"
#include <spdlog/spdlog.h>

// ─────────────────────────────────────────────────────────────────────────────
// Inputs (12 total):
// 0 x [1, d_model]
// 1 conv_state [1, d_inner, d_conv]
// 2 ssm_state [1, d_inner, d_state]
// 3 w_in [2*d_inner, d_model]
// 4 w_conv [d_inner, 1, d_conv]
// 5 b_conv [d_inner]
// 6 w_xproj [dt_rank+2*d_state, d_inner]
// 7 w_dt [d_inner, dt_rank]
// 8 b_dt [d_inner]
// 9 A_log [d_inner, d_state]
// 10 D [d_inner]
// 11 w_out [d_model, d_inner]
//
// Outputs (3 total):
// 0 y [1, d_model]
// 1 conv_state_new [1, d_inner, d_conv]
// 2 ssm_state_new [1, d_inner, d_state]
// ─────────────────────────────────────────────────────────────────────────────

MambaBlock::MambaBlock(SimulationConfig config,
                       Model* model,
                       onnx::NodeProto& node_proto,
                       uint32_t target_core)
    : Operation(config, model, node_proto, target_core)
{
    // The base Operation constructor already populated _inputs from node_proto.input()
    // We just verify / log them for clarity
    spdlog::trace("MambaBlock {} created - base inputs count: {}", _name, _inputs.size());

    // ── Extract shapes ────────────────────────────────────────────────────────
    if (_inputs.size() < 12) {
        spdlog::error("MambaBlock {}: expected at least 12 inputs, got {}", _name, _inputs.size());
        // You may want to throw or handle gracefully depending on your error policy
    }

    Tensor* x_tensor          = get_input(0);
    Tensor* conv_state_tensor = get_input(1);
    Tensor* ssm_state_tensor  = get_input(2);

    _input_shape      = x_tensor->get_dims();          // [1, d_model]
    _conv_state_shape = conv_state_tensor->get_dims(); // [1, d_inner, d_conv]
    _ssm_state_shape  = ssm_state_tensor->get_dims();  // [1, d_inner, d_state]

    _batch    = _input_shape[0];
    _d_model  = _input_shape[1];
    _d_inner  = _conv_state_shape[1];
    _d_conv   = _conv_state_shape[2];
    _d_state  = _ssm_state_shape[2];

    uint32_t xproj_out_dim = get_input(6)->get_dims()[0];
    _dt_rank = xproj_out_dim - 2 * _d_state;

    _output_shape = _input_shape;  // y shape = x shape

    // ── Create / register OUTPUT tensors ──────────────────────────────────────
    // Output 0: y
    std::string y_name = node_proto.output(0);
    Tensor* y_tensor = _model->find_tensor(y_name);
    if (!y_tensor) {
        auto new_y = std::make_unique<Tensor>(
            _id, y_name, _output_shape, _config.precision, /*produced=*/false);
        y_tensor = new_y.get();
        _model->add_tensor(std::move(new_y));
    } else {
        y_tensor->redefine_tensor(_id, _output_shape);
    }
    _outputs.push_back(y_tensor->get_id());
    y_tensor->add_child_node(this);  // important for dependency propagation

    // Output 1: conv_state_new (only if node has this output)
    if (node_proto.output_size() > 1) {
        std::string conv_new_name = node_proto.output(1);
        Tensor* conv_new_tensor = _model->find_tensor(conv_new_name);
        if (!conv_new_tensor) {
            auto new_conv = std::make_unique<Tensor>(
                _id, conv_new_name, _conv_state_shape, _config.precision, false);
            conv_new_tensor = new_conv.get();
            _model->add_tensor(std::move(new_conv));
        } else {
            conv_new_tensor->redefine_tensor(_id, _conv_state_shape);
        }
        _outputs.push_back(conv_new_tensor->get_id());
        conv_new_tensor->add_child_node(this);
    }

    // Output 2: ssm_state_new (only if node has this output)
    if (node_proto.output_size() > 2) {
        std::string ssm_new_name = node_proto.output(2);
        Tensor* ssm_new_tensor = _model->find_tensor(ssm_new_name);
        if (!ssm_new_tensor) {
            auto new_ssm = std::make_unique<Tensor>(
                _id, ssm_new_name, _ssm_state_shape, _config.precision, false);
            ssm_new_tensor = new_ssm.get();
            _model->add_tensor(std::move(new_ssm));
        } else {
            ssm_new_tensor->redefine_tensor(_id, _ssm_state_shape);
        }
        _outputs.push_back(ssm_new_tensor->get_id());
        ssm_new_tensor->add_child_node(this);
    }

    // Debug logging - very helpful during integration
    spdlog::info("MambaBlock {} created with:", (this)->check_executable() ? "executable" : "non-executable");
    spdlog::info("MambaBlock {} wired:", _name);
    spdlog::info("  inputs  ({}):", _inputs.size());
    for (size_t i = 0; i < _inputs.size(); ++i) {
        spdlog::info("    [{:2}] {}", i, get_input(i)->get_name());
    }
    spdlog::info("  outputs ({}):", _outputs.size());
    for (size_t i = 0; i < _outputs.size(); ++i) {
        spdlog::info("    [{:2}] {}", i, get_output(i)->get_name());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The rest of your implementation (initialize_tiles, push_tile, inject_dep,
// initialize_instructions, etc.) remains almost unchanged.
// Only minor suggestion: add trace logging in initialize_instructions
// to confirm SRAM layout / operand addresses if needed.
void MambaBlock::initialize_tiles(MappingTable& mapping_table)
{
    spdlog::trace("Initializing MambaBlock {} as SINGLE tile on core {}", _name, target_core);

    auto tile = std::make_unique<Tile>(Tile{
        .status      = Tile::Status::INITIALIZED,
        .optype      = get_name() + "/full",
        .layer_id    = _id,
        .fused_op_id = 0,
        .accum       = false,
        .core_id     = (int)target_core,
    });

    initialize_instructions(tile.get());

    _tiles.push_back(std::move(tile));

    spdlog::info("MambaBlock {}: 1 tile created", _name);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<addr_type> make_addrs(addr_type base,
                                         uint32_t  nbytes,
                                         uint32_t  req)
{
    std::set<addr_type> s;
    for (int off = 0; off < (int)nbytes; off += (int)req)
        s.insert(base + (addr_type)off);
    if (s.empty()) s.insert(base);
    return {s.begin(), s.end()};
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize_instructions
//
// SRAM layout — per-stage weight regions are NON-OVERLAPPING so no tile can
// clobber a weight buffer still in use by another tile running concurrently.
//
//   SPAD_BASE
//   ├─ sram_x             [d_model]
//   ├─ sram_conv_st       [d_inner * d_conv]
//   ├─ sram_ssm_st        [d_inner * d_state]
//   ├─ sram_w_in_proj     [2*d_inner*d_model]
//   ├─ sram_w_conv1d      [d_inner*d_conv + d_inner]    (w_conv || b_conv)
//   ├─ sram_w_xproj       [(dt_rank+2*d_state)*d_inner]
//   ├─ sram_w_dt_proj     [d_inner*dt_rank + d_inner]   (w_dt  || b_dt)
//   ├─ sram_w_ssm         [d_inner*d_state + d_inner]   (A_log || D)
//   ├─ sram_w_out_proj    [d_model*d_inner]
//   ├─ sram_xin           [d_inner]
//   ├─ sram_z             [d_inner]
//   ├─ sram_xact          [d_inner]
//   ├─ sram_dt_raw        [dt_rank]
//   ├─ sram_B             [d_state]
//   ├─ sram_C             [d_state]
//   ├─ sram_dt            [d_inner]
//   ├─ sram_y             [d_inner]
//   ACCUM_SPAD_BASE
//   └─ sram_out           [d_model]
// ─────────────────────────────────────────────────────────────────────────────
void MambaBlock::initialize_instructions(Tile* tile)
{
    // ── byte sizes ────────────────────────────────────────────────────────────
    const uint32_t Bm   = bytes(_d_model);
    const uint32_t Bi   = bytes(_d_inner);
    const uint32_t Bs   = bytes(_d_state);
    const uint32_t Bdt  = bytes(_dt_rank);
    const uint32_t Bcs  = bytes(_d_inner * _d_conv);
    const uint32_t Bss  = bytes(_d_inner * _d_state);
    const uint32_t req  = _config.dram_req_size;

    const uint32_t Bw_in_proj  = bytes(2 * _d_inner * _d_model);
    const uint32_t Bwconv      = bytes(_d_inner * _d_conv);
    const uint32_t Bw_conv1d   = Bwconv + Bi;
    const uint32_t Bw_xproj    = bytes((_dt_rank + 2 * _d_state) * _d_inner);
    const uint32_t Bwdt        = bytes(_d_inner * _dt_rank);
    const uint32_t Bw_dt_proj  = Bwdt + Bi;
    const uint32_t Bw_ssm      = Bss + Bi;
    const uint32_t Bw_out_proj = bytes(_d_model * _d_inner);

    auto chunks = [&](uint32_t nb) -> uint32_t { return std::max(1u, nb / req); };

    // ── SRAM addresses ────────────────────────────────────────────────────────
    addr_type sram_x          = SPAD_BASE;
    addr_type sram_conv_st    = sram_x         + Bm;
    addr_type sram_ssm_st     = sram_conv_st   + Bcs;

    addr_type sram_w_in_proj  = sram_ssm_st    + Bss;
    addr_type sram_w_conv1d   = sram_w_in_proj + Bw_in_proj;
    addr_type sram_w_xproj    = sram_w_conv1d  + Bw_conv1d;
    addr_type sram_w_dt_proj  = sram_w_xproj   + Bw_xproj;
    addr_type sram_w_ssm      = sram_w_dt_proj + Bw_dt_proj;
    addr_type sram_w_out_proj = sram_w_ssm     + Bw_ssm;

    addr_type sram_xin    = sram_w_out_proj + Bw_out_proj;
    addr_type sram_z      = sram_xin        + Bi;
    addr_type sram_xact   = sram_z          + Bi;
    addr_type sram_dt_raw = sram_xact       + Bi;
    addr_type sram_B      = sram_dt_raw     + Bdt;
    addr_type sram_C      = sram_B          + Bs;
    addr_type sram_dt     = sram_C          + Bs;
    addr_type sram_y      = sram_dt         + Bi;

    addr_type sram_out    = ACCUM_SPAD_BASE;

    // ── DRAM addresses ────────────────────────────────────────────────────────
    addr_type dram_x        = get_operand_addr(_INPUT_OPERAND);
    addr_type dram_conv_st  = get_operand_addr(_INPUT_OPERAND + 1);
    addr_type dram_ssm_st   = get_operand_addr(_INPUT_OPERAND + 2);
    addr_type dram_w_in     = get_operand_addr(_INPUT_OPERAND + 3);
    addr_type dram_w_conv   = get_operand_addr(_INPUT_OPERAND + 4);
    addr_type dram_b_conv   = get_operand_addr(_INPUT_OPERAND + 5);
    addr_type dram_w_xproj  = get_operand_addr(_INPUT_OPERAND + 6);
    addr_type dram_w_dt     = get_operand_addr(_INPUT_OPERAND + 7);
    addr_type dram_b_dt     = get_operand_addr(_INPUT_OPERAND + 8);
    addr_type dram_A_log    = get_operand_addr(_INPUT_OPERAND + 9);
    addr_type dram_D        = get_operand_addr(_INPUT_OPERAND + 10);
    addr_type dram_w_out    = get_operand_addr(_INPUT_OPERAND + 11);
    addr_type dram_y        = get_operand_addr(_OUTPUT_OPERAND);
    addr_type dram_conv_new = get_operand_addr(_OUTPUT_OPERAND + 1);
    addr_type dram_ssm_new  = get_operand_addr(_OUTPUT_OPERAND + 2);
    // 1. in_proj
    {
        auto ax = make_addrs(dram_x, Bm, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_x,
            .size = (uint32_t)ax.size(),
            .src_addrs = ax,
            .operand_id = _INPUT_OPERAND + 0
        }));

        auto awi = make_addrs(dram_w_in, Bw_in_proj, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_w_in_proj,
            .size = (uint32_t)awi.size(),
            .src_addrs = awi,
            .operand_id = _INPUT_OPERAND + 3
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_xin,
            .size = chunks(2 * Bi),
            .compute_size = 2 * Bi,
            .src_addrs = {sram_x, sram_w_in_proj},
            .tile_m = 2 * _d_inner,
            .tile_k = _d_model,
            .tile_n = _batch
        }));
    }

    // 2. conv1d
    {
        auto acs = make_addrs(dram_conv_st, Bcs, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_conv_st,
            .size = (uint32_t)acs.size(),
            .src_addrs = acs,
            .operand_id = _INPUT_OPERAND + 1
            // IMPORTANT: no inject_dep here — we assume conv_state is already produced
        }));

        auto awc = make_addrs(dram_w_conv, Bwconv, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_w_conv1d,
            .size = (uint32_t)awc.size(),
            .src_addrs = awc,
            .operand_id = _INPUT_OPERAND + 4
        }));

        auto abc = make_addrs(dram_b_conv, Bi, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_w_conv1d + Bwconv,
            .size = (uint32_t)abc.size(),
            .src_addrs = abc,
            .operand_id = _INPUT_OPERAND + 5
        }));

        // Shift + insert xin (depends on previous GEMM via order)
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::COMP,
            .dest_addr = sram_conv_st,
            .size = chunks(Bcs),
            .compute_size = Bcs,
            .src_addrs = {sram_conv_st, sram_xin}
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_xact,
            .size = chunks(Bi),
            .compute_size = Bi,
            .src_addrs = {sram_conv_st, sram_w_conv1d},
            .tile_m = _d_inner,
            .tile_k = _d_conv,
            .tile_n = _batch
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::COMP,
            .dest_addr = sram_xact,
            .size = chunks(Bi),
            .compute_size = Bi,
            .src_addrs = {sram_xact}
        }));

        auto acn = make_addrs(dram_conv_new, Bcs, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_conv_st,
            .size = (uint32_t)acn.size(),
            .src_addrs = acn,
            .operand_id = _OUTPUT_OPERAND + 1
        }));
    }

    // 3. x_proj
    {
        auto axp = make_addrs(dram_w_xproj, Bw_xproj, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_w_xproj,
            .size = (uint32_t)axp.size(),
            .src_addrs = axp,
            .operand_id = _INPUT_OPERAND + 6
        }));

        uint32_t xproj_out = _dt_rank + 2 * _d_state;
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_dt_raw,
            .size = chunks(bytes(xproj_out)),
            .compute_size = bytes(xproj_out),
            .src_addrs = {sram_xact, sram_w_xproj},
            .tile_m = xproj_out,
            .tile_k = _d_inner,
            .tile_n = _batch
        }));
    }

    // 4. dt_proj + softplus
    {
        auto awt = make_addrs(dram_w_dt, Bwdt, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_w_dt_proj,
            .size = (uint32_t)awt.size(),
            .src_addrs = awt,
            .operand_id = _INPUT_OPERAND + 7
        }));

        auto abt = make_addrs(dram_b_dt, Bi, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_w_dt_proj + Bwdt,
            .size = (uint32_t)abt.size(),
            .src_addrs = abt,
            .operand_id = _INPUT_OPERAND + 8
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_dt,
            .size = chunks(Bi),
            .compute_size = Bi,
            .src_addrs = {sram_dt_raw, sram_w_dt_proj},
            .tile_m = _d_inner,
            .tile_k = _dt_rank,
            .tile_n = _batch
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::ADD,
            .dest_addr = sram_dt,
            .size = chunks(Bi),
            .compute_size = Bi,
            .src_addrs = {sram_dt, sram_w_dt_proj + Bwdt}
        }));

        // softplus = log(1 + exp(dt))   → approximated with two EXP + ADD pairs
        for (int s = 0; s < 2; ++s) {
            tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
                .opcode = Opcode::EXP,
                .dest_addr = sram_dt,
                .size = chunks(Bi),
                .compute_size = Bi,
                .src_addrs = {sram_dt}
            }));
            tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
                .opcode = Opcode::ADD,
                .dest_addr = sram_dt,
                .size = chunks(Bi),
                .compute_size = Bi,
                .src_addrs = {sram_dt}
            }));
        }
    }

    // 5. ssm_update
    {
        auto ass = make_addrs(dram_ssm_st, Bss, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_ssm_st,
            .size = (uint32_t)ass.size(),
            .src_addrs = ass,
            .operand_id = _INPUT_OPERAND + 2
        }));

        auto aal = make_addrs(dram_A_log, Bss, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_w_ssm,
            .size = (uint32_t)aal.size(),
            .src_addrs = aal,
            .operand_id = _INPUT_OPERAND + 9
        }));

        auto aD = make_addrs(dram_D, Bi, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_w_ssm + Bss,
            .size = (uint32_t)aD.size(),
            .src_addrs = aD,
            .operand_id = _INPUT_OPERAND + 10
        }));

        // dA = exp(dt * A_log)
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::EXP,
            .dest_addr = sram_ssm_st,
            .size = chunks(Bss),
            .compute_size = Bss,
            .src_addrs = {sram_dt, sram_w_ssm}
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MUL,
            .dest_addr = sram_y,
            .size = chunks(Bss),
            .compute_size = Bss,
            .src_addrs = {sram_dt, sram_B}
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MUL,
            .dest_addr = sram_ssm_st,
            .size = chunks(Bss),
            .compute_size = Bss,
            .src_addrs = {sram_ssm_st, sram_xact}
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::ADD,
            .dest_addr = sram_ssm_st,
            .size = chunks(Bss),
            .compute_size = Bss,
            .src_addrs = {sram_ssm_st, sram_y}
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_y,
            .size = chunks(Bi),
            .compute_size = Bi,
            .src_addrs = {sram_ssm_st, sram_C},
            .tile_m = _d_inner,
            .tile_k = _d_state,
            .tile_n = _batch
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MUL,
            .dest_addr = sram_y,
            .size = chunks(Bi),
            .compute_size = Bi,
            .src_addrs = {sram_xact, sram_w_ssm + Bss}
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::ADD,
            .dest_addr = sram_y,
            .size = chunks(Bi),
            .compute_size = Bi,
            .src_addrs = {sram_y, sram_y}
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::COMP,
            .dest_addr = sram_z,
            .size = chunks(Bi),
            .compute_size = Bi,
            .src_addrs = {sram_z}
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MUL,
            .dest_addr = sram_y,
            .size = chunks(Bi),
            .compute_size = Bi,
            .src_addrs = {sram_y, sram_z}
        }));

        auto asn = make_addrs(dram_ssm_new, Bss, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_ssm_st,
            .size = (uint32_t)asn.size(),
            .src_addrs = asn,
            .operand_id = _OUTPUT_OPERAND + 2
        }));
    }

    // 6. out_proj
    {
        auto awo = make_addrs(dram_w_out, Bw_out_proj, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_w_out_proj,
            .size = (uint32_t)awo.size(),
            .src_addrs = awo,
            .operand_id = _INPUT_OPERAND + 11
        }));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_out,
            .size = chunks(Bm),
            .compute_size = Bm,
            .src_addrs = {sram_y, sram_w_out_proj},
            .tile_m = _d_model,
            .tile_k = _d_inner,
            .tile_n = _batch
        }));

        auto ay = make_addrs(dram_y, Bm, req);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_out,
            .size = (uint32_t)ay.size(),
            .src_addrs = ay,
            .operand_id = _OUTPUT_OPERAND + 0
        }));
    }

    spdlog::info("[{}] Single tile has {} instructions", get_name(), tile->instructions.size());
}