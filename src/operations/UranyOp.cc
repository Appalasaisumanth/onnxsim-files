#include "UnaryOp.h"
#include "../Model.h"

// ─────────────────────────────────────────────────────────────────────────────
// resolve_opcodes
// Maps each ONNX op-type to the EXACT ordered list of hardware opcodes needed.
// Each entry in _opcodes becomes one COMP-class instruction in every tile.
//
// Decompositions (all element-wise, one full pass per opcode):
//
//  Relu    : max(0, x)
//              → COMP
//
//  Neg     : -x
//              → COMP
//
//  Abs     : |x|
//              → COMP
//
//  Sqrt    : √x
//              → COMP
//
//  Cast    : bit-cast / type convert  (no arithmetic meaning in simulation)
//              → COMP
//
//  Exp     : e^x
//              → EXP
//
//  Gelu    : x · Φ(x)   (hardware GELU unit)
//              → GELU
//
//  Tanh    : (e^2x − 1) / (e^2x + 1)
//              step 1: EXP   compute e^2x  (scale handled by preceding MUL)
//              step 2: MUL   2·x  (pre-scale input)
//              step 3: ADD   numerator  = e^2x − 1
//              step 4: ADD   denominator= e^2x + 1
//              step 5: DIV   result     = num / denom
//            Emitted order: MUL → EXP → ADD → ADD → DIV
//
//  Sigmoid : 1 / (1 + e^−x)
//              step 1: COMP  negate  −x
//              step 2: EXP   e^−x
//              step 3: ADD   1 + e^−x
//              step 4: DIV   1 / (1 + e^−x)
//            Emitted order: COMP → EXP → ADD → DIV
//
//  Erf     : Abramowitz & Stegun rational approximation (max |err| < 2.5e-5)
//              erf(x) ≈ sign(x)·(1 − p(t)·e^(−x²))
//              t = 1 / (1 + 0.47047·|x|)
//              p(t) = t·(0.3480242 − t·(0.0958798 − t·0.7478556))
//
//              step 1: COMP  |x| and sign(x)          → abs + sign
//              step 2: MUL   0.47047·|x|               → scale
//              step 3: ADD   1 + 0.47047·|x|           → denominator of t
//              step 4: DIV   t = 1 / (1 + 0.47047·|x|)
//              step 5: MUL   t² = t·t
//              step 6: MUL   t³ = t²·t
//              step 7: MUL   a3·t³
//              step 8: ADD   a2·t² − a3·t³  (ADD with negated coeff)
//              step 9: MUL   t·(…)
//              step 10: ADD  a1·t + …        polynomial p(t)
//              step 11: COMP −x²             (negate then square via COMP)
//              step 12: EXP  e^(−x²)
//              step 13: MUL  p(t)·e^(−x²)
//              step 14: COMP 1 − p(t)·e^(−x²)
//              step 15: MUL  sign(x) · result
//            Emitted order:
//              COMP → MUL → ADD → DIV →
//              MUL  → MUL → MUL → ADD → MUL → ADD →
//              COMP → EXP → MUL → COMP → MUL
// ─────────────────────────────────────────────────────────────────────────────
void UnaryOp::resolve_opcodes(const std::string& optype) {
    _opcodes.clear();

    if (optype == "Relu" ||
        optype == "Neg"  ||
        optype == "Abs"  ||
        optype == "Sqrt" ||
        optype == "Cast") {
        _opcodes = { Opcode::COMP };
        return;
    }

    if (optype == "Exp") {
        _opcodes = { Opcode::EXP };
        return;
    }

    if (optype == "Gelu") {
        _opcodes = { Opcode::GELU };
        return;
    }

    if (optype == "Tanh") {
        // (e^2x − 1) / (e^2x + 1)
        // MUL(2x) → EXP(e^2x) → ADD(num=e^2x−1) → ADD(den=e^2x+1) → DIV
        _opcodes = {
            Opcode::MUL,   // 2·x
            Opcode::EXP,   // e^(2x)
            Opcode::ADD,   // numerator:   e^2x − 1
            Opcode::ADD,   // denominator: e^2x + 1
            Opcode::DIV,   // result:      num / den
        };
        return;
    }
    if (optype == "Silu" || optype == "SiLU") {
    // SiLU(x) = x · sigmoid(x) = x / (1 + e^{-x})
    // Reuse sigmoid decomposition then multiply by x:
    //   step 1: COMP  negate:   −x
    //   step 2: EXP           e^(−x)
    //   step 3: ADD   1 + e^(−x)
    //   step 4: DIV   sigmoid = 1 / (1 + e^(−x))
    //   step 5: MUL   x · sigmoid(x)
    _opcodes = {
        Opcode::COMP,  // −x
        Opcode::EXP,   // e^(−x)
        Opcode::ADD,   // 1 + e^(−x)
        Opcode::DIV,   // sigmoid(x) = 1 / (1 + e^(−x))
        Opcode::MUL,   // x · sigmoid(x)
    };
    return;
}

    if (optype == "Sigmoid") {
        // 1 / (1 + e^−x)
        // COMP(−x) → EXP(e^−x) → ADD(1+e^−x) → DIV(1/…)
        _opcodes = {
            Opcode::COMP,  // negate: −x
            Opcode::EXP,   // e^(−x)
            Opcode::ADD,   // 1 + e^(−x)
            Opcode::DIV,   // 1 / (1 + e^(−x))
        };
        return;
    }

    if (optype == "Erf") {
        // Abramowitz & Stegun rational approximation
        // erf(x) ≈ sign(x)·(1 − p(t)·e^(−x²)),  t=1/(1+0.47047|x|)
        _opcodes = {
            Opcode::COMP,  //  1: |x|, save sign(x)
            Opcode::MUL,   //  2: 0.47047·|x|
            Opcode::ADD,   //  3: 1 + 0.47047·|x|
            Opcode::DIV,   //  4: t = 1 / (1 + 0.47047·|x|)
            Opcode::MUL,   //  5: t²
            Opcode::MUL,   //  6: t³
            Opcode::MUL,   //  7: a3·t³
            Opcode::ADD,   //  8: a2·t² − a3·t³
            Opcode::MUL,   //  9: t·(…)
            Opcode::ADD,   // 10: a1·t + (…)   → p(t) complete
            Opcode::COMP,  // 11: −x²
            Opcode::EXP,   // 12: e^(−x²)
            Opcode::MUL,   // 13: p(t)·e^(−x²)
            Opcode::COMP,  // 14: 1 − p(t)·e^(−x²)
            Opcode::MUL,   // 15: sign(x)·result
        };
        return;
    }
    // ── in resolve_opcodes(), before the fallback warn ──────────────────────────

    if (optype == "Softplus") {
        // softplus(x) = log(1 + e^x)   — exact, no approximation
        //   step 1: EXP  →  e^x
        //   step 2: ADD  →  1 + e^x
        //   step 3: LOG  →  log(1 + e^x)
        _opcodes = {
            Opcode::EXP,   // e^x
            Opcode::ADD,   // 1 + e^x
            Opcode::EXP,   // log(1 + e^x)  //log and exp have same opcode in our simulation since we treat log as a black-box approximation
        };
        return;
    }

    // fallback
    spdlog::warn("[UnaryOp] Unknown op-type '{}', defaulting to COMP", optype);
    _opcodes = { Opcode::COMP };
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

UnaryOp::UnaryOp(SimulationConfig config, Model* model,
                 onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {

    resolve_opcodes(node_proto.op_type());

    _input_shape = get_input(0)->get_dims();

    _batch_size = 1;
    _seq        = 1;
    _dk         = 1;
    for (auto d : _input_shape)
        _dk *= d;

    _output_shape = _input_shape;

    Tensor* pre_defined = _model->find_tensor(node_proto.output(0));
    if (pre_defined == nullptr) {
        auto out = std::make_unique<Tensor>(
            _id, node_proto.output(0), _output_shape, _config.precision, false);
        _outputs.push_back(out.get()->get_id());
        _model->add_tensor(std::move(out));
    } else {
        pre_defined->redefine_tensor(_id, _output_shape);
    }

    calculate_loops();
}

UnaryOp::UnaryOp(SimulationConfig config, Model* model,
                 std::string name,
                 std::map<std::string, std::string>& attributes,
                 uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
    resolve_opcodes(name);
}

UnaryOp::UnaryOp(const UnaryOp& src)
    : Operation(src),
      _input_shape(src._input_shape),
      _output_shape(src._output_shape),
      _batch_size(src._batch_size),
      _seq(src._seq),
      _dk(src._dk),
      _tokens_per_tile(src._tokens_per_tile),
      _opcodes(src._opcodes) {}

// ─────────────────────────────────────────────────────────────────────────────
// Tiling
// ─────────────────────────────────────────────────────────────────────────────

void UnaryOp::calculate_loops() {
    uint32_t sram_capacity = _config.core_config[target_core].spad_size KB / 2;
    _tokens_per_tile = sram_capacity / (_config.precision);
    if (_tokens_per_tile == 0) _tokens_per_tile = 1;
    if (_tokens_per_tile > _dk) _tokens_per_tile = _dk;
}

void UnaryOp::initialize_tiles(MappingTable& mapping_table) {
    for (uint32_t offset = 0; offset < _dk; offset += _tokens_per_tile) {
        uint32_t remain = std::min(_dk - offset, _tokens_per_tile);

        auto tile = std::make_unique<Tile>(Tile{
            .status   = Tile::Status::INITIALIZED,
            .optype   = get_name(),
            .layer_id = _id,
            .accum    = false,
        });

        Mapping mapping;
        _tiles.push_back(std::move(tile));
        initialize_instructions(_tiles.back().get(), mapping, offset, remain);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize_instructions
//
// Emits for ONE tile:
//   MOVIN                          – load input tile from DRAM
//   [opcode_0]                     – first primitive pass
//   [opcode_1]                     – second primitive pass  (if multi-step op)
//   ...
//   [opcode_N]
//   MOVOUT                         – write result tile to DRAM
//
// Every intermediate pass reads and writes the SAME sram_base region so the
// scratchpad acts as a single accumulator buffer – matching how COMP/EXP/etc.
// work in the existing codebase (in-place element-wise transforms).
// ─────────────────────────────────────────────────────────────────────────────
void UnaryOp::initialize_instructions(Tile*    tile,
                                       Mapping  mapping,
                                       uint32_t token_offset,
                                       uint32_t tokens) {
    addr_type sram_base   = SPAD_BASE;
    addr_type input_addr  = get_operand_addr(_INPUT_OPERAND);
    addr_type output_addr = get_operand_addr(_OUTPUT_OPERAND);

    uint32_t tile_bytes = tokens * _config.precision;
    uint32_t base_off   = token_offset * _config.precision;

    // ── collect DRAM address sets ────────────────────────────────────────────
    std::set<addr_type> dram_in_addrs;
    std::set<addr_type> dram_out_addrs;

    for (int off = 0; off < (int)tile_bytes; off += (int)_config.dram_req_size) {
        dram_in_addrs .insert(input_addr  + base_off + off);
        dram_out_addrs.insert(output_addr + base_off + off);
    }

    // ── MOVIN: load input tile ───────────────────────────────────────────────
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode     = Opcode::MOVIN,
        .dest_addr  = sram_base,
        .size       = (uint32_t)dram_in_addrs.size(),
        .src_addrs  = std::vector<addr_type>(dram_in_addrs.begin(),
                                             dram_in_addrs.end()),
        .operand_id = _INPUT_OPERAND,
    }));

    // ── arithmetic passes: one instruction per opcode in _opcodes ────────────
    // Each pass is an in-place element-wise transform over the scratchpad tile.
    // compute_size = number of elements × bytes-per-element.
    // size         = number of DRAM-request-sized chunks (used for latency
    //                accounting in the simulator core).
    for (Opcode opc : _opcodes) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode       = opc,
            .dest_addr    = sram_base,
            .size         = tile_bytes / _config.dram_req_size,
            .compute_size = tile_bytes,
            .src_addrs    = std::vector<addr_type>{ sram_base },
        }));
    }

    // ── MOVOUT: write result back to DRAM ────────────────────────────────────
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode     = Opcode::MOVOUT,
        .dest_addr  = sram_base,
        .size       = (uint32_t)dram_out_addrs.size(),
        .src_addrs  = std::vector<addr_type>(dram_out_addrs.begin(),
                                             dram_out_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
    }));
}