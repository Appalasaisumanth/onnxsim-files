#include "BinaryOp.h"
#include "../Model.h"

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

static uint32_t flatten(const std::vector<uint32_t>& shape)
{
    uint32_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

void BinaryOp::resolve_opcode(const std::string& optype)
{
    if      (optype == "Add")     _comp_opcode = Opcode::ADD;
    else if (optype == "Sub")     _comp_opcode = Opcode::ADD; // same latency
    else if (optype == "Mul")     _comp_opcode = Opcode::MUL;
    else if (optype == "Div")     _comp_opcode = Opcode::DIV;
    else if (optype == "Equal")   _comp_opcode = Opcode::COMP;
    else if (optype == "Where")   _comp_opcode = Opcode::COMP;
    else if (optype == "Greater") _comp_opcode = Opcode::COMP;
    else {
        spdlog::warn("[BinaryOp] Unknown op '{}'", optype);
        _comp_opcode = Opcode::ADD;
    }
}

// ─────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────

BinaryOp::BinaryOp(SimulationConfig config,
                   Model* model,
                   onnx::NodeProto& node_proto,
                   uint32_t target_core)
: Operation(config, model, node_proto, target_core)
{
    resolve_opcode(node_proto.op_type());

    _input_shape_a = get_input(0)->get_dims();
    _input_shape_b = get_input(1)->get_dims();

    uint32_t dk_a = flatten(_input_shape_a);
    uint32_t dk_b = flatten(_input_shape_b);

    _broadcast_b = dk_b < dk_a;

    _dk = std::max(dk_a, dk_b);

    _output_shape = dk_a >= dk_b ? _input_shape_a : _input_shape_b;

    Tensor* out = _model->find_tensor(node_proto.output(0));

    if (out == nullptr)
    {
        auto tensor = std::make_unique<Tensor>(
            _id,
            node_proto.output(0),
            _output_shape,
            _config.precision,
            false);

        _outputs.push_back(tensor->get_id());
        _model->add_tensor(std::move(tensor));
    }
    else
    {
        out->redefine_tensor(_id, _output_shape);
    }

    calculate_loops();
}

// ─────────────────────────────────────────────
// Loop tiling
// ─────────────────────────────────────────────

void BinaryOp::calculate_loops()
{  uint32_t spad = _config.core_config[target_core].spad_size KB;
    uint32_t usable = spad / 3;

    _tokens_per_tile = usable / _config.precision;

    if (_tokens_per_tile == 0)
        _tokens_per_tile = 1;

    if (_tokens_per_tile > _dk)
        _tokens_per_tile = _dk;
}

// ─────────────────────────────────────────────
// Tile generation
// ─────────────────────────────────────────────

void BinaryOp::initialize_tiles(MappingTable&)
{
    for (uint32_t offset = 0; offset < _dk; offset += _tokens_per_tile)
    {
        uint32_t remain =
            std::min(_dk - offset, _tokens_per_tile);

        auto tile = std::make_unique<Tile>(Tile{
            .status = Tile::Status::INITIALIZED,
            .optype = get_name(),
            .layer_id = _id,
            .accum = false
        });

        Mapping mapping;

        _tiles.push_back(std::move(tile));

        initialize_instructions(
            _tiles.back().get(),
            mapping,
            offset,
            remain);
    }
}

// ─────────────────────────────────────────────
// Instruction generation
// ─────────────────────────────────────────────

void BinaryOp::initialize_instructions(
    Tile* tile,
    Mapping,
    uint32_t offset,
    uint32_t tokens)
{
    addr_type sram_a   = SPAD_BASE;
addr_type sram_b   = sram_a + _tokens_per_tile * _config.precision;
addr_type sram_out = sram_b + _tokens_per_tile * _config.precision;

    addr_type dram_a = get_operand_addr(_INPUT_OPERAND);
    addr_type dram_b = get_operand_addr(_INPUT_OPERAND + 1);
    addr_type dram_out = get_operand_addr(_OUTPUT_OPERAND);

    uint32_t tile_bytes = tokens * _config.precision;

    std::set<addr_type> addrs_a;
    std::set<addr_type> addrs_b;
    std::set<addr_type> addrs_out;

    uint32_t base_off = offset * _config.precision;

    // operand A
    for (uint32_t off = 0;
         off < tile_bytes;
         off += _config.dram_req_size)
    {
        addrs_a.insert(dram_a + base_off + off);
    }

    // operand B
    if (_broadcast_b)
    {
        addrs_b.insert(dram_b);  // single value reused
    }
    else
    {
        for (uint32_t off = 0;
             off < tile_bytes;
             off += _config.dram_req_size)
        {
            addrs_b.insert(dram_b + base_off + off);
        }
    }

    // output
    for (uint32_t off = 0;
         off < tile_bytes;
         off += _config.dram_req_size)
    {
        addrs_out.insert(dram_out + base_off + off);
    }

    // MOVIN A
    tile->instructions.push_back(
        std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_a,
            .size = (uint32_t)addrs_a.size(),
            .src_addrs =
                std::vector<addr_type>(addrs_a.begin(), addrs_a.end()),
            .operand_id = _INPUT_OPERAND
        }));

    // MOVIN B
    tile->instructions.push_back(
        std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_b,
            .size = (uint32_t)addrs_b.size(),
            .src_addrs =
                std::vector<addr_type>(addrs_b.begin(), addrs_b.end()),
            .operand_id = _INPUT_OPERAND + 1
        }));

    // compute
    tile->instructions.push_back(
        std::make_unique<Instruction>(Instruction{
            .opcode = _comp_opcode,
            .dest_addr = sram_out,
            .size = tile_bytes / _config.dram_req_size,
            .compute_size = tile_bytes,
            .src_addrs = {sram_a, sram_b}
        }));

    // MOVOUT
    tile->instructions.push_back(
        std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_out,
            .size = (uint32_t)addrs_out.size(),
            .src_addrs =
                std::vector<addr_type>(addrs_out.begin(), addrs_out.end()),
            .operand_id = _OUTPUT_OPERAND
        }));
}