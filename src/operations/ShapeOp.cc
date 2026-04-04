#include "ShapeOp.h"
#include "../Model.h"

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static uint32_t shape_flatten(const std::vector<uint32_t>& shape) {
    uint32_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// ─────────────────────────────────────────────────────────────────────────────

void ShapeOp::resolve_output_shape(onnx::NodeProto& node_proto) {
    const std::string& optype = node_proto.op_type();

    // ─────────────────────────────────────────────────────────────
    // Identity-like ops
    // ─────────────────────────────────────────────────────────────
    if (optype == "Identity" || optype == "Transpose") {
        _output_shape = _input_shape;
        return;
    }

    // ─────────────────────────────────────────────────────────────
    // Shape
    // ─────────────────────────────────────────────────────────────
    if (optype == "Shape") {
        _output_shape = { (uint32_t)_input_shape.size() };
        return;
    }

    // ─────────────────────────────────────────────────────────────
    // Expand
    // ─────────────────────────────────────────────────────────────
    if (optype == "Expand") {
        _output_shape = get_input(1)->get_dims();  // target shape
        return;
    }

    // ─────────────────────────────────────────────────────────────
    // Concat
    // ─────────────────────────────────────────────────────────────
    if (optype == "Concat") {
        int axis = 0;
        for (auto& attr : node_proto.attribute()) {
            if (attr.name() == "axis") axis = attr.i();
        }

        _output_shape = get_input(0)->get_dims();

        uint32_t sum = 0;
        for (int i = 0; i < node_proto.input_size(); i++) {
            sum += get_input(i)->get_dims()[axis];
        }

        _output_shape[axis] = sum;
        return;
    }

    // ─────────────────────────────────────────────────────────────
    // Reshape (FIXED)
    // ─────────────────────────────────────────────────────────────
    if (optype == "Reshape") {
        auto target = get_input(1)->get_dims(); // fallback

        _output_shape.clear();

        uint32_t known_product = 1;
        int minus_one_idx = -1;

        for (int i = 0; i < target.size(); i++) {
            int dim = target[i];

            if (dim == 0) {
                _output_shape.push_back(_input_shape[i]);
                known_product *= _input_shape[i];
            }
            else if (dim == -1) {
                minus_one_idx = i;
                _output_shape.push_back(1); // temp
            }
            else {
                _output_shape.push_back(dim);
                known_product *= dim;
            }
        }

        // handle -1
        if (minus_one_idx != -1) {
            uint32_t total = shape_flatten(_input_shape);
            _output_shape[minus_one_idx] = total / known_product;
        }

        return;
    }

    // ─────────────────────────────────────────────────────────────
    // Slice (SAFE fallback)
    // ─────────────────────────────────────────────────────────────
   
       if (optype == "Slice") {
    // 🔥 DO NOT guess shape
    // Instead, use pre-defined tensor from model graph

    Tensor* out_tensor = _model->find_tensor(node_proto.output(0));

    if (out_tensor != nullptr && !out_tensor->get_dims().empty()) {
        _output_shape = out_tensor->get_dims();
    } else {
        // fallback (safe)
        _output_shape = _input_shape;
    }

    return;
}
    

    // ─────────────────────────────────────────────────────────────
    // ReduceSum
    // ─────────────────────────────────────────────────────────────
    if (optype == "ReduceSum") {
        bool keepdims = true;

        for (auto& attr : node_proto.attribute()) {
            if (attr.name() == "keepdims") {
                keepdims = attr.i();
            }
        }

        _output_shape = _input_shape;

        if (!_output_shape.empty()) {
            if (keepdims) {
                _output_shape.back() = 1;
            } else {
                _output_shape.pop_back();
            }
        }

        if (_output_shape.empty()) _output_shape = {1};

        return;
    }

    // ─────────────────────────────────────────────────────────────
    // Unsqueeze
    // ─────────────────────────────────────────────────────────────
    if (optype == "Unsqueeze") {
        _output_shape = _input_shape;

        // ideally read axes input → fallback append
        _output_shape.push_back(1);

        return;
    }

    // ─────────────────────────────────────────────────────────────
    // Squeeze
    // ─────────────────────────────────────────────────────────────
    if (optype == "Squeeze") {
        _output_shape.clear();

        for (auto d : _input_shape) {
            if (d != 1) _output_shape.push_back(d);
        }

        if (_output_shape.empty()) _output_shape = {1};

        return;
    }

    // ─────────────────────────────────────────────────────────────
    // Gather
    // ─────────────────────────────────────────────────────────────
    if (optype == "Gather") {
        _output_shape = _input_shape;
        return;
    }

    // ─────────────────────────────────────────────────────────────
    // Constant / ConstantOfShape
    // ─────────────────────────────────────────────────────────────
    if (optype == "Constant" || optype == "ConstantOfShape") {
        _output_shape = {1};
        return;
    }

    // ─────────────────────────────────────────────────────────────
    // fallback
    // ─────────────────────────────────────────────────────────────
    spdlog::info("[ShapeOp] {} -> input: ", optype);
for (auto d : _input_shape) std::cerr << d << " ";

std::cerr << " | output: ";
for (auto d : _output_shape) std::cerr << d << " ";

std::cerr << std::endl;
    _output_shape = _input_shape;
}
// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

ShapeOp::ShapeOp(SimulationConfig config, Model* model,
                 onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core),
      _axis(0) {

    const std::string& optype = node_proto.op_type();

    // ── read axis attribute (Flatten only) ───────────────────────────────
    if (optype == "Flatten") {
        for (const auto& attr : node_proto.attribute()) {
            if (attr.name() == "axis") {
                _axis = static_cast<uint32_t>(attr.i());
                break;
            }
        }
        if (_axis == 0) _axis = 1;  // ONNX default for Flatten is 1
    }
    
    // ── input shape ──────────────────────────────────────────────────────
    // Constant / ConstantOfShape have no data input – guard accordingly.
    if (optype == "Constant" || optype == "ConstantOfShape") {
        _input_shape = {1};
    } else {
        _input_shape = get_input(0)->get_dims();
    }

    _dk = shape_flatten(_input_shape);

    // ── output shape ─────────────────────────────────────────────────────
    resolve_output_shape(node_proto);

    // ── register output tensor ───────────────────────────────────────────
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

ShapeOp::ShapeOp(SimulationConfig config, Model* model,
                 std::string name,
                 std::map<std::string, std::string>& attributes,
                 uint32_t target_core)
    : Operation(config, model, name, attributes, target_core),
      _axis(0) {

    // Programmatic construction – parse axis from attributes if present.
    auto it = attributes.find("axis");
    if (it != attributes.end())
        _axis = static_cast<uint32_t>(std::stoi(it->second));

    // TODO: populate _input_shape / _output_shape / _dk from attributes.
}

ShapeOp::ShapeOp(const ShapeOp& src)
    : Operation(src),
      _input_shape(src._input_shape),
      _output_shape(src._output_shape),
      _dk(src._dk),
      _tokens_per_tile(src._tokens_per_tile),
      _axis(src._axis) {}

// ─────────────────────────────────────────────────────────────────────────────
// Tiling
// ─────────────────────────────────────────────────────────────────────────────

void ShapeOp::calculate_loops() {
    // Use half the scratchpad: one half for input, one half for output.
    uint32_t sram_capacity =
        _config.core_config[target_core].spad_size KB / 2;

    _tokens_per_tile = sram_capacity / _config.precision;
    if (_tokens_per_tile == 0) _tokens_per_tile = 1;
    if (_tokens_per_tile > _dk) _tokens_per_tile = _dk;

    // spdlog::info("[ShapeOp:{}] dk={} tokens_per_tile={}",
    //              get_name(), _dk, _tokens_per_tile);
}

void ShapeOp::initialize_tiles(MappingTable& mapping_table) {
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

void ShapeOp::initialize_instructions(Tile*    tile,
                                       Mapping  mapping,
                                       uint32_t token_offset,
                                       uint32_t tokens) {
    addr_type sram_base   = SPAD_BASE;
    addr_type input_addr  = get_operand_addr(_INPUT_OPERAND);
    addr_type output_addr = get_operand_addr(_OUTPUT_OPERAND);

    // ── collect DRAM addresses ───────────────────────────────────────────
    std::set<addr_type> dram_in_addrs;
    std::set<addr_type> dram_out_addrs;

    uint32_t tile_bytes = tokens * _config.precision;
    uint32_t base_off   = token_offset * _config.precision;

    for (int off = 0; off < (int)tile_bytes; off += (int)_config.dram_req_size) {
        dram_in_addrs .insert(input_addr  + base_off + off);
        dram_out_addrs.insert(output_addr + base_off + off);
    }

    // ── MOVIN: load input tile ───────────────────────────────────────────
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode     = Opcode::MOVIN,
        .dest_addr  = sram_base,
        .size       = (uint32_t)dram_in_addrs.size(),
        .src_addrs  = std::vector<addr_type>(dram_in_addrs.begin(),
                                             dram_in_addrs.end()),
        .operand_id = _INPUT_OPERAND,
    }));

    // ── NO COMP instruction – pure data movement ─────────────────────────

    // ── MOVOUT: write tile to output DRAM location ───────────────────────
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode     = Opcode::MOVOUT,
        .dest_addr  = sram_base,
        .size       = (uint32_t)dram_out_addrs.size(),
        .src_addrs  = std::vector<addr_type>(dram_out_addrs.begin(),
                                             dram_out_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
    }));
}