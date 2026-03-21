#include "Relu.h"
#include "../Model.h"

Relu::Relu(SimulationConfig config, Model* model,
           onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {

    /* Load input info from node */
    _input_shape = get_input(0)->get_dims();

    assert(_input_shape.size() == 3);
    _batch_size = 1;
    _seq = 1;
    _dk = 1;

    for(auto d : _input_shape)
    _dk *= d;
    
    _output_shape = _input_shape;

    Tensor* pre_defined_tensor = _model->find_tensor(node_proto.output(0));
    if (pre_defined_tensor == nullptr) {
        std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
            _id, node_proto.output(0), _output_shape, _config.precision, false);
        _outputs.push_back(output_tensor.get()->get_id());
        _model->add_tensor(std::move(output_tensor));
    } else {
        pre_defined_tensor->redefine_tensor(_id, _output_shape);
    }

    calculate_loops();
}

Relu::Relu(SimulationConfig config, Model* model,
           std::string name, std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
    // TODO: implement this
}

void Relu::initialize_tiles(MappingTable& mapping_table) {
    for (uint32_t tokens = 0; tokens < _seq * _batch_size; tokens += _tokens_per_tile) {
        uint32_t remain_tokens = std::min(_seq * _batch_size - tokens, _tokens_per_tile);

        std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
            .status   = Tile::Status::INITIALIZED,
            .optype   = get_name(),
            .layer_id = _id,
            .accum    = false,
        });

        Mapping mapping;
        _tiles.push_back(std::move(tile));
        initialize_instructions(_tiles.back().get(), mapping, tokens, remain_tokens);
    }
}

void Relu::initialize_instructions(Tile* tile, Mapping mapping,
                                   uint32_t token_offset, uint32_t tokens) {
    addr_type sram_base = SPAD_BASE;

    addr_type input_addr  = get_operand_addr(_INPUT_OPERAND);
    addr_type output_addr = get_operand_addr(_OUTPUT_OPERAND);

    /* Collect DRAM addresses for input and output */
    std::set<addr_type> dram_input_addrs;
    std::set<addr_type> dram_output_addrs;

    for (int offset = 0; offset < tokens * _dk * _config.precision; offset += _config.dram_req_size) {
        dram_input_addrs.insert(input_addr   + token_offset * _dk * _config.precision + offset);
        dram_output_addrs.insert(output_addr + token_offset * _dk * _config.precision + offset);
    }

    /* MOVIN: load input tile */
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode     = Opcode::MOVIN,
        .dest_addr  = sram_base,
        .size       = (uint32_t)dram_input_addrs.size(),
        .src_addrs  = std::vector<addr_type>(dram_input_addrs.begin(), dram_input_addrs.end()),
        .operand_id = _INPUT_OPERAND,
    }));

    /* COMP: element-wise ReLU  max(0, x) */
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode      = Opcode::COMP,
        .dest_addr   = sram_base,
        .size        = _dk * tokens * _config.precision / _config.dram_req_size,
        .compute_size = _dk * tokens * _config.precision,
        .src_addrs   = std::vector<addr_type>{sram_base},
    }));

    /* MOVOUT: write result back to DRAM */
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode     = Opcode::MOVOUT,
        .dest_addr  = sram_base,
        .size       = (uint32_t)dram_output_addrs.size(),
        .src_addrs  = std::vector<addr_type>(dram_output_addrs.begin(), dram_output_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
    }));
}

void Relu::calculate_loops() {
    uint32_t size_per_token = _config.precision;
    uint32_t sram_capacity = _config.core_config[target_core].spad_size KB / 2;
    _tokens_per_tile = sram_capacity / size_per_token;
    if (_tokens_per_tile > _dk)
    _tokens_per_tile = _dk;
    spdlog::info("[ReLU] tokens_per_tile: {}", _tokens_per_tile);
}