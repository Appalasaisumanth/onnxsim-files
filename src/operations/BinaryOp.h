#pragma once
#include "Operation.h"

/**
 * BinaryOp
 * Covers element-wise two-input ops:
 *   Add, Sub, Mul, Div, Equal, Where
 *
 * Both inputs are loaded into the scratchpad, the chosen opcode is applied,
 * and the result is written back to DRAM.
 *
 * Broadcasting is NOT modelled at the instruction level – we use the larger
 * of the two input shapes as the working element count (_dk), matching the
 * convention used by other ONNXim operations.
 */
class BinaryOp : public Operation {
public:
    BinaryOp(SimulationConfig config, Model* model,
             onnx::NodeProto& node_proto, uint32_t target_core = 0);

    BinaryOp(SimulationConfig config, Model* model,
             std::string name,
             std::map<std::string, std::string>& attributes,
             uint32_t target_core = 0);

    /* shape info */
    std::vector<uint32_t> _input_shape_a;   // first  operand
    std::vector<uint32_t> _input_shape_b;   // second operand
    std::vector<uint32_t> _output_shape;

    uint32_t _batch_size;
    uint32_t _seq;
    uint32_t _dk;               // flattened element count (larger of the two)
    uint32_t _tokens_per_tile;
    bool _broadcast_b;         // whether operand B is broadcast (single value)

    /* opcode emitted for the COMP stage */
    Opcode _comp_opcode;

    void calculate_loops();
    void initialize_tiles(MappingTable& mapping_table) override;
    void initialize_instructions(Tile* tile, Mapping mapping,
                                 uint32_t token_offset, uint32_t tokens);

private:
    void resolve_opcode(const std::string& optype);
};