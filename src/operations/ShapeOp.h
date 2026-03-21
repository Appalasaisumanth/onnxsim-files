#pragma once
#include "Operation.h"

/**
 * ShapeOp
 * Handles all shape-manipulation / metadata-only ONNX operations:
 *   Reshape, Transpose, Unsqueeze, Squeeze, Expand,
 *   Gather, Flatten, Constant, ConstantOfShape, Identity
 *
 * These ops produce NO arithmetic – they only move data from DRAM
 * into the scratchpad and back out (MOVIN + MOVOUT only).
 *
 * The _axis field is used by Flatten (and ignored by all other ops).
 */
class ShapeOp : public Operation {
public:
    ShapeOp(SimulationConfig config, Model* model,
            onnx::NodeProto& node_proto, uint32_t target_core = 0);

    ShapeOp(SimulationConfig config, Model* model,
            std::string name,
            std::map<std::string, std::string>& attributes,
            uint32_t target_core = 0);

    ShapeOp(const ShapeOp& src);

    void initialize_tiles(MappingTable& mapping_table) override;
    void initialize_instructions(Tile* tile, Mapping mapping,
                                 uint32_t token_offset, uint32_t tokens);

private:
    /* ── shape bookkeeping ───────────────────────────────────────────────── */
    std::vector<uint32_t> _input_shape;
    std::vector<uint32_t> _output_shape;

    uint32_t _dk;               // flattened element count
    uint32_t _tokens_per_tile;

    /* axis attribute – only meaningful for Flatten, 0 otherwise */
    uint32_t _axis;

    /* ── helpers ─────────────────────────────────────────────────────────── */
    void calculate_loops();

    /**
     * Derive _output_shape from the node.
     * For Constant / ConstantOfShape there is no input tensor;
     * we read the shape from the node attributes instead.
     */
    void resolve_output_shape(onnx::NodeProto& node_proto);
};