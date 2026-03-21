#pragma once
#include "Operation.h"
#include <string>
#include <unordered_map>

class MambaBlock : public Operation {
public:
    MambaBlock(SimulationConfig  config,
               Model*            model,
               onnx::NodeProto&  node_proto,
               uint32_t          target_core);

    void initialize_tiles(MappingTable& mapping_table) override;

private:
    void initialize_instructions(Tile* tile);

    // ── derived dimensions ────────────────────────────────────────────────────
    uint32_t _batch;
    uint32_t _d_model;
    uint32_t _d_inner;
    uint32_t _d_conv;
    uint32_t _d_state;
    uint32_t _dt_rank;

    // ── shapes ────────────────────────────────────────────────────────────────
    std::vector<uint32_t> _input_shape;
    std::vector<uint32_t> _conv_state_shape;
    std::vector<uint32_t> _ssm_state_shape;
    std::vector<uint32_t> _output_shape;

    // ── last instruction id per stage (for cross-tile instruction deps) ───────
    // Only instructions have dependent_ids — Tile has no parent/child fields.
    std::unordered_map<std::string, std::string> _stage_last_instr_id;

    // ── helpers ───────────────────────────────────────────────────────────────
    inline uint32_t bytes(uint32_t elems) const {
        return elems * _config.precision;
    }
};