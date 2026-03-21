#pragma once
#include "Operation.h"

/**
 * UnaryOp
 * Each op emits the EXACT primitive instruction sequence that implements it:
 *
 *  Relu     : COMP                          (max(0,x))
 *  Neg      : COMP                          (-x)
 *  Abs      : COMP                          (|x|)
 *  Sqrt     : COMP                          (√x)
 *  Exp      : EXP                           (e^x)
 *  Gelu     : GELU                          (x·Φ(x))
 *  Tanh     : EXP(2x) → ADD(+1,−1) → DIV   ((e^2x−1)/(e^2x+1))
 *  Sigmoid  : EXP(−x) → ADD(+1)    → DIV   (1/(1+e^−x))
 *  Erf      : MUL(x/√2) → EXP → MUL → ADD → DIV
 *             exact Abramowitz & Stegun rational decomposition:
 *             sign(x)·(1 − (a1·t+a2·t²+a3·t³)·e^(−x²))  t=1/(1+0.47047·|x|)
 *             mapped to: COMP(|x|,sign) → MUL → ADD → EXP → MUL → ADD → DIV
 *  Cast     : COMP                          (type reinterpret, 1 pass)
 */
class UnaryOp : public Operation {
public:
    UnaryOp(SimulationConfig config, Model* model,
            onnx::NodeProto& node_proto, uint32_t target_core = 0);

    UnaryOp(SimulationConfig config, Model* model,
            std::string name,
            std::map<std::string, std::string>& attributes,
            uint32_t target_core = 0);

    UnaryOp(const UnaryOp& src);

    std::vector<uint32_t> _input_shape;
    std::vector<uint32_t> _output_shape;

    uint32_t _batch_size;
    uint32_t _seq;
    uint32_t _dk;               // total element count (flat)
    uint32_t _tokens_per_tile;

    void calculate_loops();
    void initialize_tiles(MappingTable& mapping_table) override;

    /* emit the exact instruction sequence for one tile */
    void initialize_instructions(Tile* tile, Mapping mapping,
                                 uint32_t token_offset, uint32_t tokens);

private:
    /* ordered list of opcodes to emit for this op-type */
    std::vector<Opcode> _opcodes;

    void resolve_opcodes(const std::string& optype);
};