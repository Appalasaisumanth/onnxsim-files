#include "OperationFactory.h"

#include "AdaptiveAvgPool.h"
#include "Conv.h"
#include "ConvOS.h"
#include "ConvWS.h"
#include "Flatten.h"
#include "Gemm.h"
#include "GemmWS.h"
#include "GlobalAvgPool.h"
#include "Operation.h"
#include "Attention.h"
#include "Dummy.h"
#include "EmbedLayerNorm.h"
#include "SkipLayerNorm.h"
#include "BiasGelu.h"
#include "MaxPool.h"
#include "BinaryOp.h"
#include "UnaryOp.h"
#include "ShapeOp.h"
#include "MambaBlock.h"

SimulationConfig OperationFactory::_config = SimulationConfig();

void OperationFactory::initialize(SimulationConfig config) { _config = config; }

// ─────────────────────────────────────────────────────────────────────────────
// create_operation
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<Operation> OperationFactory::create_operation(
    Model* model, onnx::NodeProto& node_proto, uint32_t target_core) {

  const std::string& op = node_proto.op_type();

  // ── compute-heavy specialised ops ─────────────────────────────────────────
  if (op == "Conv" || op == "FusedConv") {
    if (_config.core_config[target_core].core_type == CoreType::SYSTOLIC_OS)
      return std::make_unique<ConvOS>(_config, model, node_proto, target_core);
    else if (_config.core_config[target_core].core_type == CoreType::SYSTOLIC_WS)
      return std::make_unique<ConvWS>(_config, model, node_proto, target_core);
  }

  if (op == "Gemm" || op == "FusedGemm") {
    if (_config.core_config[target_core].core_type == CoreType::SYSTOLIC_WS)
      return std::make_unique<GemmWS>(_config, model, node_proto, target_core);
  }

  if (op == "MatMul") {
    return std::make_unique<GemmWS>(_config, model, node_proto, false, target_core);
  }

  if (op == "MaxPool") {
    return std::make_unique<MaxPool>(_config, model, node_proto, target_core);
  }

  if (op == "GlobalAveragePool") {
    return std::make_unique<GlobalAvgPool>(_config, model, node_proto, target_core);
  }

  if (op == "AdaptiveAveragePool" || op == "AveragePool") {
    return std::make_unique<AdaptiveAvgPool>(_config, model, node_proto, target_core);
  }

  if (op == "Flatten") {
    return std::make_unique<Flatten>(_config, model, node_proto, target_core);
  }

  if (op == "Attention") {
    return std::make_unique<Attention>(_config, model, node_proto, target_core);
  }

  if (op == "Cast") {
    return std::make_unique<Dummy>(_config, model, node_proto, target_core);
  }

  if (op == "EmbedLayerNormalization" || op == "LayerNormalization") {
    return std::make_unique<EmbedLayerNorm>(_config, model, node_proto, target_core);
  }

  if (op == "SkipLayerNormalization" || op== "SkipLayerNorm") {
    return std::make_unique<SkipLayerNorm>(_config, model, node_proto, target_core);
  }

  if (op == "BiasGelu" || op == "FastGelu") {
    return std::make_unique<BiasGelu>(_config, model, node_proto, target_core);
  }

  if (op == "ReorderOutput") {
    return std::make_unique<Dummy>(_config, model, node_proto, target_core);
  }
   if (op == "MambaBlock") {
    return std::make_unique<MambaBlock>(_config, model, node_proto, target_core);
  }

  // ── shape / metadata ops (MOVIN + MOVOUT only, no COMP) ───────────────────
  // Note: Flatten already handled above by its own class.
  if (op == "Reshape"         ||
      op == "Transpose"       ||
      op == "Unsqueeze"       ||
      op==  "Slice"           ||
      op == "Squeeze"         ||
      op == "Expand"          ||
      op == "Gather"          ||
      op == "Constant"        ||
      op == "ConstantOfShape" ||
      op == "Concat"          ||
      op == "Identity") {
    return std::make_unique<ShapeOp>(_config, model, node_proto, target_core);
  }

  // ── unary element-wise arithmetic ops ─────────────────────────────────────
  // Note: Cast already handled above by Dummy.
  if (op == "Relu"    ||
      op == "Erf"     ||
      op == "Tanh"    ||
      op == "Sigmoid" ||
      op == "Neg"     ||
      op == "Abs"     ||
      op == "Sqrt"    ||
      op == "Exp"     ||
      op == "Silu" || op == "SiLU" ||
      op == "Softplus" ||
      op == "Gelu") {
    return std::make_unique<UnaryOp>(_config, model, node_proto, target_core);
  }

  // ── binary element-wise arithmetic ops ────────────────────────────────────
  if (op == "Add"   ||
      op == "Sub"   ||
      op == "Mul"   ||
      op == "Div"   ||
      op == "Equal" ||
      op == "Greater" ||  
      op == "Where") {
    return std::make_unique<BinaryOp>(_config, model, node_proto, target_core);
  }

  // ── fallback ──────────────────────────────────────────────────────────────
  spdlog::warn("Node Proto optype \"{}\" returned dummy operator!",
               node_proto.op_type().c_str());
  return std::make_unique<Dummy>(_config, model, node_proto, target_core);
}

// ─────────────────────────────────────────────────────────────────────────────
// copy_operation
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<Operation> OperationFactory::copy_operation(Operation* op) {

  const std::string& optype = op->get_optype();

  // ── compute-heavy specialised ops ─────────────────────────────────────────
  if (optype == "Conv" || optype == "FusedConv") {
    if (_config.core_config[op->target_core].core_type == CoreType::SYSTOLIC_OS)
      return std::make_unique<ConvOS>(*dynamic_cast<ConvOS*>(op));
    else if (_config.core_config[op->target_core].core_type == CoreType::SYSTOLIC_WS)
      return std::make_unique<ConvWS>(*dynamic_cast<ConvWS*>(op));
  }

  if (optype == "Gemm" || optype == "FusedGemm") {
    if (_config.core_config[op->target_core].core_type == CoreType::SYSTOLIC_WS)
      return std::make_unique<GemmWS>(*dynamic_cast<GemmWS*>(op));
  }

  if (optype == "MatMul") {
    return std::make_unique<GemmWS>(*dynamic_cast<GemmWS*>(op));
  }

  if (optype == "MaxPool") {
    return std::make_unique<MaxPool>(*dynamic_cast<MaxPool*>(op));
  }

  if (optype == "GlobalAveragePool") {
    return std::make_unique<GlobalAvgPool>(*dynamic_cast<GlobalAvgPool*>(op));
  }
    if (optype == "MambaBlock") {
    return std::make_unique<MambaBlock>(*dynamic_cast<MambaBlock*>(op));
  }

  if (optype == "AdaptiveAveragePool" || optype == "AveragePool") {
    return std::make_unique<AdaptiveAvgPool>(*dynamic_cast<AdaptiveAvgPool*>(op));
  }

  if (optype == "Flatten") {
    return std::make_unique<Flatten>(*dynamic_cast<Flatten*>(op));
  }

  if (optype == "Attention") {
    return std::make_unique<Attention>(*dynamic_cast<Attention*>(op));
  }

  if (optype == "Cast") {
    return std::make_unique<Dummy>(*dynamic_cast<Dummy*>(op));
  }

  if (optype == "EmbedLayerNormalization" || optype == "LayerNormalization") {
    return std::make_unique<EmbedLayerNorm>(*dynamic_cast<EmbedLayerNorm*>(op));
  }

  if (optype == "SkipLayerNormalization" || optype == "SkipLayerNorm") {
    return std::make_unique<SkipLayerNorm>(*dynamic_cast<SkipLayerNorm*>(op));
  }

  if (optype == "BiasGelu" || optype == "FastGelu") {
    return std::make_unique<BiasGelu>(*dynamic_cast<BiasGelu*>(op));
  }

  if (optype == "ReorderOutput") {
    return std::make_unique<Dummy>(*dynamic_cast<Dummy*>(op));
  }

  // ── shape / metadata ops ──────────────────────────────────────────────────
  if (optype == "Reshape"         ||
      optype == "Transpose"       ||
      optype == "Unsqueeze"       ||
      optype == "Squeeze"         ||
      optype == "Slice"           ||
      optype == "Expand"          ||
      optype == "Gather"          ||
      optype == "Concat"          ||
      optype == "Constant"        ||
      optype == "ConstantOfShape" ||
      optype == "Identity") {
    return std::make_unique<ShapeOp>(*dynamic_cast<ShapeOp*>(op));
  }

  // ── unary element-wise arithmetic ops ─────────────────────────────────────
  if (optype == "Relu"    ||
      optype == "Erf"     ||
      optype == "Tanh"    ||
      optype == "Sigmoid" ||
      optype == "Neg"     ||
      optype == "Abs"     ||
      optype == "Sqrt"    ||
      optype == "Exp"     ||
      optype == "Silu" || optype == "SiLU" ||
      optype == "Softplus" ||
      optype == "Gelu") {
    return std::make_unique<UnaryOp>(*dynamic_cast<UnaryOp*>(op));
  }

  // ── binary element-wise arithmetic ops ────────────────────────────────────
  if (optype == "Add"   ||
      optype == "Sub"   ||
      optype == "Mul"   ||
      optype == "Div"   ||
      optype == "Equal" ||
      optype == "Where") {
    return std::make_unique<BinaryOp>(*dynamic_cast<BinaryOp*>(op));
  }

  // ── fallback ──────────────────────────────────────────────────────────────
  spdlog::warn("Node Proto optype \"{}\" returned dummy operator!", optype);
  return std::make_unique<Dummy>(*dynamic_cast<Dummy*>(op));
}