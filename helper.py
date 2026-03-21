import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnx

# model_name = "state-spaces/mamba-130m-hf"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# base_model = AutoModelForCausalLM.from_pretrained(model_name)

# base_model.eval()
# base_model.config.use_cache = False


# class MambaWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, input_ids, attention_mask):
#         out = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             use_cache=False,
#             return_dict=True
#         )
#         return out.logits


# model = MambaWrapper(base_model)
# model.eval()

# dummy = tokenizer("Hello world", return_tensors="pt")

# input_ids = dummy["input_ids"]
# attention_mask = dummy["attention_mask"]

# seq_len = 1
# batch = 1

# dummy_input_ids = torch.ones(batch, seq_len, dtype=torch.long)
# dummy_attention = torch.ones(batch, seq_len, dtype=torch.long)

# torch.onnx.export(
#     model,
#     (dummy_input_ids, dummy_attention),
#     "models/mamba/mamba.onnx",
#     input_names=["input_ids", "attention_mask"],
#     output_names=["logits"],
#     opset_version=18,
#     dynamo=False
#)

import onnx

m = onnx.load("models/mamba-130m-single/mamba-130m-single.onnx")

# build shape lookup
shapes = {}
for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    shapes[vi.name] = dims

for init in m.graph.initializer:
    shapes[init.name] = list(init.dims)

# print all Gemm/MatMul nodes with their shapes
for node in m.graph.node:
    if node.op_type in ("Gemm", "MatMul"):
        print(f"\n{node.op_type}  name={node.name}")
        for i, inp in enumerate(node.input):
            s = shapes.get(inp, "UNKNOWN")
            flag = " ← 1D PROBLEM" if isinstance(s, list) and len(s) < 2 else ""
            print(f"  input[{i}] '{inp}': {s}{flag}")
        for i, out in enumerate(node.output):
            s = shapes.get(out, "UNKNOWN")
            print(f"  output[{i}] '{out}': {s}")