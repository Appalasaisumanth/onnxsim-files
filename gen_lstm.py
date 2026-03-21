import torch
import torch.nn as nn

class LSTMCellManual(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Input weights
        self.W = nn.Parameter(torch.randn(input_size, 4 * hidden_size))

        # Hidden weights
        self.U = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size))

        # Bias
        self.b = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, x, h, c):

        gates = x @ self.W + h @ self.U + self.b

        i, f, g, o = torch.chunk(gates, 4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class LSTMModelManual(nn.Module):
    def __init__(self, input_size=512, hidden_size=1024):
        super().__init__()

        self.cell = LSTMCellManual(input_size, hidden_size)

    def forward(self, x, h, c):

        seq_len = x.shape[1]

        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]
            h, c = self.cell(xt, h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs, h, c

# Model config
input_size  = 512
hidden_size = 1024
batch_size  = 1
seq_len     = 10

model = LSTMModelManual(input_size, hidden_size)
model.eval()

x  = torch.randn(batch_size, seq_len, input_size)
h0 = torch.zeros(batch_size, hidden_size)
c0 = torch.zeros(batch_size, hidden_size)

torch.onnx.export(
    model,
    (x, h0, c0),
    "models/lstm/lstm.onnx",
    opset_version=18,
    input_names=["input", "h0", "c0"],
    output_names=["output", "hn", "cn"],
)

print("✅ lstm.onnx exported successfully")

# Verify
import onnx
model_onnx = onnx.load("models/lstm/lstm.onnx")
onnx.checker.check_model(model_onnx)
print("✅ ONNX model check passed")

import onnx

model = onnx.load("models/lstm/lstm.onnx")

for node in model.graph.node:
    print(node.op_type)