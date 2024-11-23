import torch
import torch.nn as nn

from ref.memristor import V_STEPS, G_STEPS


class memLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        lookup_table,
        steps=G_STEPS,
        table_size=V_STEPS,
    ):
        super(memLinear, self).__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.steps = steps
        self.table_size = table_size
        self.lookup_table = lookup_table

        # Initialize the weights (quantized to a discrete range)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Quantize weights to steps (e.g., 256 steps between [-1, 1])
        quantized_weights = torch.round((self.weight + 1) * (self.steps - 1) / 2).long()
        quantized_weights = torch.clamp(quantized_weights, 0, self.steps - 1)

        # For each input in x, map to closest quantized input value
        quantized_inputs = torch.round((x + 1) * (self.table_size - 1) / 2).long()
        quantized_inputs = torch.clamp(quantized_inputs, 0, self.table_size - 1)

        # Fetch values from the lookup table based on quantized weights and inputs
        output = torch.zeros(x.size(0), self.out_features, device=x.device)
        for i in range(self.out_features):
            for j in range(self.in_features):
                # For each weight and input pair, get the precomputed value from the lookup table
                table_values = self.lookup_table[
                    quantized_weights[i, j], quantized_inputs[:, j]
                ]
                output[:, i] += table_values

        output += self.bias
        return output
