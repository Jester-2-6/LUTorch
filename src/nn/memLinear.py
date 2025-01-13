import torch
import torch.nn as nn
from torch.autograd import Function

from ref.memristor import V_STEPS, G_STEPS
from utils.map import map_table_index, map_weight_index


class MemLinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lookup_table, steps, table_size):
        # Save context for backward
        ctx.save_for_backward(x, weight, bias, lookup_table)
        ctx.steps = steps
        ctx.table_size = table_size

        # Quantize weights to steps
        quantized_weights = torch.round((weight + 1) * (steps - 1) / 2).long()
        quantized_weights = torch.clamp(quantized_weights, 0, steps - 1)

        # Quantize inputs to table_size
        quantized_inputs = torch.round((x + 1) * (table_size - 1) / 2).long()
        quantized_inputs = torch.clamp(quantized_inputs, 0, table_size - 1)

        # Fetch values from the lookup table
        output = torch.zeros(x.size(0), weight.size(0), device=x.device)
        for i in range(weight.size(0)):  # Iterate over output features
            for j in range(weight.size(1)):  # Iterate over input features
                table_values = lookup_table[
                    quantized_weights[i, j], quantized_inputs[:, j]
                ]
                output[:, i] += table_values

        # Add bias
        output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, lookup_table = ctx.saved_tensors
        steps = ctx.steps
        table_size = ctx.table_size

        # Recompute quantized weights and inputs
        quantized_weights = torch.round((weight + 1) * (steps - 1) / 2).long()
        quantized_weights = torch.clamp(quantized_weights, 0, steps - 1)
        quantized_inputs = torch.round((x + 1) * (table_size - 1) / 2).long()
        quantized_inputs = torch.clamp(quantized_inputs, 0, table_size - 1)

        # Use differentiable operations to compute gradients
        grad_x = grad_output @ weight
        grad_weight = grad_output.t() @ x
        grad_bias = grad_output.sum(0)
        grad_lookup_table = None

        return grad_x, grad_weight, grad_bias, grad_lookup_table, None, None


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
        self.lookup_table = lookup_table.detach()
        self.lookup_table.requires_grad = False

        # Initialize the weights (quantized to a discrete range)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return MemLinearFunction.apply(
            x, self.weight, self.bias, self.lookup_table, self.steps, self.table_size
        )
