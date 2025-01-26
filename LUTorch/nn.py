import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function

from LUTorch.ref.memristor import V_STEPS, G_STEPS
from LUTorch.utils.map import map_table_index, map_weight_index


class MemLinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lookup_table, steps, table_size):
        # Save context for backward
        ctx.save_for_backward(x, weight, bias, lookup_table)
        ctx.steps = steps
        ctx.table_size = table_size

        # Quantize weights and inputs
        quantized_weights = torch.round((weight + 1) * (steps - 1) / 2).long()
        quantized_weights = torch.clamp(quantized_weights, 0, steps - 1)

        quantized_inputs = torch.round((x + 1) * (table_size - 1) / 2).long()
        quantized_inputs = torch.clamp(quantized_inputs, 0, table_size - 1)

        # Vectorized lookup table access
        weight_expanded = quantized_weights.unsqueeze(0).expand(x.size(0), -1, -1)
        input_expanded = quantized_inputs.unsqueeze(1).expand(-1, weight.size(0), -1)

        # Gather lookup table values
        table_values = lookup_table[weight_expanded, input_expanded]

        # Sum along input features
        output = table_values.sum(dim=2)

        # Add bias
        if bias is not None:
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


class memConv2dFunc(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lookup_table, stride, padding, steps, table_size):
        ctx.save_for_backward(x, weight, bias, lookup_table)
        ctx.steps = steps
        ctx.table_size = table_size
        ctx.stride = stride
        ctx.padding = padding

        # Unfold input into patches
        x_unfold = F.unfold(
            x, kernel_size=weight.shape[2], stride=stride, padding=padding
        )
        batch_size, num_features, num_patches = x_unfold.shape
        patch_size = weight.shape[2] * weight.shape[3]

        # Reshape input patches
        x_unfold = x_unfold.view(
            batch_size, num_features // patch_size, patch_size, num_patches
        )

        # Quantize weights and inputs
        quantized_weights = map_weight_index(weight, steps)
        quantized_weights = quantized_weights.view(
            1, weight.shape[0], weight.shape[1], patch_size
        )
        quantized_weights = quantized_weights.expand(batch_size, -1, -1, -1)
        quantized_weights = quantized_weights.clamp(0, lookup_table.size(0) - 1).long()

        quantized_inputs = map_table_index(x_unfold, table_size)
        quantized_inputs = quantized_inputs.clamp(0, lookup_table.size(1) - 1).long()

        # Prepare indices for vectorized lookup
        weight_indices = quantized_weights.unsqueeze(-1)  # (B, O, I, P, 1)
        input_indices = quantized_inputs.unsqueeze(1)  # (B, 1, I, P, N)

        # Perform batched lookups and sum over input channels and patch dimensions
        table_values = lookup_table[weight_indices, input_indices]
        output = table_values.sum(dim=(2, 3))  # Sum over I and P → (B, O, N)

        # Reshape output and add bias
        out_frame_size = (x.size(2) + 2 * padding - weight.shape[2]) // stride + 1
        output = output.view(
            batch_size, weight.shape[0], out_frame_size, out_frame_size
        )
        output += bias.view(1, -1, 1, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, _ = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_input = grad_weight = grad_bias = None

        # Compute gradients using PyTorch's built-in functions
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output, stride, padding
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output, stride, padding
            )
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class memLinear(nn.Linear):
    """
    A custom linear layer that uses a lookup table for quantized weights.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        lookup_table (torch.Tensor): A tensor containing the lookup table values.
        steps (int, optional): Number of steps for quantization. Default is G_STEPS.
        table_size (int, optional): Size of the lookup table. Default is V_STEPS.

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        steps (int): Number of steps for quantization.
        table_size (int): Size of the lookup table.
        lookup_table (torch.Tensor): A tensor containing the lookup table values.
        weight (torch.nn.Parameter): The learnable weights of the module.
        bias (torch.nn.Parameter): The learnable bias of the module.

    Methods:
        forward(x):
            Applies the linear transformation to the input data using the lookup table for quantized weights.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after applying the linear transformation.
    """

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


class memConv2d(nn.Conv2d):
    """
    A custom 2D convolutional layer that uses a lookup table for quantized weights.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        lookup_table (torch.Tensor): Precomputed lookup table for quantized weights.
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default is 0.
        steps (int, optional): Number of steps for quantization. Default is G_STEPS.
        table_size (int, optional): Size of the lookup table. Default is V_STEPS.

    Attributes:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (tuple): Stride of the convolution.
        padding (tuple): Zero-padding added to both sides of the input.
        steps (int): Number of steps for quantization.
        table_size (int): Size of the lookup table.
        lookup_table (torch.Tensor): Precomputed lookup table for quantized weights.
        weight (torch.nn.Parameter): Learnable weights of the module.
        bias (torch.nn.Parameter): Learnable bias of the module.

    Methods:
        forward(x):
            Applies the convolutional layer to the input tensor x using the lookup table for quantized weights.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            Returns:
                torch.Tensor: Output tensor after applying the convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        lookup_table,
        stride=1,
        padding=0,
        steps=G_STEPS,
        table_size=V_STEPS,
    ):
        super(memConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.steps = steps
        self.table_size = table_size
        self.lookup_table = lookup_table.detach()
        self.lookup_table.requires_grad = False

        # Initialize the weights (quantized)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Extract input patches using F.unfold, which flattens patches into vectors
        # Input: (batch_size, in_channels, height, width)
        # Output: (batch_size, in_channels * kernel_size * kernel_size, num_patches)
        return memConv2dFunc.apply(
            x,
            self.weight,
            self.bias,
            self.lookup_table,
            self.stride[0],
            self.padding[0],
            self.steps,
            self.table_size,
        )


class memReLu(nn.ReLU):
    def __init__(self):
        super(memReLu, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0, 1)
