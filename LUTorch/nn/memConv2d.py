import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function

from LUTorch.ref.memristor import V_STEPS, G_STEPS
from LUTorch.utils.map import map_table_index, map_weight_index


class memConv2dFunc(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lookup_table, stride, padding, steps, table_size):
        ctx.save_for_backward(x, weight, bias, lookup_table)
        ctx.steps = steps
        ctx.table_size = table_size
        ctx.stride = stride
        ctx.padding = padding

        x_unfold = F.unfold(
            x, kernel_size=weight.shape[2], stride=stride, padding=padding
        )

        # Shape of x_unfold: (batch_size, in_channels * kernel_size^2, num_patches)
        batch_size, num_features, num_patches = x_unfold.shape
        patch_size = weight.shape[2] * weight.shape[3]

        # Reshape the input patches to: (batch_size, num_patches, in_channels, kernel_size * kernel_size)
        x_unfold = x_unfold.view(
            batch_size, num_features // patch_size, patch_size, num_patches
        )

        # Quantize the weights to the number of steps (e.g., 256 levels between [-1, 1])
        quantized_weights = map_weight_index(weight, steps)
        quantized_weights = quantized_weights.view(
            1, weight.shape[0], weight.shape[1], patch_size
        ).expand(batch_size, -1, -1, -1)

        # Quantize the inputs to the table_size (e.g., 100 levels between [-1, 1])
        quantized_inputs = map_table_index(x_unfold, table_size)

        # Perform lookup table operations for each patch
        output = torch.zeros(batch_size, weight.shape[0], num_patches, device=x.device)

        # Iterate through the output channels
        for o_channel in range(weight.shape[0]):  # Output channels
            # Iterate through input channels
            for i_channel in range(weight.shape[1]):  # Input channels
                for patch_id in range(num_patches):  # Iterate through each patch size
                    # Quantized inputs shape:  torch.Size([64, 6, 25, 64])
                    # Quantized weights shape:  torch.Size([64, 16, 6, 25])
                    # Get the current quantized weight for the given output channel and patch
                    weight_value = quantized_weights[
                        :, o_channel, i_channel, :
                    ]  # Shape: [(64, 25)]

                    # Ensure weight_value is within valid range for lookup_table
                    weight_value = weight_value.clamp(
                        0, lookup_table.size(0) - 1
                    ).long()  # Shape: [64]

                    # Access the quantized input values for this input channel and all patches
                    quantized_input_values = quantized_inputs[
                        :, i_channel, :, patch_id
                    ].squeeze(-1)  # Shape: [64, 25]

                    # Prepare the input for the lookup table
                    # Ensure weight_value is of shape [64, 1] for proper broadcasting

                    # Perform lookup using weight_value and quantized_input_values
                    # Accessing the lookup table
                    table_values = lookup_table[weight_value, quantized_input_values]

                    # Now we need to accumulate across the second dimension of table_values
                    # Sum the table values across the patch dimension (dim=1)
                    summed_values = table_values.sum(dim=1)  # Shape: [64]

                    # Accumulate values into the output tensor for this output channel and patch
                    output[:, o_channel, patch_id] += summed_values

        out_frame_size = (x.size(2) + 2 * padding - weight.shape[2]) // stride + 1

        output = output.reshape(
            batch_size, weight.shape[0], out_frame_size, out_frame_size
        )

        # Add bias
        output += bias.view(1, weight.shape[0], 1, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, _ = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding) 
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


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
