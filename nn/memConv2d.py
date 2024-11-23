import torch
import torch.nn.functional as F
import torch.nn as nn

from ref.memristor import V_STEPS, G_STEPS
from utils.map import map_table_index, map_weight_index

class memConv2d(nn.Conv2d):
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
        self.lookup_table = lookup_table

        # Initialize the weights (quantized)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Extract input patches using F.unfold, which flattens patches into vectors
        # Input: (batch_size, in_channels, height, width)
        # Output: (batch_size, in_channels * kernel_size * kernel_size, num_patches)
        x_unfold = F.unfold(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )

        # Shape of x_unfold: (batch_size, in_channels * kernel_size^2, num_patches)
        batch_size, num_features, num_patches = x_unfold.shape
        patch_size = self.kernel_size * self.kernel_size

        # Reshape the input patches to: (batch_size, num_patches, in_channels, kernel_size * kernel_size)
        x_unfold = x_unfold.view(
            batch_size, num_features // patch_size, patch_size, num_patches
        )

        # Quantize the weights to the number of steps (e.g., 256 levels between [-1, 1])
        quantized_weights = map_weight_index(self.weight, self.steps)
        quantized_weights = quantized_weights.view(
            1, self.out_channels, self.in_channels, patch_size
        ).expand(batch_size, -1, -1, -1)

        # Quantize the inputs to the table_size (e.g., 100 levels between [-1, 1])
        quantized_inputs = map_table_index(x_unfold, self.table_size)

        # Perform lookup table operations for each patch
        output = torch.zeros(
            batch_size, self.out_channels, num_patches, device=x.device
        )

        # Iterate through the output channels
        for o_channel in range(self.out_channels):  # Output channels
            # Iterate through input channels
            for i_channel in range(self.in_channels):  # Input channels
                for patch_id in range(num_patches):  # Iterate through each patch size
                    # Quantized inputs shape:  torch.Size([64, 6, 25, 64])
                    # Quantized weights shape:  torch.Size([64, 16, 6, 25])
                    # Get the current quantized weight for the given output channel and patch
                    weight_value = quantized_weights[
                        :, o_channel, i_channel, :
                    ]  # Shape: [(64, 25)]

                    # Ensure weight_value is within valid range for lookup_table
                    weight_value = weight_value.clamp(
                        0, self.lookup_table.size - 1
                    ).long()  # Shape: [64]

                    # Access the quantized input values for this input channel and all patches
                    quantized_input_values = quantized_inputs[
                        :, i_channel, :, patch_id
                    ].squeeze(-1)  # Shape: [64, 25]

                    # Prepare the input for the lookup table
                    # Ensure weight_value is of shape [64, 1] for proper broadcasting
                    # weight_value_expanded = weight_value.unsqueeze(1)  # Shape: [64, 1]

                    # Perform lookup using weight_value and quantized_input_values
                    # Accessing the lookup table
                    table_values = self.lookup_table[
                        weight_value, quantized_input_values
                    ]

                    # Now we need to accumulate across the second dimension of table_values
                    # Sum the table values across the patch dimension (dim=1)
                    summed_values = table_values.sum()  # Shape: [64]

                    # Accumulate values into the output tensor for this output channel and patch
                    output[:, o_channel, patch_id] += summed_values

        # Convert the output tensor to a numpy array
        # output_np = output.detach().cpu().numpy()

        # Save the numpy array to a CSV file
        # np.savetxt("output.csv", output_np.reshape(batch_size, -1), delimiter=",")

        out_frame_size = (
            x.size(2) + 2 * self.padding[0] - self.kernel_size
        ) // self.stride[0] + 1

        output = output.reshape(
            batch_size, self.out_channels, out_frame_size, out_frame_size
        )

        # Add bias
        output += self.bias.view(1, self.out_channels, 1, 1)

        return output
