import torch


def map_table_index(x, table_size=100):
    # Map the input x to the closest quantized input value
    quantized_x = torch.round((x + 1) * (table_size - 1) / 2).long()
    quantized_x = torch.clamp(quantized_x, 0, table_size - 1)
    return quantized_x


def map_weight_index(w, steps=256):
    # Map the weight w to the closest quantized weight index
    quantized_w = torch.round((w + 1) * (steps - 1) / 2).long()
    quantized_w = torch.clamp(quantized_w, 0, steps - 1)
    return quantized_w
