import os
import sys
import torch
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from LUTorch.nn import MemLinearFunction, memLinear, memReLu, memConv2d
from LUTorch.ref.lut_loader import load_lut
from LUTorch.ref.memristor import V_STEPS, G_STEPS
from LUTorch.ref.config import TOLERANCE
import sys


class TestMemLinearFunction(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [[0.5, -0.5], [0.3, -0.3]], dtype=torch.float32, requires_grad=True
        )
        self.weight = torch.tensor(
            [[0.2, -0.2], [0.4, -0.4]], dtype=torch.float32, requires_grad=True
        )
        self.bias = torch.tensor([0.1, -0.1], dtype=torch.float32)
        self.lookup_table = torch.tensor(load_lut(), dtype=torch.float32)
        self.steps = G_STEPS
        self.table_size = V_STEPS

    def test_forward_with_bias(self):
        output = MemLinearFunction.apply(
            self.x,
            self.weight,
            self.bias,
            self.lookup_table,
            self.steps,
            self.table_size,
        )
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(torch.is_tensor(output))

    def test_forward_values(self):
        output = MemLinearFunction.apply(
            self.x,
            self.weight,
            self.bias,
            self.lookup_table,
            self.steps,
            self.table_size,
        )
        expected_output = self.x @ self.weight.T + self.bias
        print(output, expected_output)
        self.assertTrue(torch.allclose(output, expected_output, atol=TOLERANCE))


class TestMemLinear(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[0.5, -0.5], [0.3, -0.3]], dtype=torch.float32)
        self.weight = torch.tensor([[0.2, -0.2], [0.4, -0.4]], dtype=torch.float32)
        self.bias = torch.tensor([0.1, -0.1], dtype=torch.float32)
        self.lookup_table = torch.tensor(load_lut(), dtype=torch.float32)
        self.steps = G_STEPS
        self.table_size = V_STEPS

        self.torch_linear = torch.nn.Linear(2, 2)
        self.torch_linear.weight.data = self.weight
        self.torch_linear.bias.data = self.bias

        self.mem_linear = memLinear(2, 2, self.lookup_table)
        self.mem_linear.weight.data = self.weight
        self.mem_linear.bias.data = self.bias

    def test_forward(self):
        torch_output = self.torch_linear(self.x)
        mem_output = self.mem_linear(self.x)
        self.assertTrue(torch.allclose(torch_output, mem_output, atol=TOLERANCE))

    def test_backward(self):
        torch_output = self.torch_linear(self.x)
        mem_output = self.mem_linear(self.x)
        torch_output.sum().backward()
        mem_output.sum().backward()
        self.assertTrue(
            torch.allclose(
                self.torch_linear.weight.grad,
                self.mem_linear.weight.grad,
                atol=TOLERANCE,
            )
        )
        self.assertTrue(
            torch.allclose(
                self.torch_linear.bias.grad, self.mem_linear.bias.grad, atol=TOLERANCE
            )
        )


class TestMemReLU(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [[0.5, -0.5], [0.3, -0.3]], dtype=torch.float32, requires_grad=True
        )
        self.relu = torch.nn.ReLU()
        self.mem_relu = memReLu()

    def test_forward(self):
        torch_output = self.relu(self.x)
        mem_output = self.mem_relu(self.x)
        self.assertTrue(torch.allclose(torch_output, mem_output, atol=TOLERANCE))

    def test_backward(self):
        torch_output = self.relu(self.x)
        mem_output = self.mem_relu(self.x)
        torch_output.sum().backward()
        mem_output.sum().backward()
        self.assertTrue(torch.allclose(self.x.grad, self.x.grad, atol=TOLERANCE))


class TestMemConv2d(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(1, 2, 4, 4, dtype=torch.float32) / 3
        self.weight = torch.randn(1, 2, 2, 2, dtype=torch.float32)
        self.bias = torch.tensor([0.1], dtype=torch.float32)
        self.lookup_table = torch.tensor(load_lut(), dtype=torch.float32)
        self.steps = G_STEPS
        self.table_size = V_STEPS

        self.torch_conv2d = torch.nn.Conv2d(2, 1, 2)
        self.torch_conv2d.weight.data = self.weight
        self.torch_conv2d.bias.data = self.bias

        self.mem_conv2d = memConv2d(2, 1, 2, self.lookup_table)
        self.mem_conv2d.weight.data = self.weight
        self.mem_conv2d.bias.data = self.bias

    def test_forward(self):
        torch_output = self.torch_conv2d(self.x)
        mem_output = self.mem_conv2d(self.x)

        print(torch_output)
        print(mem_output)
        self.assertTrue(torch.allclose(torch_output, mem_output, atol=TOLERANCE))

    def test_backward(self):
        torch_output = self.torch_conv2d(self.x)
        mem_output = self.mem_conv2d(self.x)
        torch_output.sum().backward()
        mem_output.sum().backward()
        self.assertTrue(
            torch.allclose(
                self.torch_conv2d.weight.grad,
                self.mem_conv2d.weight.grad,
                atol=TOLERANCE,
            )
        )
        self.assertTrue(
            torch.allclose(
                self.torch_conv2d.bias.grad, self.mem_conv2d.bias.grad, atol=TOLERANCE
            )
        )


if __name__ == "__main__":
    unittest.main()
