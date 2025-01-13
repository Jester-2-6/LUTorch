# LUTorch
Python Library to Simulate Memristor Crossbar Array Networks. Based on PyTorch

## Installation

To install LUTorch, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/LUTorch.git
cd LUTorch
pip install -r requirements.txt
```

## Usage

Here's a simple example to get you started, using a LeNet model:

```python
import torch

from LUTorch.nn.memConv2d import memConv2d
from LUTorch.nn.memLinear import memLinear
from LUTorch.nn.memReLu import memReLu

class LeNetMem(torch.nn.Module):
    def __init__(self, lut):
        super(LeNetMem, self).__init__()
        self.conv1 = memConv2d(1, 6, 5, lut)
        self.relu = memReLu()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = memConv2d(6, 16, 5, lut)
        self.fc1 = memLinear(16 * 4 * 4, 120, lut)
        self.fc2 = memLinear(120, 84, lut)
        self.fc3 = memLinear(84, 10, lut)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
```

## Features

- Simulate memristor crossbar arrays
- Integrate with PyTorch for advanced neural network simulations

## Note

- Always use `memReLu` instead of `torch.nn.ReLU` to simulate memristor crossbar arrays, or you may encounter overflow issues since all values should be contained in the range of 0 to 1.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
