import torch
import sys

# adding Folder_2 to the system path
sys.path.insert(0, "..")
from nn.memConv2d import memConv2d
from nn.memLinear import memLinear
from nn.memReLu import memReLu

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
