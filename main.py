from networks.LeNetMem import LeNetMem
from LUTorch.ref.lut_loader import load_lut
import torch
from torchvision import datasets, transforms

lut = torch.tensor(load_lut())
model = LeNetMem(lut)

# Load the model from a .pth file
model_path = 'weights/lenet_mnist.pth'
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Define the MNIST test dataset and dataloader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.MNIST('data/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Test the model on the MNIST dataset
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the MNIST test images: {100 * correct / total}%')
