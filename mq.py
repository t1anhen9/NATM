from torchvision import transforms
from torchvision.datasets import CIFAR100

transform = transforms.Compose([
    transforms.Normalize(mean = [0.5071, 0.4867, 0.4408], std = [0.2675, 0.2565, 0.2761]),
    transforms.ToTensor()])
dataset = CIFAR100('./dataset/cifar100', train=True, download=True, transform=transform)


print(dataset.data)
print(dataset.data[5])

print(type(dataset.data))
print(type(dataset.data[5]))