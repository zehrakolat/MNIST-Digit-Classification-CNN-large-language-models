import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


# TRANSFORM (FIXED)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])


# DATASET
full_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# SPLIT
train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
remaining = len(full_dataset) - train_size - val_size

train_subset, val_subset, _ = random_split(
    full_dataset,
    [train_size, val_size, remaining]
)

# DATALOADERS
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# VISUAL CHECK
def show_samples(dataset):
    plt.figure(figsize=(12, 3))
    for i in range(10):
        img, label = dataset[i]
        plt.subplot(1, 10, i+1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(label)
        plt.axis("off")
    plt.show()

show_samples(full_dataset)

print("Train:", len(train_subset))
print("Val:", len(val_subset))
print("Test:", len(test_dataset))