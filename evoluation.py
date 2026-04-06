import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from model_arch import DigitClassifier

# DEVICE
device = torch.device("cpu")

# MODEL 
model = DigitClassifier()

model.load_state_dict(torch.load("best_model_zehra.pth", map_location=device))
model.eval()


# DATA
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# PREDICTIONS
all_preds, all_labels, all_images = [], [], []

with torch.no_grad():
    for images, labels in test_loader:

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        all_images.extend(images)

# METRICS
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = (all_preds == all_labels).mean() * 100
print("Accuracy:", acc)

# CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix ({acc:.2f}%)")
plt.show()

# REPORT
print(classification_report(all_labels, all_preds))

# MISCLASSIFIED
wrong = np.where(all_preds != all_labels)[0]

fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    if i >= len(wrong):
        ax.axis("off")
        continue

    idx = wrong[i]
    ax.imshow(all_images[idx].squeeze(), cmap="gray")
    ax.set_title(f"T:{all_labels[idx]} P:{all_preds[idx]}", color="red")
    ax.axis("off")

plt.tight_layout()
plt.show()