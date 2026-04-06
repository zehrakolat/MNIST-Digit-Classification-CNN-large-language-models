import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt

from Data_Preprocessing import train_loader, val_loader
from model_arch import DigitClassifier

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DigitClassifier().to(device)

# LOSS + OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# EARLY STOPPING
best_val_loss = float("inf")
patience = 5
counter = 0

train_losses, val_losses = [], []

print("Training started...")

# TRAIN LOOP
for epoch in range(30):

    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        train_loss += loss.item()

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_loss += criterion(model(images), labels).item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)

    train_losses.append(avg_train)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    scheduler.step()

    # SAVE BEST MODEL
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        counter = 0
        torch.save(model.state_dict(), "best_model_zehra.pth")
        print("Saved best model!")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# LOSS GRAPH
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.title("Training Curve")
plt.show()

print("Training finished.")