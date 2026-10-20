import torch
import torch.nn as nn

from dataset import BATCH_SIZE, load_dataset
from modules import ConvNeXtClassifier

EPOCHS = 300
LEARNING_RATE = 1e-4


def train_model(model, train_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def main():
    train_loader = load_dataset("train", batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(
        f"Training on {device}: {len(train_loader.dataset)} training images, "
        f"{EPOCHS} epochs, batch size {BATCH_SIZE}."
    )

    for epoch in range(1, EPOCHS + 1):
        epoch_loss, epoch_accuracy = train_model(
            model, train_loader, loss_function, optimizer, device
        )
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {epoch_loss:.4f} | "
            f"train accuracy: {epoch_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()
