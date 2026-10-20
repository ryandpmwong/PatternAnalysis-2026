from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from dataset import BATCH_SIZE, CLASS_TO_LABEL, load_train_validation
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


def evaluate_model(model, validation_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    ad_correct = 0
    ad_total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss_function(logits, labels).item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size
            ad_mask = labels == CLASS_TO_LABEL["AD"]
            ad_correct += ((predictions == CLASS_TO_LABEL["AD"]) & ad_mask).sum().item()
            ad_total += ad_mask.sum().item()

    ad_recall = ad_correct / ad_total if ad_total else 0.0
    false_negative_rate = 1.0 - ad_recall if ad_total else 0.0
    return total_loss / total, correct / total, ad_recall, false_negative_rate


def print_class_counts(name, data_loader):
    subset = data_loader.dataset
    label_counts = Counter(subset.dataset.samples[index][1] for index in subset.indices)
    class_counts = ", ".join(
        f"{class_name}={label_counts[label]}"
        for class_name, label in CLASS_TO_LABEL.items()
    )
    print(f"{name} set: {class_counts}")


def main():
    train_loader, validation_loader = load_train_validation(batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(
        f"Training on {device}: {len(train_loader.dataset)} training images, "
        f"{len(validation_loader.dataset)} validation images, "
        f"{EPOCHS} epochs, batch size {BATCH_SIZE}."
    )
    print_class_counts("Train", train_loader)
    print_class_counts("Validation", validation_loader)

    epochs = []
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    ad_recalls = []
    false_negative_rates = []

    plt.ion()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    loss_axis, accuracy_axis, recall_axis, false_negative_axis = axes.flat

    for epoch in range(1, EPOCHS + 1):
        epoch_loss, epoch_accuracy = train_model(
            model, train_loader, loss_function, optimizer, device
        )
        (
            validation_loss,
            validation_accuracy,
            ad_recall,
            false_negative_rate,
        ) = evaluate_model(model, validation_loader, loss_function, device)

        epochs.append(epoch)
        train_losses.append(epoch_loss)
        validation_losses.append(validation_loss)
        train_accuracies.append(epoch_accuracy)
        validation_accuracies.append(validation_accuracy)
        ad_recalls.append(ad_recall)
        false_negative_rates.append(false_negative_rate)

        # Plot training and validation loss.
        loss_axis.clear()
        loss_axis.plot(epochs, train_losses, marker="o", label="Train")
        loss_axis.plot(epochs, validation_losses, marker="o", label="Validation")
        loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Loss")
        loss_axis.legend()
        loss_axis.grid(True, alpha=0.3)

        # Plot training and validation accuracy.
        accuracy_axis.clear()
        accuracy_axis.plot(epochs, train_accuracies, marker="o", label="Train")
        accuracy_axis.plot(
            epochs, validation_accuracies, marker="o", label="Validation"
        )
        accuracy_axis.set(
            title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1)
        )
        accuracy_axis.legend()
        accuracy_axis.grid(True, alpha=0.3)

        # Plot validation recall for the AD class.
        recall_axis.clear()
        recall_axis.plot(epochs, ad_recalls, marker="o", color="tab:green")
        recall_axis.set(
            title="Validation AD Recall",
            xlabel="Epoch",
            ylabel="Recall",
            ylim=(0, 1),
        )
        recall_axis.grid(True, alpha=0.3)

        # Plot validation false-negative rate for the AD class.
        false_negative_axis.clear()
        false_negative_axis.plot(
            epochs, false_negative_rates, marker="o", color="tab:red"
        )
        false_negative_axis.set(
            title="Validation AD False-Negative Rate",
            xlabel="Epoch",
            ylabel="False-negative rate",
            ylim=(0, 1),
        )
        false_negative_axis.grid(True, alpha=0.3)

        figure.tight_layout()
        figure.canvas.draw_idle()
        plt.pause(0.01)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {epoch_loss:.4f} | "
            f"train accuracy: {epoch_accuracy:.2%} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%} | "
            f"AD recall: {ad_recall:.2%} | "
            f"AD false-negative rate: {false_negative_rate:.2%}"
        )

    plot_directory = Path("./plots")
    plot_directory.mkdir(parents=True, exist_ok=True)
    plot_path = plot_directory / "training_metrics.png"
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved training plots to {plot_path}")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
