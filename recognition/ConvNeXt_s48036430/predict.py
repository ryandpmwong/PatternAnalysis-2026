from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import BATCH_SIZE, CLASS_TO_LABEL, load_dataset, load_train_validation
from modules import ConvNeXtClassifier

MODEL_PATH = Path("./models/convnext_best.pth")
PLOT_PATH = Path("./plots/test_metrics_vs_threshold.png")


def collect_ad_probabilities(model, data_loader, device, loss_function=None):
    model.eval()
    probabilities = []
    labels = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, batch_labels in data_loader:
            batch_labels = batch_labels.to(device)
            logits = model(images.to(device))
            probabilities.append(
                torch.softmax(logits, dim=1)[:, CLASS_TO_LABEL["AD"]].cpu()
            )
            labels.append(batch_labels.cpu())
            if loss_function is not None:
                batch_size = batch_labels.size(0)
                total_loss += loss_function(logits, batch_labels).item() * batch_size
                total += batch_size

    ad_probabilities = torch.cat(probabilities)
    collected_labels = torch.cat(labels)
    average_loss = total_loss / total if total else None
    return ad_probabilities, collected_labels, average_loss


def find_best_ad_threshold(ad_probabilities, labels):
    unique_probabilities = torch.unique(ad_probabilities, sorted=True)
    candidates = [0.0, 0.5, 1.0]
    candidates.extend(
        ((unique_probabilities[:-1] + unique_probabilities[1:]) / 2).tolist()
    )

    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in candidates:
        predictions = torch.where(
            ad_probabilities >= threshold,
            CLASS_TO_LABEL["AD"],
            CLASS_TO_LABEL["NC"],
        )
        accuracy = (predictions == labels).float().mean().item()
        if (accuracy, -abs(threshold - 0.5)) > (
            best_accuracy,
            -abs(best_threshold - 0.5),
        ):
            best_threshold = threshold
            best_accuracy = accuracy

    return best_threshold, best_accuracy


def calculate_metrics(ad_probabilities, labels, threshold):
    predictions = torch.where(
        ad_probabilities >= threshold,
        CLASS_TO_LABEL["AD"],
        CLASS_TO_LABEL["NC"],
    )
    accuracy = (predictions == labels).float().mean().item()
    ad_mask = labels == CLASS_TO_LABEL["AD"]
    ad_total = ad_mask.sum().item()
    ad_correct = ((predictions == CLASS_TO_LABEL["AD"]) & ad_mask).sum().item()
    ad_recall = ad_correct / ad_total if ad_total else 0.0
    return accuracy, ad_recall


def plot_test_metrics_by_threshold(ad_probabilities, labels, plot_path):
    thresholds = torch.arange(1, 100, dtype=torch.float32) / 100
    metrics = [
        calculate_metrics(ad_probabilities, labels, threshold.item())
        for threshold in thresholds
    ]
    accuracies, ad_recalls = zip(*metrics)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(thresholds.tolist(), accuracies, label="Test accuracy")
    axis.plot(thresholds.tolist(), ad_recalls, label="Test AD recall")
    axis.set(
        title="Test Metrics by AD Probability Threshold",
        xlabel="AD probability threshold",
        ylabel="Metric",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.set_xticks([index / 10 for index in range(11)])
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def generate_threshold_plot(
    model, validation_loader, test_loader, device, loss_function, plot_path
):
    validation_probabilities, validation_labels, validation_loss = (
        collect_ad_probabilities(
            model, validation_loader, device, loss_function=loss_function
        )
    )
    threshold, validation_accuracy = find_best_ad_threshold(
        validation_probabilities, validation_labels
    )
    _, validation_ad_recall = calculate_metrics(
        validation_probabilities, validation_labels, threshold
    )
    print(
        f"Validation-tuned AD threshold: {threshold:.4f} | "
        f"validation accuracy: {validation_accuracy:.2%} | "
        f"AD recall: {validation_ad_recall:.2%} | "
        f"loss: {validation_loss:.4f}"
    )

    test_probabilities, test_labels, test_loss = collect_ad_probabilities(
        model, test_loader, device, loss_function=loss_function
    )
    for name, test_threshold in (
        ("validation-tuned", threshold),
        ("default threshold 0.5", 0.5),
    ):
        test_accuracy, test_ad_recall = calculate_metrics(
            test_probabilities, test_labels, test_threshold
        )
        print(
            f"Test results ({name}) | "
            f"loss: {test_loss:.4f} | "
            f"accuracy: {test_accuracy:.2%} | "
            f"AD recall: {test_ad_recall:.2%}"
        )

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_test_metrics_by_threshold(test_probabilities, test_labels, plot_path)
    print(f"Saved test threshold metrics plot to {plot_path}")


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = ConvNeXtClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, validation_loader = load_train_validation(batch_size=BATCH_SIZE)
    test_loader = load_dataset("test", batch_size=BATCH_SIZE)
    loss_function = torch.nn.CrossEntropyLoss()

    print(f"Loaded model checkpoint from {MODEL_PATH} on {device}.")
    generate_threshold_plot(
        model,
        validation_loader,
        test_loader,
        device,
        loss_function,
        PLOT_PATH,
    )


if __name__ == "__main__":
    main()
