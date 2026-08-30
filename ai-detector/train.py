import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cnn_model import VoiceCNN
from dataset import VoiceDataset


def resolve_max_samples():
    raw_value = os.environ.get("MAX_SAMPLES")
    if raw_value is None or raw_value.strip() == "":
        return None

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("MAX_SAMPLES must be an integer or empty") from exc

    if value <= 0:
        return None
    return value

def main():
    # Reproducibility
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset and split
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    dataset = VoiceDataset(root_dir)

    MAX_SAMPLES = resolve_max_samples()
    if MAX_SAMPLES is not None and len(dataset) > MAX_SAMPLES:
        real_samples = [sample for sample in dataset.samples if sample[1] == 0]
        fake_samples = [sample for sample in dataset.samples if sample[1] == 1]
        keep_count = min(len(real_samples), len(fake_samples), MAX_SAMPLES // 2)
        if keep_count < 1:
            raise ValueError("MAX_SAMPLES is too small. Use a value >= 2 to retain at least one real and one fake clip.")
        dataset.samples = real_samples[:keep_count] + fake_samples[:keep_count]

    if len(dataset) < 2:
        raise ValueError("Dataset is empty. Add real/fake audio samples before training.")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = VoiceCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    best_val_acc = 0.0
    EPOCHS = int(os.environ.get("EPOCHS", "5"))
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_cnn_model.pth")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / max(total, 1)
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / max(val_total, 1)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {running_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Acc: {val_acc:.2f}%"
        )

    print("\nTraining completed!")
    print("Best validation accuracy:", round(best_val_acc, 2), "%")
    print("Model saved at:", os.path.abspath(MODEL_PATH))


if __name__ == "__main__":
    main()