import torch
from torch.utils.data import DataLoader, random_split

from dataset import VoiceDataset
from cnn_model import VoiceCNN


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Dataset
dataset = VoiceDataset("data")


# Same split as training
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)


print("Test samples:", len(test_dataset))


# DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


# Load model
model = VoiceCNN().to(device)

model.load_state_dict(
    torch.load("voice_cnn_model.pth", map_location=device)
)

model.eval()

print("Model loaded successfully")


# Confusion matrix counters

# Actual REAL
real_correct = 0       # REAL → REAL
real_as_fake = 0       # REAL → FAKE

# Actual FAKE
fake_correct = 0       # FAKE → FAKE
fake_as_real = 0       # FAKE → REAL


with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        for actual, prediction in zip(labels, predicted):

            actual = actual.item()
            prediction = prediction.item()

            if actual == 0:  # REAL

                if prediction == 0:
                    real_correct += 1
                else:
                    real_as_fake += 1

            else:  # FAKE

                if prediction == 1:
                    fake_correct += 1
                else:
                    fake_as_real += 1


# Totals
total_real = real_correct + real_as_fake
total_fake = fake_correct + fake_as_real

total_correct = real_correct + fake_correct
total_samples = total_real + total_fake

accuracy = 100 * total_correct / total_samples


print("\n========== CONFUSION MATRIX ==========\n")

print("Actual REAL:")
print(f"  Predicted REAL: {real_correct}")
print(f"  Predicted FAKE: {real_as_fake}")

print("\nActual FAKE:")
print(f"  Predicted FAKE: {fake_correct}")
print(f"  Predicted REAL: {fake_as_real}")

print("\n======================================")

print(f"\nTotal REAL samples: {total_real}")
print(f"Total FAKE samples: {total_fake}")

print(f"\nOverall Accuracy: {accuracy:.2f}%")

if total_real > 0:
    real_accuracy = 100 * real_correct / total_real
    print(f"REAL detection accuracy: {real_accuracy:.2f}%")

if total_fake > 0:
    fake_accuracy = 100 * fake_correct / total_fake
    print(f"FAKE detection accuracy: {fake_accuracy:.2f}%")