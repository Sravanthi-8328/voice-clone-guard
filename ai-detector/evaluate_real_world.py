import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import VoiceDataset
from cnn_model import VoiceCNN

DATA_DIR = os.path.join(ROOT, "data")
MODEL_PATH = os.path.join(ROOT, "voice_cnn_model.pth")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model checkpoint not found. Run ai-detector/train.py before evaluating on real-world audio."
    )

valid_dataset = VoiceDataset(DATA_DIR, metadata_path=os.path.join(DATA_DIR, "validation_labels.csv"))
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VoiceCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print("Validation samples:", total)
print("Correct predictions:", correct)
print("Validation accuracy:", round(100.0 * correct / max(total, 1), 2), "%")
