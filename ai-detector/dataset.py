import csv
import os
import sys
import torch
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_PIPELINE_DIR = os.path.join(BASE_DIR, "audio-pipeline")
if AUDIO_PIPELINE_DIR not in sys.path:
    sys.path.append(AUDIO_PIPELINE_DIR)

from audio_processor import preprocess_audio

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def _find_audio_files(root_dir):
    if not os.path.exists(root_dir):
        return []

    audio_files = []
    for directory, _, file_names in os.walk(root_dir):
        directory_name = os.path.basename(directory).lower()
        if directory_name in {"test", "validation", "tmp", "tmpdir"}:
            continue
        for file_name in file_names:
            if file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                audio_files.append(os.path.join(directory, file_name))
    return sorted(audio_files)


class VoiceDataset(Dataset):
    def __init__(self, data_dir, metadata_path=None):
        self.samples = []
        metadata_path = metadata_path or os.path.join(data_dir, "labels.csv")

        if os.path.exists(metadata_path):
            with open(metadata_path, newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    file_name = row.get("file_path") or row.get("path") or row.get("filename")
                    label_value = row.get("label")
                    if not file_name or label_value is None:
                        continue
                    file_path = os.path.join(data_dir, file_name) if not os.path.isabs(file_name) else file_name
                    if not os.path.exists(file_path):
                        continue
                    if label_value.lower() in {"real", "0", "human", "genuine"}:
                        label = 0
                    elif label_value.lower() in {"fake", "1", "ai", "synthetic"}:
                        label = 1
                    else:
                        continue
                    self.samples.append((file_path, label))
            return

        for file_path in _find_audio_files(data_dir):
            normalized_path = os.path.normpath(file_path)
            parent_name = os.path.basename(os.path.dirname(normalized_path)).lower()
            if "real" in parent_name:
                label = 0
            elif "fake" in parent_name:
                label = 1
            else:
                continue
            self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        mel = preprocess_audio(file_path)
        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return mel, label