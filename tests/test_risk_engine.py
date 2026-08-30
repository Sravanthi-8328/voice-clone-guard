import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AI_DETECTOR_DIR = ROOT / "ai-detector"
if str(AI_DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DETECTOR_DIR))

from dataset import VoiceDataset
from risk_engine import calculate_risk


def test_low_risk_for_clean_voice():
    result = calculate_risk(0.08, prosody_score=0.12, acoustic_anomaly=0.18)

    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] == "LOW"
    assert "recommendation" in result
    assert "ai_probability" in result


def test_high_risk_for_detected_impersonation_signals():
    result = calculate_risk(0.92, prosody_score=0.91, acoustic_anomaly=0.88)

    assert result["risk_level"] == "HIGH"
    assert result["risk_score"] >= 75
    assert result["ai_probability"] >= 0.8
    assert result["combined_score"] >= 0.7


def test_dataset_ignores_untrusted_test_directory(tmp_path):
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    test_dir = tmp_path / "test"
    real_dir.mkdir()
    fake_dir.mkdir()
    test_dir.mkdir()

    (real_dir / "real1.wav").write_bytes(b"fake")
    (fake_dir / "fake1.wav").write_bytes(b"fake")
    (test_dir / "possibly_real_but_untrusted.wav").write_bytes(b"fake")

    metadata = tmp_path / "labels.csv"
    with open(metadata, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_path", "label"])
        writer.writerow(["real/real1.wav", "real"])
        writer.writerow(["fake/fake1.wav", "fake"])

    dataset = VoiceDataset(str(tmp_path), metadata_path=str(metadata))
    labels = sorted(label for _, label in dataset.samples)
    assert labels == [0, 1]
    assert len(dataset) == 2
