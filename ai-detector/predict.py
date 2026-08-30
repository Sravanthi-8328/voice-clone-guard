import os
import sys
import torch

from cnn_model import VoiceCNN
from risk_engine import calculate_risk
from prosody_analysis import calculate_prosody_risk


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_PIPELINE_DIR = os.path.join(BASE_DIR, "audio-pipeline")

if AUDIO_PIPELINE_DIR not in sys.path:
    sys.path.append(AUDIO_PIPELINE_DIR)

from audio_processor import preprocess_audio


# ==========================================
# DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD CNN MODEL
# ==========================================

model = VoiceCNN().to(device)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "voice_cnn_model.pth"
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("Model loaded successfully")
print("Using device:", device)


# ==========================================
# AUDIO ANALYSIS FUNCTION
# ==========================================

def analyze_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    mel = preprocess_audio(file_path)
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(mel)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = predicted.item()
    real_probability = probabilities[0][0].item()
    fake_probability = probabilities[0][1].item()
    confidence_score = confidence.item() * 100

    prosody = calculate_prosody_risk(file_path)
    acoustic_anomaly = min(1.0, max(0.0, 0.5 * fake_probability + 0.5 * prosody["prosody_score"]))
    risk_result = calculate_risk(
        fake_probability,
        prosody_score=prosody["prosody_score"],
        acoustic_anomaly=acoustic_anomaly,
    )

    return {
        "voice_classification": "REAL HUMAN VOICE" if predicted_class == 0 else "FAKE / AI-GENERATED VOICE",
        "real_probability": round(real_probability, 4),
        "ai_probability": round(fake_probability, 4),
        "model_confidence": round(confidence_score, 2),
        "prosody": prosody,
        "risk": risk_result,
    }


def predict_audio(file_path):
    try:
        result = analyze_audio(file_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print("\n========== VOICE INTEGRITY REPORT ==========")
    print(f"Voice Classification: {result['voice_classification']}")
    print(f"\nReal Voice Probability: {result['real_probability'] * 100:.2f}%")
    print(f"AI Voice Probability: {result['ai_probability'] * 100:.2f}%")
    print(f"Model Confidence: {result['model_confidence']:.2f}%")
    print(f"\nOverall Risk Score: {result['risk']['risk_score']}/100")
    print(f"Risk Level: {result['risk']['risk_level']}")
    print("\nRecommended Action:")
    print(result["risk"]["recommendation"])
    print("\n============================================\n")


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            'python predict.py "path_to_audio.wav"'
        )

    else:

        predict_audio(sys.argv[1])