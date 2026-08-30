import math
import os

import numpy as np
import soundfile as sf


def _read_audio(file_path):
    audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
    if audio is None or np.size(audio) == 0:
        raise ValueError(f"Audio file is empty: {file_path}")
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), sr


def _signal_energy(samples):
    return float(np.mean(np.square(samples)))


def _short_time_energy(signal, frame_size=256, hop=128):
    if signal.size == 0:
        return []
    energies = []
    for start in range(0, len(signal) - frame_size + 1, hop):
        frame = signal[start:start + frame_size]
        energies.append(_signal_energy(frame))
    return energies


def calculate_prosody_risk(file_path):
    """Return prosody anomaly scores aligned with the Phase 2 plan: pitch, energy, pauses, speaking rate."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    signal, sr = _read_audio(file_path)
    if signal.size == 0:
        return {
            "pitch_variation": 0.0,
            "energy_variation": 0.0,
            "pause_ratio": 0.0,
            "speaking_rate": 0.0,
            "prosody_score": 0.0,
        }

    signal = np.asarray(signal)
    signal = signal - np.mean(signal)
    amplitude = np.abs(signal)

    pitch_variation = float(np.std(np.abs(np.diff(signal))) / (np.std(signal) + 1e-6))
    energy = _short_time_energy(signal)
    energy_variation = float(np.std(energy) / (np.mean(energy) + 1e-6)) if energy else 0.0

    silence_threshold = max(1e-5, 0.05 * np.max(amplitude))
    pauses = np.where(amplitude < silence_threshold)[0]
    pause_ratio = float(len(pauses) / max(1, len(signal)))

    voiced_segments = np.where(amplitude > silence_threshold)[0]
    speaking_rate = float(len(voiced_segments) / max(1, len(signal) / sr)) if sr else 0.0

    prosody_score = float(
        min(
            1.0,
            0.35 * min(1.0, pitch_variation * 2.5)
            + 0.25 * min(1.0, energy_variation * 1.5)
            + 0.20 * min(1.0, pause_ratio * 25)
            + 0.20 * min(1.0, max(0.0, speaking_rate - 2.0) * 0.5)
        )
    )

    return {
        "pitch_variation": round(float(pitch_variation), 4),
        "energy_variation": round(float(energy_variation), 4),
        "pause_ratio": round(float(pause_ratio), 4),
        "speaking_rate": round(float(speaking_rate), 4),
        "prosody_score": round(float(prosody_score), 4),
    }
