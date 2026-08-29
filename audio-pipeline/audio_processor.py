import torch
import torchaudio
import numpy as np

def preprocess_audio(file_path: str, target_sr: int = 16000, n_mels: int = 64, target_frames: int = 128) -> np.ndarray:
    """
    Loads an audio file, resamples to 16kHz, converts to mono, normalizes amplitude,
    generates a Mel-Spectrogram matrix, and pads/crops time frames to 128 width.
    """
    wav, sr = torchaudio.load(file_path)
    
    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        
    # Convert to mono & normalize
    wav = wav.mean(dim=0, keepdim=True)
    wav = wav / (wav.abs().max() + 1e-9)
    
    # Compute Mel Spectrogram (in dB)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, hop_length=256, n_mels=n_mels
    )(wav)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel = mel_db.squeeze(0).numpy()
    
    # Pad or crop to target_frames (128)
    _, T = mel.shape
    if T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")
    elif T > target_frames:
        mel = mel[:, :target_frames]
        
    return mel.astype(np.float32)


def extract_features(wav_tensor, sr: int = 16000) -> dict:
    """
    Extracts DSP acoustic metrics: RMS energy, silence ratio,
    spectral centroid, spectral spread, and 13 MFCCs.
    """
    wav = wav_tensor.cpu().numpy() if hasattr(wav_tensor, "cpu") else wav_tensor
    duration = len(wav) / sr

    # RMS Energy & Silence Ratio
    rms = np.sqrt(np.mean(wav**2) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    silence_ratio = float(np.mean(np.abs(wav) < (10 ** (-40 / 20))))

    # Frequency Spectrum Metrics
    spec = np.abs(np.fft.rfft(wav, n=2048)) + 1e-9
    freqs = np.fft.rfftfreq(2048, d=1.0 / sr)
    spec_norm = spec / spec.sum()

    centroid_hz = float((freqs * spec_norm).sum())
    spread_hz = float(np.sqrt(((freqs - centroid_hz) ** 2 * spec_norm).sum()))

    # MFCC Extraction (13 coefficients)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=13, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 64}
    )
    mfcc = mfcc_transform(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()

    return {
        "duration_sec": round(duration, 3),
        "rms_db": round(rms_db, 2),
        "silence_ratio": round(silence_ratio, 3),
        "spectral_centroid_hz": round(centroid_hz, 1),
        "spectral_spread_hz": round(spread_hz, 1),
        "mfcc_mean": [round(v, 3) for v in np.mean(mfcc, axis=1)],
        "mfcc_std": [round(v, 3) for v in np.std(mfcc, axis=1)],
    }