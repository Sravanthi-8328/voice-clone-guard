import numpy as np
import torch
import torchaudio
import soundfile as sf


def _load_audio(file_path: str):
    try:
        audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if audio is None or audio.size == 0:
            raise ValueError("Audio file is empty")
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)
        return torch.from_numpy(np.asarray(audio, dtype=np.float32)), sr
    except Exception:
        audio, sr = torchaudio.load(file_path)
        audio = audio.float()
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        return audio, sr


def _standardize_mel(mel: np.ndarray):
    mel_mean = np.mean(mel)
    mel_std = np.std(mel)
    mel = (mel - mel_mean) / (mel_std + 1e-8)
    return mel.astype(np.float32)


def preprocess_audio(
    file_path: str,
    target_sr: int = 16000,
    n_mels: int = 64,
    target_frames: int = 256,
    max_chunks: int = 3,
) -> np.ndarray:
    wav, sr = _load_audio(file_path)

    if wav.numel() == 0:
        raise ValueError(f"No usable waveform found in {file_path}")

    wav = wav.unsqueeze(0)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    max_value = wav.abs().max()
    if max_value > 0:
        wav = wav / max_value

    chunk_size = max(1, wav.shape[-1] // max_chunks)
    chunk_mels = []

    for start in range(0, wav.shape[-1], chunk_size):
        chunk = wav[:, start:start + chunk_size]
        if chunk.shape[-1] < 160:
            continue

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
        )
        mel_spec = mel_transform(chunk)
        mel_db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel_spec)
        mel = mel_db.squeeze(0).numpy()
        mel = _standardize_mel(mel)

        _, time_steps = mel.shape
        if time_steps < target_frames:
            mel = np.pad(mel, ((0, 0), (0, target_frames - time_steps)), mode="constant")
        elif time_steps > target_frames:
            start_index = (time_steps - target_frames) // 2
            mel = mel[:, start_index:start_index + target_frames]

        chunk_mels.append(mel)

    if not chunk_mels:
        raise ValueError(f"Unable to derive mel features from {file_path}")

    aggregated = np.mean(np.stack(chunk_mels, axis=0), axis=0)
    return aggregated.astype(np.float32)