"""
GPU-assisted sanitizer used by the web interface.

This path keeps file I/O and encoding on the CPU, but moves the expensive
spectral perturbation pass to Torch/CUDA in bounded chunks. It is intentionally
conservative for small NVIDIA cards such as the GTX 1060 3GB.
"""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


DEFAULT_GPU_CHUNK_SECONDS = 20.0
DEFAULT_GPU_OVERLAP_SECONDS = 0.25


def cuda_available() -> bool:
    """Return True when Torch can run CUDA kernels."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_web_sanitize(
    input_file,
    output_file=None,
    paranoid_mode: bool = False,
    output_format: Optional[str] = None,
    chunk_seconds: float = DEFAULT_GPU_CHUNK_SECONDS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Sanitize an audio file with a bounded CUDA spectral pass.

    The function raises RuntimeError when CUDA is unavailable so callers can
    choose an explicit fallback path.
    """
    if not verbose:
        with contextlib.redirect_stdout(io.StringIO()):
            return gpu_web_sanitize(
                input_file=input_file,
                output_file=output_file,
                paranoid_mode=paranoid_mode,
                output_format=output_format,
                chunk_seconds=chunk_seconds,
                verbose=True,
            )

    if not cuda_available():
        raise RuntimeError("CUDA is not available")

    input_path = Path(input_file)
    output_path = _resolve_output_path(input_path, output_file, output_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    audio, sr = _load_audio(input_path)
    duration = audio.shape[0] / sr if sr else 0.0

    print("GPU WEB SANITIZATION")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Paranoid mode: {paranoid_mode}")

    processed, chunks_processed, gpu_name = _process_audio_on_gpu(
        audio,
        sr,
        paranoid_mode=paranoid_mode,
        chunk_seconds=chunk_seconds,
    )

    _write_audio(processed, sr, output_path)
    total_time = time.time() - start_time

    return {
        "success": True,
        "output_file": str(output_path),
        "stats": {
            "metadata_removed": 1,
            "watermarks_removed": 0,
            "watermarks_detected": 0,
            "processing_time": total_time,
            "processing_speed": f"{duration / max(total_time, 1e-9):.1f}x real-time",
            "processing_engine": "gpu_cuda_web",
            "gpu_acceleration": True,
            "gpu_device": gpu_name,
            "gpu_chunks_processed": chunks_processed,
            "gpu_chunk_seconds": chunk_seconds,
        },
    }


def _resolve_output_path(
    input_path: Path,
    output_file,
    output_format: Optional[str],
) -> Path:
    normalized_format = (output_format or "").lower().lstrip(".")
    if normalized_format in ("", "preserve"):
        normalized_format = input_path.suffix.lstrip(".").lower()
    if not normalized_format:
        normalized_format = "wav"

    if output_file is not None:
        return Path(output_file).with_suffix(f".{normalized_format}")
    return input_path.with_suffix(f".clean.{normalized_format}")


def _load_audio(input_path: Path) -> Tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(audio, dtype=np.float32), int(sr)
    except (RuntimeError, OSError, ValueError):
        audio, sr = librosa.load(str(input_path), sr=None, mono=False, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        else:
            audio = audio.T
        return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def _process_audio_on_gpu(
    audio: np.ndarray,
    sr: int,
    paranoid_mode: bool,
    chunk_seconds: float,
) -> Tuple[np.ndarray, int, str]:
    import torch

    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.size == 0:
        return audio.astype(np.float32), 0, torch.cuda.get_device_name(0)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    chunk_samples = max(2048, int(float(chunk_seconds) * sr))
    overlap_samples = min(
        int(DEFAULT_GPU_OVERLAP_SECONDS * sr),
        max(0, chunk_samples // 4),
    )
    step = max(1, chunk_samples - overlap_samples)

    output = np.zeros_like(audio, dtype=np.float32)
    weights = np.zeros((audio.shape[0], 1), dtype=np.float32)
    chunks_processed = 0

    for start in range(0, audio.shape[0], step):
        end = min(start + chunk_samples, audio.shape[0])
        if end <= start:
            continue

        chunk = audio[start:end]
        processed = _process_chunk_torch(chunk, sr, paranoid_mode, device)
        weight = _chunk_weight(end - start, overlap_samples, start, end, audio.shape[0])

        output[start:end] += processed * weight[:, None]
        weights[start:end] += weight[:, None]
        chunks_processed += 1

        if end >= audio.shape[0]:
            break

    weights = np.where(weights <= 0, 1.0, weights)
    output = output / weights

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return np.clip(output, -1.0, 1.0).astype(np.float32), chunks_processed, gpu_name


def _process_chunk_torch(
    chunk: np.ndarray,
    sr: int,
    paranoid_mode: bool,
    device,
) -> np.ndarray:
    import torch

    x = torch.as_tensor(chunk, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x[:, None]

    original_rms = torch.sqrt(torch.mean(x * x, dim=0, keepdim=True)).clamp_min(1e-8)
    centered = x - torch.mean(x, dim=0, keepdim=True)

    spec = torch.fft.rfft(centered, dim=0)
    magnitude = torch.abs(spec)
    phase = torch.angle(spec)

    freqs = torch.fft.rfftfreq(centered.shape[0], d=1 / sr, device=device)
    nyquist = max(sr / 2, 1.0)
    freq_weight = torch.clamp(freqs / nyquist, 0, 1) ** 1.5

    base = 0.00035 if not paranoid_mode else 0.0007
    high = 0.0016 if not paranoid_mode else 0.0032
    jitter_std = base + high * freq_weight
    jitter_std = jitter_std[:, None]

    # Extra high-frequency decorrelation where watermark-like energy usually lives.
    hf_mask = (freqs >= 14_000)[:, None]
    jitter_std = torch.where(hf_mask, jitter_std * (1.8 if paranoid_mode else 1.35), jitter_std)
    jitter_std[0, :] = 0.0

    phase_noise = torch.randn_like(phase) * jitter_std
    modified = magnitude * torch.exp(1j * (phase + phase_noise))
    y = torch.fft.irfft(modified, n=centered.shape[0], dim=0)

    # Blend with the original and restore RMS to keep the result stable.
    blend = 0.28 if paranoid_mode else 0.18
    y = (1.0 - blend) * x + blend * y
    new_rms = torch.sqrt(torch.mean(y * y, dim=0, keepdim=True)).clamp_min(1e-8)
    y = y * (original_rms / new_rms)

    return y.detach().cpu().numpy().astype(np.float32)


def _chunk_weight(
    length: int,
    overlap_samples: int,
    start: int,
    end: int,
    total_length: int,
) -> np.ndarray:
    weight = np.ones(length, dtype=np.float32)
    fade = min(overlap_samples, length // 2)
    if fade > 0 and start > 0:
        weight[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    if fade > 0 and end < total_length:
        weight[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
    return weight


def _write_audio(audio: np.ndarray, sr: int, output_path: Path) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]

    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    if peak > 1.0:
        audio = audio / peak

    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    output_format = output_path.suffix.lstrip(".").lower()

    if output_format == "mp3":
        from pydub import AudioSegment

        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=audio_int16.shape[1],
        )
        segment.export(
            str(output_path),
            format="mp3",
            bitrate="320k",
            parameters=[
                "-map_metadata",
                "-1",
                "-write_xing",
                "0",
                "-id3v2_version",
                "0",
                "-write_id3v1",
                "0",
            ],
        )
        return

    sf_format = output_format.upper()
    if sf_format == "FLAC":
        sf.write(str(output_path), audio_int16, sr, format="FLAC")
    else:
        sf.write(str(output_path), audio_int16, sr, format=sf_format, subtype="PCM_16")
