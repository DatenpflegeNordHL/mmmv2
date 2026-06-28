"""Audio export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


SUBTYPE_BY_BIT_DEPTH = {
    16: "PCM_16",
    24: "PCM_24",
    32: "FLOAT",
}


def write_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    *,
    bit_depth: int = 24,
) -> Path:
    """Write audio to WAV/FLAC/AIFF/MP3 while preserving sample rate."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".mp3":
        _write_mp3(output_path, audio, sample_rate)
        return output_path

    subtype = SUBTYPE_BY_BIT_DEPTH.get(int(bit_depth), "PCM_24")
    write_data = np.asarray(audio, dtype=np.float64)
    if write_data.ndim == 2 and write_data.shape[1] == 1:
        write_data = write_data[:, 0]
    sf.write(str(output_path), write_data, sample_rate, subtype=subtype)
    return output_path


def _write_mp3(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Export MP3 through ffmpeg without ID3/container metadata."""
    from pydub import AudioSegment

    write_data = np.asarray(audio, dtype=np.float64)
    if write_data.ndim == 1:
        write_data = write_data[:, None]
    peak = float(np.max(np.abs(write_data))) if write_data.size else 0.0
    if peak > 1.0:
        write_data = write_data / peak
    pcm = np.clip(write_data, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    segment = AudioSegment(
        pcm_i16.tobytes(),
        frame_rate=int(sample_rate),
        sample_width=2,
        channels=int(pcm_i16.shape[1]),
    )
    segment.export(
        str(path),
        format="mp3",
        bitrate="320k",
        parameters=[
            "-map_metadata",
            "-1",
            "-write_id3v1",
            "0",
            "-id3v2_version",
            "0",
            "-write_xing",
            "0",
        ],
    )
