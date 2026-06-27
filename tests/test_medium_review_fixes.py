import sys
import threading
import time
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

import mmm.cli as cli_module
import mmm.core.audio_sanitizer as audio_sanitizer_module
import mmm.turbo_analysis as turbo_analysis_module
from mmm.cli import cli
from mmm.config.config_manager import ConfigManager
from mmm.config.defaults import DEFAULT_CONFIG
from mmm.core.audio_sanitizer import AudioSanitizer
from mmm.detection.metadata_scanner import MetadataScanner
from mmm.optimized_processor import OptimizedAudioProcessor
from mmm.preserving_sanitizer import preserving_sanitize
from mmm.server import create_app
from mmm.sanitization.metadata_cleaner import MetadataCleaner
from mmm.turbo_analysis import turbo_analysis


def test_sanitize_audio_loads_once_and_uses_single_final_output(tmp_path):
    input_file = tmp_path / "input.wav"
    output_file = tmp_path / "out" / "output.wav"
    input_file.write_bytes(b"source")

    sanitizer = object.__new__(AudioSanitizer)
    sanitizer.input_file = input_file
    sanitizer.output_file = output_file
    sanitizer.output_format = "wav"
    sanitizer.paranoid_mode = False
    sanitizer.config = {}
    sanitizer.audio_data = None
    sanitizer.sample_rate = None
    sanitizer.original_hash = "original"
    sanitizer.metadata_crimes = []
    sanitizer.processing_stats = {
        "metadata_removed": 0,
        "watermarks_detected": 0,
        "watermarks_removed": 0,
        "quality_loss": 0.0,
        "processing_time": 0,
    }

    load_calls = []

    def fake_load_audio():
        load_calls.append(True)
        sanitizer.audio_data = np.zeros((8, 1), dtype=np.float32)
        sanitizer.sample_rate = 8000
        sanitizer.original_hash = "original"
        return True

    clean_calls = []

    class RecordingMetadataCleaner:
        def clean_file(self, temp_path, final_path):
            clean_calls.append((temp_path, final_path))
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_bytes(temp_path.read_bytes())
            return {"success": True, "tags_removed": 1, "chunks_removed": 0, "errors": []}

        def _verify_metadata_present(self, _path):
            return False

    sanitizer.load_audio = fake_load_audio
    sanitizer.metadata_scanner = SimpleNamespace(
        scan_file=lambda _path: {
            "tags": [{"key": "TXXX", "suspicious": True}],
            "suspicious_chunks": [],
            "hidden_data": [],
        }
    )
    sanitizer.watermark_detector = SimpleNamespace(
        detect_all=lambda _audio, _sr: {"detected": []}
    )
    sanitizer.statistical_analyzer = SimpleNamespace(
        analyze=lambda _audio, _sr: {"anomalies": []}
    )
    sanitizer.metadata_cleaner = RecordingMetadataCleaner()
    sanitizer.spectral_cleaner = SimpleNamespace(
        clean_watermarks=lambda audio, _sr: {
            "cleaned_audio": audio,
            "watermarks_found": 0,
            "watermarks_removed": 0,
        }
    )
    sanitizer.fingerprint_remover = SimpleNamespace(
        remove_fingerprints=lambda audio, _sr: {"cleaned_audio": audio}
    )
    def fake_save_audio(_audio, output_file=None):
        output_file.write_bytes(b"clean")

    sanitizer._save_audio = fake_save_audio
    sanitizer._calculate_quality_loss = lambda: 0.0
    sanitizer._calculate_file_hash = lambda _path: "final"

    result = AudioSanitizer.sanitize_audio(sanitizer)

    assert result["success"] is True
    assert len(load_calls) == 1
    assert result["stats"]["metadata_removed"] == 1
    assert len(clean_calls) == 1
    assert clean_calls[0][0] != output_file
    assert clean_calls[0][1] == output_file
    assert not clean_calls[0][0].exists()
    assert output_file.read_bytes() == b"clean"


def test_quality_loss_records_metric_error(monkeypatch, tmp_path):
    import mmm.core.audio_sanitizer as audio_module

    sanitizer = object.__new__(AudioSanitizer)
    sanitizer.input_file = tmp_path / "input.wav"
    sanitizer.output_file = tmp_path / "output.wav"
    sanitizer.sample_rate = 8000
    sanitizer.processing_stats = {}

    monkeypatch.setattr(
        audio_module.librosa,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )

    assert AudioSanitizer._calculate_quality_loss(sanitizer) == 0.0
    assert sanitizer.processing_stats["quality_loss_error"] == "decode failed"


def test_load_audio_preserves_multichannel_layout(monkeypatch, tmp_path):
    input_file = tmp_path / "stereo.wav"
    input_file.write_bytes(b"fake audio bytes")
    calls = {}

    def fake_load(path, sr=None, mono=False):
        calls["mono"] = mono
        return np.zeros((2, 4), dtype=np.float32), 8000

    monkeypatch.setattr(audio_sanitizer_module.librosa, "load", fake_load)

    sanitizer = AudioSanitizer(input_file=input_file)

    assert sanitizer.load_audio() is True
    assert calls["mono"] is False
    assert sanitizer.audio_data.shape == (4, 2)


def test_massacre_uses_worker_threads(monkeypatch, tmp_path):
    for index in range(4):
        (tmp_path / f"track{index}.wav").write_bytes(b"audio")

    thread_names = set()

    def fake_process(file_path, output_dir, paranoid, backup, turbo):
        thread_names.add(threading.current_thread().name)
        time.sleep(0.01)
        return {"success": True, "mode": "turbo", "output_file": str(file_path)}

    monkeypatch.setattr(cli_module, "_process_massacre_file", fake_process)

    result = CliRunner().invoke(
        cli,
        ["massacre", str(tmp_path), "--workers", "2", "--turbo"],
    )

    assert result.exit_code == 0
    assert any(name != "MainThread" for name in thread_names)


def test_massacre_exits_nonzero_when_any_file_fails(monkeypatch, tmp_path):
    (tmp_path / "track.wav").write_bytes(b"audio")

    monkeypatch.setattr(
        cli_module,
        "_process_massacre_file",
        lambda *_args, **_kwargs: {"success": False, "error": "boom"},
    )

    result = CliRunner().invoke(cli, ["massacre", str(tmp_path), "--workers", "1"])

    assert result.exit_code == 1
    assert "failure" in result.output.lower()


def test_massacre_rejects_excessive_worker_count(tmp_path):
    (tmp_path / "track.wav").write_bytes(b"audio")

    result = CliRunner().invoke(cli, ["massacre", str(tmp_path), "--workers", "33"])

    assert result.exit_code != 0


def test_turbo_massacre_helper_creates_backup_before_preserving(monkeypatch, tmp_path):
    input_file = tmp_path / "track.wav"
    output_dir = tmp_path / "out"
    input_file.write_bytes(b"audio")

    monkeypatch.setitem(
        sys.modules,
        "mmm.turbo_analysis",
        SimpleNamespace(turbo_analysis=lambda _path: {"total_threats": 2}),
    )
    monkeypatch.setitem(
        sys.modules,
        "mmm.preserving_sanitizer",
        SimpleNamespace(
            preserving_sanitize=lambda *_args, **_kwargs: {
                "success": True,
                "output_file": str(output_dir / "track.wav"),
            }
        ),
    )

    result = cli_module._process_massacre_file(
        input_file, output_dir, paranoid=False, backup=True, turbo=True
    )

    assert result["success"] is True
    assert result["mode"] == "turbo"
    assert input_file.with_suffix(".backup.wav").read_bytes() == b"audio"


def test_config_manager_deep_copies_nested_defaults(tmp_path):
    manager = ConfigManager(config_file=tmp_path / "config.yaml")
    manager.set("audio_processing.sample_rate", 12345)

    assert DEFAULT_CONFIG["audio_processing"]["sample_rate"] is None

    fresh = ConfigManager(config_file=tmp_path / "fresh.yaml")
    assert fresh.get("audio_processing.sample_rate") is None

    exported = fresh.get_config()
    exported["audio_processing"]["sample_rate"] = 999
    assert fresh.get("audio_processing.sample_rate") is None


def test_import_config_rejects_non_mapping_yaml(tmp_path):
    manager = ConfigManager(config_file=tmp_path / "config.yaml")
    import_file = tmp_path / "import.yaml"
    import_file.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(Exception, match="YAML mapping"):
        manager.import_config(import_file)


def test_cli_module_import_does_not_create_global_config():
    assert not hasattr(cli_module, "config")


def test_preserving_sanitize_accepts_string_input_for_missing_file(tmp_path):
    result = preserving_sanitize(str(tmp_path / "missing.wav"))

    assert result["success"] is False
    assert "Input file not found" in result["error"]


def test_preserving_sanitize_quiet_mode_suppresses_output(tmp_path):
    buffer = StringIO()

    with redirect_stdout(buffer):
        result = preserving_sanitize(str(tmp_path / "missing.wav"), verbose=False)

    assert result["success"] is False
    assert buffer.getvalue() == ""


@pytest.mark.parametrize("chunk_duration", [0, -1, float("nan"), "nope"])
def test_turbo_analysis_rejects_invalid_chunk_duration(tmp_path, chunk_duration):
    with pytest.raises(ValueError, match="chunk_duration"):
        turbo_analysis(tmp_path / "missing.wav", chunk_duration=chunk_duration)


def test_turbo_analysis_quiet_mode_suppresses_output(monkeypatch, tmp_path):
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake")

    class FakeProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def load_audio_optimized(self, _path):
            return np.zeros(4, dtype=np.float32), 2

    monkeypatch.setattr(turbo_analysis_module, "OptimizedAudioProcessor", FakeProcessor)
    monkeypatch.setattr(turbo_analysis_module, "_detect_gpu_name", lambda: "test")
    monkeypatch.setattr(turbo_analysis_module, "_is_gpu_available", lambda: False)
    monkeypatch.setattr(
        turbo_analysis_module,
        "analyze_audio_chunk_gpu",
        lambda _args: {
            "watermarks": {"detected": False, "confidence": 0.0},
            "processing_time": 0.0,
        },
    )
    monkeypatch.setattr(
        turbo_analysis_module,
        "MetadataScanner",
        lambda: SimpleNamespace(
            scan_file=lambda _path: {
                "tags": [],
                "suspicious_chunks": [],
                "hidden_data": [],
            }
        ),
    )

    buffer = StringIO()
    with redirect_stdout(buffer):
        result = turbo_analysis(input_file, chunk_duration=1.0, verbose=False)

    assert result["file_info"]["path"] == str(input_file)
    assert buffer.getvalue() == ""


@pytest.mark.parametrize("chunk_duration", [0, -1, float("nan"), "bad"])
def test_optimized_processor_rejects_invalid_chunk_duration(chunk_duration):
    processor = OptimizedAudioProcessor(use_gpu=False, use_multiprocessing=False)

    with pytest.raises(ValueError, match="chunk_duration"):
        processor.detect_watermarks_parallel(
            np.zeros(8, dtype=np.float32),
            sample_rate=8000,
            chunk_duration=chunk_duration,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("TIT2", "Piano Concerto"),
        ("TPE1", "Brian Eno"),
        ("TALB", "Main Theme"),
    ],
)
def test_metadata_scanner_does_not_flag_plain_ai_substrings(key, value):
    assert MetadataScanner()._is_tag_suspicious(key, value) is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("TXXX:watermark", "abc"),
        ("comment", "generated by OpenAI"),
        ("PRIV", "uuid 123e4567-e89b-12d3-a456-426614174000"),
    ],
)
def test_metadata_scanner_still_flags_strong_indicators(key, value):
    assert MetadataScanner()._is_tag_suspicious(key, value) is True


def test_setup_py_matches_runtime_requirements_for_audio_paths():
    root = Path(__file__).resolve().parents[1]
    setup_text = (root / "setup.py").read_text(encoding="utf-8")
    requirements_text = (root / "requirements.txt").read_text(encoding="utf-8")

    for dependency in ("soxr>=0.3.0", "numba>=0.59.0"):
        assert dependency in setup_text
        assert dependency in requirements_text


def test_generic_metadata_cleaner_fails_closed_without_copy(monkeypatch, tmp_path):
    input_file = tmp_path / "input.flac"
    output_file = tmp_path / "output.flac"
    input_file.write_bytes(b"metadata-bearing original")

    monkeypatch.setattr(
        "mmm.sanitization.metadata_cleaner.AudioSegment.from_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )

    result = MetadataCleaner().clean_file(input_file, output_file)

    assert result["success"] is False
    assert "copy fallback" in " ".join(result["errors"]).lower()
    assert not output_file.exists()


def test_metadata_verification_handles_small_files(monkeypatch, tmp_path):
    tiny_file = tmp_path / "tiny.bin"
    tiny_file.write_bytes(b"small")
    monkeypatch.setattr("mmm.sanitization.metadata_cleaner.MutagenFile", lambda _p: None)

    assert MetadataCleaner()._verify_metadata_present(tiny_file) is False


def test_server_rejects_extension_only_fake_audio_upload():
    app = create_app(max_file_size=1024)
    client = app.test_client()

    response = client.post(
        "/api/upload",
        data={
            "file": (BytesIO(b"not really audio"), "fake.wav"),
            "format": "preserve",
            "paranoid": "false",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Invalid or unsupported audio" in response.get_json()["error"]


def test_server_download_registry_has_lock():
    app = create_app(max_file_size=1024)

    assert "DOWNLOAD_REGISTRY_LOCK" in app.config
    assert hasattr(app.config["DOWNLOAD_REGISTRY_LOCK"], "acquire")
