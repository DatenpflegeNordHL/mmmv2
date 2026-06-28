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
import mmm.gpu_web_sanitizer as gpu_web_module
import mmm.server as server_module
import mmm.turbo_analysis as turbo_analysis_module
from mmm.cli import cli
from mmm.config.config_manager import ConfigManager
from mmm.config.defaults import DEFAULT_CONFIG
from mmm.core.audio_sanitizer import AudioSanitizer
from mmm.detection.metadata_scanner import MetadataScanner
from mmm.detection.watermark_detector import WatermarkDetector
from mmm.forensic_report import write_forensic_report
from mmm.gpu_web_sanitizer import _verify_signal_delta
from mmm.optimized_processor import OptimizedAudioProcessor
from mmm.optimized_processor import GPUAcceleratedWatermarkDetector
from mmm.preserving_sanitizer import preserving_sanitize
from mmm.server import create_app
from mmm.sanitization.fingerprint_remover import FingerprintRemover
from mmm.sanitization.metadata_cleaner import MetadataCleaner
from mmm.sanitization.spectral_cleaner import SpectralCleaner
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
        clean_watermarks=lambda audio, _sr, **_kwargs: {
            "cleaned_audio": audio,
            "watermarks_found": 0,
            "watermarks_removed": 0,
        }
    )
    sanitizer.fingerprint_remover = SimpleNamespace(
        remove_fingerprints=lambda audio, _sr: {"cleaned_audio": audio}
    )
    def fake_save_audio(_audio, output_file=None):
        output_file.parent.mkdir(parents=True, exist_ok=True)
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

    monkeypatch.setattr(cli_module, "_cuda_available_for_worker_limit", lambda: False)
    monkeypatch.setattr(cli_module, "_process_massacre_file", fake_process)

    result = CliRunner().invoke(
        cli,
        ["massacre", str(tmp_path), "--workers", "2", "--turbo"],
    )

    assert result.exit_code == 0
    assert any(name != "MainThread" for name in thread_names)


def test_massacre_recursive_finds_flac_by_default(monkeypatch, tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    input_file = nested / "track.flac"
    input_file.write_bytes(b"audio")
    seen = []

    def fake_process(file_path, output_dir, paranoid, backup, turbo):
        seen.append(file_path)
        return {"success": True, "mode": "regular", "output_file": str(file_path)}

    monkeypatch.setattr(cli_module, "_process_massacre_file", fake_process)

    result = CliRunner().invoke(
        cli,
        ["massacre", str(tmp_path), "--recursive", "--workers", "1", "--no-turbo"],
    )

    assert result.exit_code == 0
    assert seen == [input_file]


def test_turbo_massacre_limits_workers_when_cuda_available(monkeypatch, tmp_path):
    for index in range(3):
        (tmp_path / f"track{index}.wav").write_bytes(b"audio")
    thread_names = []

    def fake_process(file_path, output_dir, paranoid, backup, turbo):
        thread_names.append(threading.current_thread().name)
        return {"success": True, "mode": "turbo", "output_file": str(file_path)}

    monkeypatch.setattr(cli_module, "_cuda_available_for_worker_limit", lambda: True)
    monkeypatch.setattr(cli_module, "_process_massacre_file", fake_process)

    result = CliRunner().invoke(
        cli,
        ["massacre", str(tmp_path), "--workers", "3", "--turbo"],
    )

    assert result.exit_code == 0
    assert set(thread_names) == {"MainThread"}
    assert "limiting workers to 1" in result.output.lower()


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
    assert "config_manager" not in vars(cli_module)
    assert callable(cli_module.config)


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
    input_file = tmp_path / "input.ogg"
    output_file = tmp_path / "output.ogg"
    input_file.write_bytes(b"metadata-bearing original")

    monkeypatch.setattr(
        "mmm.sanitization.metadata_cleaner.AudioSegment.from_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )

    result = MetadataCleaner().clean_file(input_file, output_file)

    assert result["success"] is False
    assert "copy fallback" in " ".join(result["errors"]).lower()
    assert not output_file.exists()


def test_flac_metadata_cleaner_uses_native_flac_verification(monkeypatch, tmp_path):
    input_file = tmp_path / "input.flac"
    output_file = tmp_path / "output.flac"
    input_file.write_bytes(b"flac bytes")
    cleaned_paths = set()

    class FakeFLAC:
        def __init__(self, path):
            self.path = Path(path)
            self.tags = {} if self.path in cleaned_paths else {"TITLE": ["secret"]}
            self.pictures = [] if self.path in cleaned_paths else [object()]

        def clear(self):
            self.tags = {}

        def clear_pictures(self):
            self.pictures = []

        def save(self):
            cleaned_paths.add(self.path)

    monkeypatch.setattr("mmm.sanitization.metadata_cleaner.FLAC", FakeFLAC)

    result = MetadataCleaner().clean_file(input_file, output_file)

    assert result["success"] is True
    assert "flac_mutagen_clear" in result["methods_used"]
    assert MetadataCleaner()._verify_metadata_present(output_file) is False


def test_wav_metadata_verifier_detects_custom_chunks(tmp_path):
    wav_file = tmp_path / "custom.wav"
    fmt_chunk = b"fmt " + (16).to_bytes(4, "little") + b"\x01\x00\x01\x00" + b"\x40\x1f\x00\x00" + b"\x80>\x00\x00" + b"\x02\x00\x10\x00"
    junk_chunk = b"JUNK" + (4).to_bytes(4, "little") + b"test"
    data_chunk = b"data" + (2).to_bytes(4, "little") + b"\x00\x00"
    payload = b"WAVE" + fmt_chunk + junk_chunk + data_chunk
    wav_file.write_bytes(b"RIFF" + (len(payload)).to_bytes(4, "little") + payload)

    assert MetadataCleaner()._verify_metadata_present(wav_file) is True


def test_metadata_verification_handles_small_files(monkeypatch, tmp_path):
    tiny_file = tmp_path / "tiny.bin"
    tiny_file.write_bytes(b"small")
    monkeypatch.setattr("mmm.sanitization.metadata_cleaner.MutagenFile", lambda _p: None)

    assert MetadataCleaner()._verify_metadata_present(tiny_file) is False


def test_watermark_detector_marks_results_as_heuristic():
    rng = np.random.default_rng(123)
    audio = rng.normal(0, 0.01, size=(8192, 1)).astype(np.float32)

    results = WatermarkDetector().detect_all(audio, 44100)

    assert results["method_results"]
    for method_result in results["method_results"].values():
        assert method_result["heuristic_indicator"] is True
        assert method_result["detector_type"] == "heuristic_threshold"
        assert "calibration" in method_result
        assert "evidence_metrics" in method_result


def test_gpu_spectral_detector_returns_evidence_in_cpu_fallback():
    detector = GPUAcceleratedWatermarkDetector()
    detector.gpu_available = False
    audio = np.random.default_rng(1).normal(0, 0.01, 8192).astype(np.float32)

    result = detector.detect_spectral_patterns_gpu(audio, 44100)

    assert result["heuristic_indicator"] is True
    assert result["detector_type"] == "heuristic_threshold"
    assert "high_frequency_ratio" in result["evidence_metrics"]


def test_spectral_cleaner_records_targeted_detector_masks():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 18000 * t)).astype(np.float32).reshape(-1, 1)
    findings = {
        "detected": [
            {
                "method": "spread_spectrum",
                "details": [
                    {
                        "channel": 0,
                        "frequency": 18000,
                        "type": "suspicious_frequency",
                    }
                ],
            }
        ]
    }

    result = SpectralCleaner().clean_watermarks(audio, sr, detector_findings=findings)

    assert "targeted_detector_masks" in result["methods_used"]
    assert result["modified_regions"]
    assert result["modified_regions"][0]["start_hz"] < 18000
    assert result["modified_regions"][0]["end_hz"] > 18000


def test_micro_timing_perturbation_moves_transient(monkeypatch):
    remover = FingerprintRemover()
    sr = 8000
    audio = np.zeros(sr, dtype=np.float32)
    audio[1000] = 1.0

    monkeypatch.setattr(np.random, "randint", lambda *_args, **_kwargs: 5)

    result = remover._micro_timing_perturbation(audio, sr)["cleaned_data"]

    assert int(np.argmax(result)) != 1000


def test_fingerprint_quality_metrics_do_not_broadcast_mono_channel():
    remover = FingerprintRemover()
    sr = 8000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    original = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(-1, 1)
    cleaned = original[:, 0].copy()

    metrics = remover._calculate_quality_metrics(original, cleaned, sr)

    assert metrics["quality_preservation"] >= 0.0
    assert np.isfinite(metrics["mfcc_distance"])


def test_forensic_report_records_hashes_and_methods(tmp_path):
    input_file = tmp_path / "input.wav"
    output_file = tmp_path / "output.wav"
    input_file.write_bytes(b"input")
    output_file.write_bytes(b"output")
    stats = {
        "processing_engine": "test",
        "methods_used": ["a", "b"],
        "passes_run": 2,
        "signal_changed": True,
    }

    report_path = write_forensic_report(
        input_file,
        output_file,
        stats,
        metadata_clean=True,
        signal_delta={"signal_changed": True},
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert '"hash_changed": true' in report_text
    assert '"methods_used": [' in report_text


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
    assert "JOB_REGISTRY_LOCK" in app.config
    assert hasattr(app.config["JOB_REGISTRY_LOCK"], "acquire")


def _wait_for_job(client, job_id, timeout=2.0):
    deadline = time.time() + timeout
    last_payload = None
    while time.time() < deadline:
        response = client.get(f"/api/job/{job_id}")
        last_payload = response.get_json()
        if last_payload.get("status") in {"complete", "failed"}:
            return response
        time.sleep(0.02)
    raise AssertionError(f"job did not finish: {last_payload}")


def test_gpu_signal_delta_rejects_unchanged_audio():
    audio = np.full((2000, 2), 0.2, dtype=np.float32)

    metrics = _verify_signal_delta(audio, audio.copy())

    assert metrics["signal_changed"] is False
    assert metrics["signal_delta_ratio"] == 0.0


def test_gpu_signal_delta_rejects_unchanged_silence():
    audio = np.zeros((2000, 2), dtype=np.float32)

    metrics = _verify_signal_delta(audio, audio.copy())

    assert metrics["signal_changed"] is False
    assert metrics["signal_delta_required"] is False


def test_gpu_signal_delta_accepts_material_change():
    audio = np.full((2000, 2), 0.2, dtype=np.float32)
    processed = audio.copy()
    processed[::2] += 1e-3

    metrics = _verify_signal_delta(audio, processed)

    assert metrics["signal_changed"] is True
    assert metrics["signal_delta_ratio"] >= 1e-4


def test_gpu_web_sanitize_rejects_unchanged_written_output(monkeypatch, tmp_path):
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"input")
    original = np.full((2000, 1), 0.2, dtype=np.float32)
    processed = original.copy()
    processed[::2] += 1e-3

    monkeypatch.setattr(gpu_web_module, "cuda_available", lambda: True)
    monkeypatch.setattr(
        gpu_web_module,
        "_process_audio_on_gpu",
        lambda *_args, **_kwargs: (processed, 1, "test-gpu"),
    )
    monkeypatch.setattr(
        gpu_web_module,
        "_sha256_file",
        lambda path: "input-hash" if Path(path) == input_file else "output-hash",
    )
    monkeypatch.setattr(
        gpu_web_module,
        "_write_audio",
        lambda _audio, _sr, output_path: Path(output_path).write_bytes(b"output"),
    )
    monkeypatch.setattr(gpu_web_module, "_metadata_clean", lambda _path: True)

    def fake_load_audio(path):
        if Path(path) == input_file:
            return original.copy(), 8000
        return original.copy(), 8000

    monkeypatch.setattr(gpu_web_module, "_load_audio", fake_load_audio)

    with pytest.raises(RuntimeError, match="Written GPU output"):
        gpu_web_module.gpu_web_sanitize(input_file, verbose=False)


def test_server_upload_runs_gpu_background_job(monkeypatch):
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    monkeypatch.setattr(server_module, "_validate_audio_content", lambda _path: True)

    def fake_gpu_web_sanitize(input_file, output_file=None, **_kwargs):
        output_path = Path(input_file).with_suffix(".clean.wav")
        output_path.write_bytes(b"clean")
        return {
            "success": True,
            "output_file": str(output_path),
            "stats": {
                "processing_engine": "gpu_cuda_web",
                "gpu_acceleration": True,
                "gpu_device": "test-gpu",
            },
        }

    monkeypatch.setattr(server_module, "gpu_web_sanitize", fake_gpu_web_sanitize)

    response = client.post(
        "/api/upload",
        data={
            "file": (BytesIO(b"audio"), "test.wav"),
            "format": "preserve",
            "paranoid": "false",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    job_response = _wait_for_job(client, response.get_json()["job_id"])
    payload = job_response.get_json()
    assert payload["status"] == "complete"
    assert payload["result"]["stats"]["processing_engine"] == "gpu_cuda_web"
    assert payload["result"]["stats"]["gpu_acceleration"] is True


def test_server_upload_accepts_flac_output_option(monkeypatch):
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()
    seen = {}

    monkeypatch.setattr(server_module, "_validate_audio_content", lambda _path: True)

    def fake_gpu_web_sanitize(input_file, output_file=None, **kwargs):
        seen["output_format"] = kwargs["output_format"]
        output_path = Path(input_file).with_suffix(".clean.flac")
        output_path.write_bytes(b"clean")
        return {
            "success": True,
            "output_file": str(output_path),
            "stats": {"processing_engine": "gpu_cuda_web"},
        }

    monkeypatch.setattr(server_module, "gpu_web_sanitize", fake_gpu_web_sanitize)

    response = client.post(
        "/api/upload",
        data={
            "file": (BytesIO(b"audio"), "test.wav"),
            "format": "flac",
            "paranoid": "false",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    job_response = _wait_for_job(client, response.get_json()["job_id"])
    assert job_response.get_json()["status"] == "complete"
    assert seen["output_format"] == "flac"


def test_server_upload_falls_back_when_gpu_fails(monkeypatch):
    import mmm.preserving_sanitizer as preserving_module

    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    monkeypatch.setattr(server_module, "_validate_audio_content", lambda _path: True)

    def fake_gpu_web_sanitize(*_args, **_kwargs):
        raise RuntimeError("cuda oom")

    def fake_preserving_sanitize(input_file, output_file=None, **_kwargs):
        output_path = Path(input_file).with_suffix(".clean.wav")
        output_path.write_bytes(b"clean")
        return {
            "success": True,
            "output_file": str(output_path),
            "stats": {"processing_time": 1.0},
        }

    monkeypatch.setattr(server_module, "gpu_web_sanitize", fake_gpu_web_sanitize)
    monkeypatch.setattr(preserving_module, "preserving_sanitize", fake_preserving_sanitize)

    response = client.post(
        "/api/upload",
        data={
            "file": (BytesIO(b"audio"), "test.wav"),
            "format": "preserve",
            "paranoid": "false",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    job_response = _wait_for_job(client, response.get_json()["job_id"])
    payload = job_response.get_json()
    assert payload["status"] == "complete"
    assert payload["result"]["stats"]["processing_engine"] == "cpu_preserving_fallback"
    assert payload["result"]["stats"]["gpu_acceleration"] is False
    assert "cuda oom" in payload["result"]["stats"]["gpu_fallback_error"]
