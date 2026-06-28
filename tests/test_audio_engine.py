import json
import time
import builtins
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from audio_engine.analysis.readiness import analyze_quality
from audio_engine.dsp.pipeline import render_safe_master
from audio_engine.io.writer import write_audio
from audio_engine.naturalize.movement import render_naturalized_master
from audio_engine.reports.before_after import compare_masters
from mmm.cli import cli
import mmm.server as server_module
from mmm.server import create_app


def _write_stereo_test_file(path: Path, sample_rate: int = 44100) -> Path:
    duration = 1.25
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    left = 0.28 * np.sin(2 * np.pi * 110 * t) + 0.08 * np.sin(2 * np.pi * 6400 * t)
    right = 0.25 * np.sin(2 * np.pi * 112 * t) + 0.06 * np.sin(2 * np.pi * 6400 * t)
    audio = np.column_stack([left, right]).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_24")
    return path


def test_analyze_quality_reports_core_metrics(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")

    report = analyze_quality(input_file)

    assert report["duration_seconds"] > 1.0
    assert report["sample_rate"] == 44100
    assert report["channels"] == 2
    assert "integrated_lufs" in report
    assert "estimated_true_peak_dbtp" in report
    assert "short_term_loudness_curve" in report
    assert "low_mid_120_350_hz" in report["band_energy"]
    assert report["release_readiness"]["score"] >= 0
    assert report["release_readiness"]["checks"]


def test_safe_master_writes_audio_and_before_after_report(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")
    output_file = tmp_path / "master.wav"

    report = render_safe_master(input_file, output_file)

    assert output_file.exists()
    assert report["action"] == "safe_master"
    assert report["guardrails"]["limits"]["export_default_bit_depth"] == 24
    assert report["guardrails"]["passed"] is True
    assert report["before"]["sample_rate"] == report["after"]["sample_rate"]
    assert "integrated_lufs_delta" in report["comparison"]


def test_safe_master_honors_output_sample_rate(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav", sample_rate=44100)
    output_file = tmp_path / "master.wav"

    report = render_safe_master(input_file, output_file, output_sample_rate=48000)
    info = sf.info(output_file)

    assert info.samplerate == 48000
    assert report["after"]["sample_rate"] == 48000
    assert any(step["name"] == "sample_rate_convert" for step in report["processing_steps"])


def test_safe_master_honors_target_lufs_parameter(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")
    output_file = tmp_path / "master.wav"

    report = render_safe_master(input_file, output_file, target_lufs=-16.0)
    input_gain_step = next(
        step for step in report["processing_steps"] if step["name"] == "input_gain_staging"
    )

    expected_gain = np.clip(-16.0 - report["before"]["integrated_lufs"], -3.0, 3.0)
    assert input_gain_step["gain_db"] == expected_gain


def test_write_audio_supports_metadata_free_mp3_export(tmp_path):
    output_file = tmp_path / "master.mp3"
    sample_rate = 44100
    t = np.linspace(0, 0.25, int(sample_rate * 0.25), endpoint=False)
    audio = np.column_stack(
        [
            0.1 * np.sin(2 * np.pi * 220 * t),
            0.1 * np.sin(2 * np.pi * 221 * t),
        ]
    ).astype(np.float32)

    write_audio(output_file, audio, sample_rate)

    assert output_file.exists()
    assert output_file.stat().st_size > 0
    assert output_file.read_bytes()[:3] != b"ID3"


def test_naturalize_records_automation_and_writes_master(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")
    output_file = tmp_path / "naturalized.wav"

    report = render_naturalized_master(input_file, output_file)

    assert output_file.exists()
    assert report["action"] == "naturalize"
    assert report["automation"]
    assert all("gain_db" in point for point in report["automation"])


def test_compare_masters_returns_key_deltas(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")
    output_file = tmp_path / "master.wav"
    render_safe_master(input_file, output_file)

    report = compare_masters(input_file, output_file)

    assert report["action"] == "compare_master"
    assert "estimated_true_peak_dbtp_delta" in report["comparison"]


def test_cli_analyze_quality_writes_json_and_html(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")
    json_report = tmp_path / "report.json"
    html_report = tmp_path / "report.html"

    result = CliRunner().invoke(
        cli,
        [
            "analyze-quality",
            str(input_file),
            "--out",
            str(json_report),
            "--html",
            str(html_report),
        ],
    )

    assert result.exit_code == 0
    assert json_report.exists()
    assert html_report.exists()
    data = json.loads(json_report.read_text(encoding="utf-8"))
    assert data["release_readiness"]["score"] >= 0


def test_cli_safe_master_and_compare_write_reports(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")
    master_file = tmp_path / "master.wav"
    master_report = tmp_path / "master.json"
    compare_report = tmp_path / "compare.json"

    master_result = CliRunner().invoke(
        cli,
        [
            "safe-master",
            str(input_file),
            "--out",
            str(master_file),
            "--report",
            str(master_report),
        ],
    )
    compare_result = CliRunner().invoke(
        cli,
        [
            "compare-master",
            str(input_file),
            str(master_file),
            "--out",
            str(compare_report),
        ],
    )

    assert master_result.exit_code == 0
    assert compare_result.exit_code == 0
    assert master_file.exists()
    assert master_report.exists()
    assert compare_report.exists()


def test_web_ui_renders_audio_quality_console():
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "MMV2 Audio Quality Engine" in html
    assert "Release-Ready Stereo Mastering" in html
    assert "ENGINE ACTIVE" in html
    assert "Analyze &amp; Master" in html
    assert "Metadata Clean" in html
    assert "Aggressive full clean profile" not in html
    assert "Melodic Metadata Massacrer" not in html
    assert "Browser-based audio sanitizer" not in html
    for forbidden in ("watermark", "fingerprint", "evasion", "bypass", "Legacy GPU"):
        assert forbidden not in html
    assert 'class="visualizer-card"' in html
    assert 'id="waveCanvas"' in html
    assert 'id="playPreviewBtn"' in html
    assert 'id="previewAudio"' in html
    assert "/api/preview/" in html
    assert "requestAnimationFrame" in html
    assert "createMediaElementSource" in html


def test_web_ui_keeps_quality_mode_when_legacy_profile_checked():
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "const mode = selectedMode;" in html
    assert "paranoid ? 'legacy_sanitize' : selectedMode" not in html
    assert "legacy_sanitize: 'Legacy GPU Clean'" not in html


def test_web_status_exposes_quality_engine_schema():
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    response = client.get("/api/status")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["engine_version"] == "1.045"
    assert "metadata_clean" in payload["metadata_modes"]
    assert "safe_master" in payload["quality_modes"]
    assert "streaming_safe" in payload["loudness_targets"]
    assert "gpu" in payload
    assert "available" in payload["gpu"]


def test_web_status_survives_missing_torch(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    response = client.get("/api/status")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["gpu_available"] is False
    assert payload["gpu"]["backend"] == "cpu"
    assert "torch missing" in payload["gpu"]["error"]


def test_web_quality_safe_master_job_returns_metrics_and_reports(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    with input_file.open("rb") as handle:
        response = client.post(
            "/api/upload",
            data={
                "file": (handle, "mix.wav"),
                "format": "wav",
                "mode": "safe_master",
                "loudness_target": "streaming_safe",
                "true_peak_ceiling": "-1.5",
                "sample_rate_override": "preserve",
                "bit_depth_override": "24",
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    for _ in range(40):
        job_response = client.get(f"/api/job/{job_id}")
        payload = job_response.get_json()
        if payload["status"] == "complete":
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"quality job did not complete: {payload}")

    result = payload["result"]
    assert result["engine_version"] == "1.045"
    assert result["mode"] == "safe_master"
    assert result["download_token"]
    assert result["metrics_before"]["sample_rate"] == 44100
    assert result["metrics_after"]["release_readiness"]["score"] >= 0
    assert result["report_artifacts"]["json_download_token"]
    assert result["report_artifacts"]["html_download_token"]


def test_web_metadata_clean_job_runs_metadata_only_export(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    with input_file.open("rb") as handle:
        response = client.post(
            "/api/upload",
            data={
                "file": (handle, "mix.wav"),
                "format": "preserve",
                "mode": "metadata_clean",
                "paranoid": "true",
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    for _ in range(40):
        job_response = client.get(f"/api/job/{job_id}")
        payload = job_response.get_json()
        if payload["status"] == "complete":
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"metadata clean job did not complete: {payload}")

    result = payload["result"]
    stats = result["stats"]
    assert result["mode"] == "metadata_clean"
    assert result["download_token"]
    assert result["metrics_before"] is None
    assert result["metrics_after"] is None
    assert stats["processing_engine"] == "metadata_clean_export"
    assert stats["gpu_acceleration"] is False
    assert stats["metadata_clean"] is True
    assert all("watermark" not in method for method in stats["methods_used"])
    assert all("fingerprint" not in method for method in stats["methods_used"])


def test_web_legacy_sanitize_alias_maps_to_metadata_clean_export(tmp_path):
    input_file = _write_stereo_test_file(tmp_path / "input.wav")
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    with input_file.open("rb") as handle:
        response = client.post(
            "/api/upload",
            data={
                "file": (handle, "mix.wav"),
                "format": "preserve",
                "mode": "legacy_sanitize",
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    for _ in range(40):
        job_response = client.get(f"/api/job/{job_id}")
        payload = job_response.get_json()
        if payload["status"] == "complete":
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"legacy alias job did not complete: {payload}")

    assert payload["result"]["mode"] == "metadata_clean"
    assert payload["result"]["stats"]["processing_engine"] == "metadata_clean_export"


def test_web_preview_endpoint_does_not_consume_download_token(tmp_path):
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()
    artifact = Path(app.config["UPLOAD_FOLDER"]) / "preview.txt"
    artifact.write_text("preview-data", encoding="utf-8")
    token = server_module._register_download(app, artifact, "preview.txt")

    preview = client.get(f"/api/preview/{token}")
    download = client.get(f"/api/download/{token}")

    assert preview.status_code == 200
    assert preview.get_data(as_text=True) == "preview-data"
    assert download.status_code == 200
    assert download.get_data(as_text=True) == "preview-data"


def test_quality_output_extension_supports_mp3(tmp_path):
    assert server_module._quality_output_extension(tmp_path / "mix.wav", "mp3") == ".mp3"


def test_web_rejects_unsupported_quality_upload_extension():
    app = create_app(max_file_size=1024 * 1024)
    client = app.test_client()

    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(b"not audio"), "notes.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]
