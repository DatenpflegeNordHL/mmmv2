# Audio Quality Engine Integration Note

## Product Line

The Audio Quality Engine is a local analysis and mastering-support pipeline for
finished stereo files. It focuses on release readiness, streaming translation,
safe mastering, mix-repair, and report generation. It does not require a DAW,
does not host external plugins, and does not require cloud services.

## First Integration Slice

The first slice adds a top-level `audio_engine` package with isolated modules:

- `io`: local WAV/FLAC/AIFF loading and writing.
- `analysis`: loudness, dynamics, spectrum, stereo, transient, segment, and
  release-readiness metrics.
- `guardrails`: central conservative limits and render validation.
- `dsp`: gain staging, highpass, FFT EQ, mono bass, light compression, subtle
  saturation, and final peak limiting.
- `naturalize`: coarse energy-segment automation before safe mastering.
- `reports`: JSON, HTML, and before/after comparison helpers.
- `reference`: optional reference-QC metrics and moderate suggestions.

## CLI Surface

The commands are attached to the existing Click root command:

```bash
mmm analyze-quality input.wav --out report.json --html report.html
mmm safe-master input.wav --out master.wav --report report.json
mmm naturalize input.wav --out naturalized_master.wav --report report.json
mmm compare-master input.wav master.wav --out before_after.json --html before_after.html
mmm reference-qc target.wav reference.wav --out reference_report.json
```

`setup.py` also exposes `mmv2=mmm.cli:cli` as an alias for the same command
surface.

## Web Surface

The Flask web UI is positioned as **MMV2 Audio Quality Engine** and presents a
local mastering console rather than a sanitizer landing page. The existing
upload/job/download routes remain in place:

- `GET /` renders the dark Audio Quality Console.
- `POST /api/upload` accepts the existing upload flow plus optional quality
  fields: `mode`, `loudness_target`, `true_peak_ceiling`,
  `sample_rate_override`, and `bit_depth_override`.
- `GET /api/job/<job_id>` returns backward-compatible job state plus quality
  fields such as `engine_version`, `metrics_before`, `metrics_after`,
  `report_artifacts`, `waveform_artifact`, and `processing_steps`.
- `GET /api/download/<token>` continues to serve temporary artifacts.

Current web modes:

- `analyze_only`: JSON/HTML quality report, no rendered master.
- `safe_master`: conservative safe-master render.
- `naturalize`: subtle segment automation followed by safe-master render.
- `full_release`: currently mapped to safe-master render with full reporting.
- omitted/legacy mode: existing GPU/preserving sanitizer compatibility path.

Deployment note for `api.datenpflege-nord.de`: keep the existing systemd service
and Cloudflare/nginx routing pointed at the same Flask app port. No new
frontend build step, plugin host, cloud job, or persistent upload storage is
required.

## Guardrail Defaults

- `max_eq_gain_db`: 2.0
- `max_dynamic_eq_gain_reduction_db`: 3.0
- `max_compressor_gain_reduction_db`: 1.5
- `max_limiter_gain_reduction_db`: 3.0
- `max_width_change_percent`: 10
- `limiter_ceiling_dbtp`: -1.5
- `preserve_sample_rate`: true
- `export_default_bit_depth`: 24

## Current Limitations

The current LUFS implementation auto-uses `pyloudnorm` when installed and falls
back to a documented RMS approximation otherwise. The safe-master limiter is a
deterministic peak ceiling scaler, not a commercial-grade true-peak limiter. The
reference-QC module reports moderate suggestions only and does not copy reference
curves.

## Next Practical Steps

1. Add `pyloudnorm` as an explicit dependency when standards-grade LUFS becomes
   required for production reports.
2. Add codec preview helpers through optional FFmpeg calls.
3. Expand HTML reports with plots once the JSON schema stabilizes.
4. Add preset loading for the JSON files under `audio_engine/presets`.
5. Keep Demucs/Open-Unmix strictly optional for analysis-only diagnostics.
