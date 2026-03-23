# 🎵 Melodic Metadata Massacrer (MMM)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)]()

> *"In the symphony of digital rights, we are the conductors of chaos."* 🎼⚡

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/6b342199-dbdd-446b-8c6f-983e50ef5625" />

**MMM** is a Python CLI tool that strips metadata, disrupts watermark patterns, and applies spectral perturbation to MP3 and WAV audio files, making it harder for AI-detection systems to identify them as machine-generated.

> **Note**: Sanitization re-encodes audio (MP3 at 320kbps, WAV at PCM_16) and applies subtle spectral modifications. This is not lossless — audio quality is preserved but not bit-identical.

## 🎭 Features

### Core Capabilities
- **Complete Metadata Annihilation**: Removes ID3, RIFF INFO, FLAC tags, and custom chunks
- **AI Watermark Detection**: Identifies spread spectrum, echo-based, and statistical watermarks
- **Spectral Cleaning**: Advanced frequency-domain watermark removal
- **Fingerprint Elimination**: Normalizes AI-generated statistical patterns
- **Paranoid Mode**: Maximum destruction with multiple cleaning passes
- **Batch Processing**: Parallel processing of entire directories
- **Verification Engine**: Before/after comparison with forensic reporting

### Detection Methods
- Spread spectrum watermarks
- Echo-based signatures
- Statistical pattern analysis
- Phase modulation detection
- Amplitude modulation analysis
- Frequency domain anomalies

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/geeknik/mmm.git
cd mmm

# Create virtual environment (Python 3.9+)
python3 -m venv mmm_env
source mmm_env/bin/activate  # On Windows: mmm_env\Scripts\activate

# Install the package and all dependencies
pip install -e .
```

### GPU Acceleration (Optional)

For faster analysis on systems with NVIDIA GPUs:

```bash
# Install GPU acceleration packages (NVIDIA GPU required)
pip install cupy-cuda12x torch torchaudio

# Verify GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- 4GB+ VRAM recommended
- CUDA 12.x compatible drivers

### Performance

Processing speed depends on hardware, file length, and mode:

| Mode | 3.5 min MP3 (4.9 MB) | Notes |
|------|----------------------|-------|
| Turbo (CPU) | ~70s | 3x real-time, `--turbo` flag |
| Turbo + Paranoid (CPU) | ~99s | 2.2x real-time, all stealth flags |
| Turbo (GPU) | Significantly faster | Requires NVIDIA GPU + CUDA |
| Regular (no turbo) | Very slow | Runs 6 O(N) watermark detection methods; not recommended for files > 1 min |

### Basic Usage

```bash
# Sanitize a single file (recommended: turbo + paranoid)
mmm obliterate music.mp3 --turbo --paranoid -o clean_music.mp3

# Quick sanitize without paranoid
mmm obliterate music.mp3 --turbo -o clean_music.mp3

# Batch process directory
mmm massacre /path/to/music --paranoid --workers 8

# Analyze file without modifying
mmm analyze music.mp3 --turbo
```

## 🔧 Configuration

MMM uses YAML configuration files for customization:

```yaml
# ~/.config/mmm/config.yaml
paranoia_level: medium
preserve_quality: high
watermark_detection:
  - spread_spectrum
  - echo_based
  - statistical
output_format: preserve
backup_originals: true
```

### Presets

- **`stealth`**: Maximum paranoia, quality preservation
- **`stealth-plus`**: Stealth with advanced flags optimized for detector evasion
- **`fast`**: Quick processing, basic cleaning
- **`quality`**: Preserve maximum audio quality
- **`research`**: Deep analysis, detailed logging

```bash
# Use preset
mmm config preset stealth

# Create custom preset
mmm config create my_preset --paranoid maximum --quality high
```

## 🎯 Commands

### `obliterate`
Complete sanitization of individual files

```bash
mmm obliterate INPUT_FILE [OPTIONS]

Options:
  -o, --output PATH     Output file path
  --paranoid           Maximum destruction mode
  --verify             Verify watermark removal
  --backup             Create backup of original
  --format FORMAT      Output format (preserve/mp3/wav)
  --turbo              Enable turbo mode (faster, uses preserving sanitizer)
```

### `massacre`
Batch processing of directories

```bash
mmm massacre DIRECTORY [OPTIONS]

Options:
  -d, --output-dir PATH  Output directory
  -e, --extension TEXT   File extensions (multiple)
  -w, --workers INT      Parallel workers
  --paranoid            Paranoid mode
  --backup              Create backups
```

### `analyze`
Forensic analysis without modification

```bash
mmm analyze INPUT_FILE                # Regular mode (slow on long files)
mmm analyze INPUT_FILE --turbo        # Turbo mode (faster)
```

### `config`
Configuration management

```bash
mmm config              Show current config
mmm config preset NAME  Apply preset
mmm config list         List available presets
mmm config create NAME  Create custom preset
mmm config delete NAME  Delete custom preset
mmm config reset        Reset to defaults
```

## 🎛️ Advanced Stealth Flags

These are opt-in, fine-grained toggles for research tuning. Defaults keep audio quality high; enable selectively:

- `--gated-resample-nudge/--no-gated-resample-nudge` (default off): ultra-tiny resample up/down applied only on higher-energy segments (minimal audibility, good stealth).
- `--phase-noise/--no-phase-noise` (default on): tiny FFT phase noise.
- `--phase-swirl/--no-phase-swirl` (default on): light all-pass swirl.
- `--phase-dither/--no-phase-dither` (default on), `--comb-mask/--no-comb-mask`, `--transient-shift/--no-transient-shift`: earlier experimental steps (may affect audio; leave off unless testing).
- `--masked-hf-phase/--no-masked-hf-phase` (default off): HF-only masked phase noise.
- `--micro-eq-flutter/--no-micro-eq-flutter` (default off): RMS-gated, <0.013 dB band flutter.
- `--hf-decorrelate/--no-hf-decorrelate` (default off): decorrelate only 12–16 kHz band.
- `--refined-transient/--no-refined-transient` (default off): ultra-small, onset-gated shifts.
- `--adaptive-transient/--no-adaptive-transient` (default off): onset-strength adaptive micro-shifts (~0.03–0.08 ms) with light blending.

### Maximum stealth (all flags enabled)

```bash
mmm obliterate input.mp3 -o output.mp3 --turbo --paranoid \
  --masked-hf-phase --gated-resample-nudge --micro-eq-flutter \
  --hf-decorrelate --adaptive-transient
```

### Preset shortcut

```bash
mmm config preset stealth-plus
```

Preset `stealth-plus` includes advanced flags:
- phase_dither=False, comb_mask=False, transient_shift=False
- phase_swirl=False, masked_hf_phase=False, resample_nudge=False
- gated_resample_nudge=True, phase_noise=True
- micro_eq_flutter=False, hf_decorrelate=False
- refined_transient=False, adaptive_transient=False

Notes on pattern suppression counts:
- "Patterns Found/Suppressed" in sanitization results come from the spectral cleaner's suppression actions (e.g., attenuating suspicious bands/patterns) and do not imply detector-verified watermarks unless the detector reports them.
- Verification threat counts include metadata/container anomalies and detector findings; if the detector reports zero watermarks, remaining threats are likely metadata/binary anomalies rather than confirmed watermarks.

## 🧪 Detector Notes (Research)

We test against third-party detectors to understand robustness (not to guarantee evasion). Results on a Suno-generated 3.5 min MP3 (March 2026):

**SubmitHub / SHLabs results:**

| Mode | Verdict | Spectral: Human | Spectral: Pure AI | Temporal: Human | Temporal: Pure AI |
|------|---------|-----------------|-------------------|-----------------|-------------------|
| Turbo (default flags) | Possible AI Detected | 15% | 36% | 44% | 6% |
| Turbo + Paranoid + All Flags | **Inconclusive** | **43%** | **15%** | **65% (likely)** | **1% (highly unlikely)** |

- Paranoid mode with all stealth flags shifted the verdict from "Possible AI Detected" to "Inconclusive"
- Temporal: Pure AI dropped to 1% ("highly unlikely")
- Aggressive stacks (phase dither / comb mask / transient shift) degraded audio; not recommended

Always audition audio locally before running external checks.

## 🛡️ Legal & Ethical Notice

⚠️ **IMPORTANT**: This tool is designed **exclusively for authorized security research and educational purposes**.

- Use only on files you own or have explicit permission to modify
- You are responsible for compliance with applicable laws and terms of service
- The developers do not condone or support copyright infringement
- This tool demonstrates vulnerabilities in watermarking systems for research purposes

## 📊 Technical Details

### Architecture

```
┌─────────────────────────────────────────┐
│                CLI Layer                │
│  Click-based interface with personality  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│            Core Processing              │
│  • AudioSanitizer main engine          │
│  • PreservingSanitizer (turbo mode)    │
│  • FileProcessor for batch operations   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          Detection Modules              │
│  • WatermarkDetector                   │
│  • MetadataScanner                     │
│  • StatisticalAnalyzer                  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Sanitization Modules            │
│  • MetadataCleaner                     │
│  • SpectralCleaner                     │
│  • FingerprintRemover                  │
└─────────────────────────────────────────┘
```

### Dependencies

- **Core**: Click, NumPy, SciPy
- **Audio**: Librosa, PyDub, SoundFile, Resampy
- **Metadata**: Mutagen
- **UI**: Rich, Colorama
- **GPU (optional)**: CuPy, PyTorch

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests (note: some tests are skipped due to high memory usage)
pytest tests/ -v

# Run specific test file
pytest tests/test_audio_sanitizer.py -v
```

> **Warning**: The full test suite can consume significant RAM due to audio processing operations. Run individual test files rather than the full suite on memory-constrained systems.

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style

```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black mmm/

# Lint code
flake8 mmm/

# Type checking
mypy mmm/
```

## 🔬 Research Applications

MMM is designed for academic and security research:

- **Watermark Vulnerability Assessment**: Test resilience of audio watermarking systems
- **Privacy Research**: Study audio fingerprinting and tracking
- **Educational**: Demonstrate audio steganography techniques
- **Security Auditing**: Verify effectiveness of watermark removal

## 🎨 CLI Experience

MMM features a unique hacker-aesthetic interface:

```
┌─────────────────────────────────────────────┐
│  ♪♫ MELODIC METADATA MASSACRER v2.0 ♫♪    │
│     "Making AI detectors cry since 2025"    │
└─────────────────────────────────────────────┘

🔍 Scanning: dystopian_symphony.mp3
😈 Found 1 threats... time to DELETE THEM ALL!
🌊 Beginning audio sanitization protocol...
✨ File sanitized! Your AI overlords will never know... 🤫
```

## 📝 License

See the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Open-source audio processing community
- Security researchers in digital watermarking
- Python audio processing ecosystem

---

**Remember**: With great audio comes great responsibility. Use wisely. 🎼💀
