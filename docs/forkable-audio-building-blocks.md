# Forkable Audio Building Blocks

This note tracks practical open-source building blocks for the local Audio
Quality Engine. The current implementation intentionally uses the dependencies
already present in MMMv2 first: `soundfile`, `numpy`, `scipy`, and optional
`librosa`.

| Project | Source | License | Purpose | Fit for MMMv2 |
| --- | --- | --- | --- | --- |
| pyloudnorm | https://github.com/csteinmetz1/pyloudnorm | MIT-style permissive | ITU-R BS.1770 loudness measurement. | Good future direct dependency for more accurate integrated LUFS/LRA. Current code has a fallback and auto-uses it if installed. |
| libebur128 | https://github.com/jiixyj/libebur128 | MIT | Fast EBU R128 loudness library. | Good future native dependency or subprocess helper if packaging native libraries is acceptable. |
| Matchering | https://github.com/sergree/matchering | GPL-3.0 | Reference matching and mastering workflow. | Inspiration or separately isolated tool only; avoid direct dependency in the core package unless GPL obligations are intended. |
| librosa | https://github.com/librosa/librosa | ISC | Music/audio analysis, onsets, tempo, features. | Already a dependency; use selectively for heavier MIR features, not basic reports. |
| scipy.signal | https://github.com/scipy/scipy | BSD-3-Clause | Filters, envelopes, resampling, DSP primitives. | Direct dependency already present; preferred for first-party DSP modules. |
| soundfile/libsndfile | https://github.com/bastibe/python-soundfile / https://github.com/libsndfile/libsndfile | soundfile BSD-style; libsndfile LGPL-2.1 | WAV/FLAC/AIFF I/O. | Direct dependency already present and used for local file I/O. |
| FFmpeg | https://ffmpeg.org/legal.html | LGPL by default, GPL if built with GPL parts | Format conversion, codec preview, export helpers. | Optional executable dependency only; do not vendor. Keep CLI calls deterministic. |
| Demucs | https://github.com/facebookresearch/demucs | MIT | Source separation / pseudo-stem diagnostics. | Future optional diagnostic feature only; not a default mastering step. |
| Open-Unmix | https://github.com/sigsep/open-unmix-pytorch | MIT | Source separation / pseudo-stem diagnostics. | Future optional diagnostic feature only; not a default mastering step. |
| Essentia | https://github.com/MTG/essentia | AGPL-3.0, commercial licensing available | Broad MIR feature extraction. | Powerful but license-sensitive. Keep optional and isolated unless deployment/licensing is explicitly approved. |

## Current Decision

The first Audio Quality Engine pass uses only local, reproducible code and
existing dependencies. It does not host VST/AU/LV2 plugins, does not require a
DAW, and does not require cloud services.

## Recommended Adoption Order

1. Keep `soundfile`, `numpy`, and `scipy.signal` as the stable core.
2. Add `pyloudnorm` later for standards-grade LUFS if report precision becomes
   product-critical.
3. Use FFmpeg only as an optional system executable for conversion and codec
   preview.
4. Treat Matchering as architecture/reference inspiration unless GPL integration
   is an intentional product decision.
5. Keep Demucs/Open-Unmix as opt-in diagnostics, not render-path dependencies.
6. Evaluate Essentia only with explicit AGPL/commercial-license review.
