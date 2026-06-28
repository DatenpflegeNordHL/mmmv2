"""
Fingerprint remover for eliminating statistical AI fingerprints
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Any, Tuple
import librosa


class FingerprintRemover:
    """
    Removes statistical fingerprints that identify AI-generated audio
    """

    def __init__(
        self,
        paranoid_mode: bool = False,
        target_rms: float = 0.15,
        entropy_range: tuple = (6.0, 9.0),
        kurtosis_range: tuple = (1.5, 4.0),
        skewness_range: tuple = (-0.3, 0.3),
        temporal_variance: tuple = (0.01, 0.15),
    ):
        self.paranoid_mode = paranoid_mode
        self.target_rms = target_rms
        self.human_audio_targets = {
            "entropy_range": entropy_range,
            "kurtosis_range": kurtosis_range,
            "skewness_range": skewness_range,
            "temporal_variance": temporal_variance,
        }

    def remove_fingerprints(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Remove statistical AI fingerprints from audio

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate

        Returns:
            Dict containing fingerprint removal results
        """
        result = {
            "cleaned_audio": audio_data.copy(),
            "fingerprints_detected": [],
            "removal_methods": [],
            "quality_metrics": {},
            "temporal_metrics": [],
        }

        # Ensure stereo handling
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=1)

        cleaned_channels = []

        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]
            cleaned_channel = channel_data.copy()

            # Method 1: Statistical normalization
            stats_result = self._normalize_statistics(cleaned_channel)
            cleaned_channel = stats_result["cleaned_data"]
            if stats_result["fingerprints_detected"]:
                result["fingerprints_detected"].extend(
                    stats_result["fingerprints_detected"]
                )
            result["removal_methods"].append("statistical_normalization")

            # Method 2: Temporal randomization
            temporal_result = self._temporal_randomization(cleaned_channel, sample_rate)
            cleaned_channel = temporal_result["cleaned_data"]
            result["removal_methods"].append("temporal_randomization")

            # Method 3: Phase randomization
            if self.paranoid_mode:
                phase_result = self._phase_randomization(cleaned_channel)
                cleaned_channel = phase_result["cleaned_data"]
                result["removal_methods"].append("phase_randomization")

            # Method 4: Micro-timing perturbation
            timing_result = self._micro_timing_perturbation(
                cleaned_channel, sample_rate
            )
            cleaned_channel = timing_result["cleaned_data"]
            result["removal_methods"].append("micro_timing_perturbation")
            if timing_result.get("details"):
                result["temporal_metrics"].extend(timing_result["details"])

            grid_result = self._temporal_grid_perturbation(
                cleaned_channel, sample_rate
            )
            cleaned_channel = grid_result["cleaned_data"]
            result["removal_methods"].append("temporal_grid_perturbation")
            if grid_result.get("details"):
                result["temporal_metrics"].extend(grid_result["details"])

            # Method 5: Human-like imperfections
            if self.paranoid_mode:
                imperfection_result = self._add_human_imperfections(
                    cleaned_channel, sample_rate
                )
                cleaned_channel = imperfection_result["cleaned_data"]
                result["removal_methods"].append("human_imperfections")

            cleaned_channels.append(cleaned_channel)

        # Reconstruct multi-channel audio
        if len(cleaned_channels) == 1:
            result["cleaned_audio"] = cleaned_channels[0]
        else:
            result["cleaned_audio"] = np.column_stack(cleaned_channels)

        # Calculate quality metrics
        result["quality_metrics"] = self._calculate_quality_metrics(
            audio_data, result["cleaned_audio"], sample_rate
        )

        return result

    def _normalize_statistics(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Normalize statistical characteristics to human-like levels"""
        result = {"cleaned_data": audio_data.copy(), "fingerprints_detected": []}

        # Calculate current statistics
        current_skewness = self._calculate_skewness(audio_data)
        current_kurtosis = self._calculate_kurtosis(audio_data)
        current_entropy = entropy(np.histogram(audio_data, bins=100)[0] + 1e-10)

        # Detect statistical anomalies
        anomalies = []

        if abs(current_skewness) > self.human_audio_targets["skewness_range"][1]:
            anomalies.append(
                {
                    "type": "skewness_anomaly",
                    "value": current_skewness,
                    "target_range": self.human_audio_targets["skewness_range"],
                }
            )

        if not (
            self.human_audio_targets["kurtosis_range"][0]
            <= current_kurtosis
            <= self.human_audio_targets["kurtosis_range"][1]
        ):
            anomalies.append(
                {
                    "type": "kurtosis_anomaly",
                    "value": current_kurtosis,
                    "target_range": self.human_audio_targets["kurtosis_range"],
                }
            )

        if not (
            self.human_audio_targets["entropy_range"][0]
            <= current_entropy
            <= self.human_audio_targets["entropy_range"][1]
        ):
            anomalies.append(
                {
                    "type": "entropy_anomaly",
                    "value": current_entropy,
                    "target_range": self.human_audio_targets["entropy_range"],
                }
            )

        result["fingerprints_detected"] = anomalies

        if anomalies:
            # Apply statistical correction
            corrected_data = audio_data.copy()

            # Normalize amplitude to human-like distribution
            corrected_data = self._apply_amplitude_normalization(corrected_data)

            # Add subtle randomness to break statistical patterns
            noise_level = np.std(corrected_data) * 0.001  # Very subtle
            statistical_noise = np.random.normal(0, noise_level, len(corrected_data))
            corrected_data += statistical_noise

            result["cleaned_data"] = corrected_data

        return result

    def _temporal_randomization(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Apply subtle temporal randomization"""
        result = {"cleaned_data": audio_data.copy()}

        # Calculate micro-variations to add
        # Sample-level timing jitter (sub-sample precision)
        jitter_samples = np.random.normal(0, 0.1, len(audio_data))

        # Apply timing jitter using interpolation
        from scipy.interpolate import interp1d

        original_indices = np.arange(len(audio_data))
        jittered_indices = original_indices + jitter_samples

        # Clamp indices to valid range
        jittered_indices = np.clip(jittered_indices, 0, len(audio_data) - 1)

        # Interpolate to apply jitter
        f = interp1d(
            original_indices, audio_data, kind="cubic", bounds_error=False, fill_value=0
        )
        result["cleaned_data"] = f(jittered_indices)

        return result

    def _phase_randomization(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Apply phase randomization to disrupt watermarks"""
        result = {"cleaned_data": audio_data.copy()}

        # Compute FFT
        fft_data = np.fft.fft(audio_data)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)

        # Add small random phase perturbations
        phase_noise = np.random.normal(0, 0.01, len(phase))
        modified_phase = phase + phase_noise

        # Reconstruct with modified phase
        modified_fft = magnitude * np.exp(1j * modified_phase)
        result["cleaned_data"] = np.real(np.fft.ifft(modified_fft))

        return result

    def _micro_timing_perturbation(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Apply micro-timing perturbations to break AI timing patterns"""
        result = {"cleaned_data": audio_data.copy(), "details": []}

        # Identify transient points (onsets, attacks)
        # Use high-frequency content to find transients
        cutoff_hz = min(5000.0, sample_rate * 0.45)
        if cutoff_hz <= 0:
            return result
        high_freq = signal.sosfilt(
            signal.butter(4, cutoff_hz, "hp", fs=sample_rate, output="sos"), audio_data
        )
        envelope = np.abs(signal.hilbert(high_freq))

        # Find peaks in envelope (transients)
        peaks, _ = signal.find_peaks(envelope, height=np.max(envelope) * 0.1)

        # Apply micro-timing shifts to transients
        shift_range_samples = max(
            1, int((0.0018 if self.paranoid_mode else 0.001) * sample_rate)
        )
        shifted_transients = 0
        attempted_transients = 0

        for peak in peaks:
            if 50 < peak < len(audio_data) - 50:  # Avoid edges
                attempted_transients += 1
                # Random small shift
                shift_samples = np.random.randint(
                    -shift_range_samples, shift_range_samples + 1
                )

                if shift_samples != 0:
                    # Apply local time warping using the computed target indices.
                    window_start = max(0, peak - 50)
                    window_end = min(len(audio_data), peak + 50)
                    window_size = window_end - window_start

                    # Create monotonic target indices with a Gaussian shift centered
                    # on the transient. Interpolating back onto the original grid
                    # moves the transient instead of merely resampling to same length.
                    original_indices = np.arange(window_size)
                    center = peak - window_start
                    perturbation = shift_samples * np.exp(
                        -0.5 * ((np.arange(window_size) - center) / 10) ** 2
                    )
                    target_indices = np.clip(
                        original_indices + perturbation,
                        0,
                        window_size - 1,
                    )

                    window_data = result["cleaned_data"][window_start:window_end].copy()
                    order = np.argsort(target_indices)
                    shifted_window = np.interp(
                        original_indices,
                        target_indices[order],
                        window_data[order],
                        left=window_data[0],
                        right=window_data[-1],
                    )
                    result["cleaned_data"][window_start:window_end] = shifted_window
                    shifted_transients += 1

        result["details"].append(
            {
                "metric": "micro_timing_transient_shift",
                "attempted_transients": int(attempted_transients),
                "shifted_transients": int(shifted_transients),
                "max_shift_samples": int(shift_range_samples),
            }
        )

        return result

    def _temporal_grid_perturbation(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Break machine-regular onset grids with local, bounded timing offsets.

        AI music detectors often score very consistent onset intervals as a
        temporal fingerprint. This method detects low-variance onset spacing and
        applies small per-onset shifts, with stronger but still bounded shifts in
        paranoid mode.
        """
        result = {"cleaned_data": audio_data.copy(), "details": []}
        if audio_data.size < max(256, sample_rate // 10) or sample_rate <= 0:
            return result

        envelope = np.abs(audio_data)
        smooth_window = max(3, int(sample_rate * 0.006))
        kernel = np.ones(smooth_window) / smooth_window
        envelope = np.convolve(envelope, kernel, mode="same")
        peak_height = max(float(np.max(envelope)) * 0.18, float(np.percentile(envelope, 88)))
        min_distance = max(1, int(sample_rate * 0.055))
        peaks, _ = signal.find_peaks(
            envelope,
            height=peak_height,
            distance=min_distance,
        )
        if len(peaks) < 4:
            return result

        intervals = np.diff(peaks)
        interval_cv_before = float(np.std(intervals) / (np.mean(intervals) + 1e-12))
        regular_grid = interval_cv_before < (0.11 if self.paranoid_mode else 0.08)
        if not regular_grid:
            result["details"].append(
                {
                    "metric": "temporal_grid_regularization",
                    "regular_grid_detected": False,
                    "interval_cv_before": interval_cv_before,
                    "shifted_onsets": 0,
                }
            )
            return result

        max_shift = max(
            1, int((0.006 if self.paranoid_mode else 0.0035) * sample_rate)
        )
        window_radius = max(24, int((0.018 if self.paranoid_mode else 0.012) * sample_rate))
        shifted_positions: List[int] = []
        for index, peak in enumerate(peaks[1:-1], start=1):
            if peak <= window_radius or peak >= len(audio_data) - window_radius:
                continue
            direction = -1 if index % 2 == 0 else 1
            random_component = np.random.randint(max(1, max_shift // 3), max_shift + 1)
            shift_samples = direction * int(random_component)
            self._shift_transient_window(
                result["cleaned_data"], int(peak), window_radius, shift_samples
            )
            shifted_positions.append(int(peak + shift_samples))

        if shifted_positions:
            adjusted_peaks = peaks.copy()
            adjusted_peaks[1 : 1 + len(shifted_positions)] = shifted_positions
            adjusted_intervals = np.diff(np.sort(adjusted_peaks))
            interval_cv_after = float(
                np.std(adjusted_intervals)
                / (np.mean(adjusted_intervals) + 1e-12)
            )
        else:
            interval_cv_after = interval_cv_before

        result["details"].append(
            {
                "metric": "temporal_grid_regularization",
                "regular_grid_detected": True,
                "onset_count": int(len(peaks)),
                "shifted_onsets": int(len(shifted_positions)),
                "max_shift_samples": int(max_shift),
                "interval_cv_before": interval_cv_before,
                "interval_cv_after": interval_cv_after,
            }
        )
        return result

    def _shift_transient_window(
        self,
        audio_data: np.ndarray,
        center: int,
        radius: int,
        shift_samples: int,
    ) -> None:
        """Shift one transient region in-place with crossfaded boundaries."""
        start = max(0, center - radius)
        end = min(len(audio_data), center + radius)
        if end - start < 8 or shift_samples == 0:
            return

        window = audio_data[start:end].copy()
        indices = np.arange(window.size)
        shifted = np.interp(
            np.clip(indices - shift_samples, 0, window.size - 1),
            indices,
            window,
            left=window[0],
            right=window[-1],
        )
        fade = np.hanning(window.size)
        fade = np.maximum(fade, 0.05)
        audio_data[start:end] = window * (1.0 - fade) + shifted * fade

    def _add_human_imperfections(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Add subtle human-like imperfections"""
        result = {"cleaned_data": audio_data.copy()}

        # Micro-velocity variations (simulating human performance)
        velocity_variation = 1.0 + np.random.normal(0, 0.002, len(audio_data))
        result["cleaned_data"] *= velocity_variation

        # Subtle pitch drift (more human-like)
        drift_rate = np.random.normal(0, 0.0001, len(audio_data))
        phase_drift = np.cumsum(drift_rate)
        drift_modulation = np.exp(1j * phase_drift)

        # Apply pitch drift in frequency domain
        fft_data = np.fft.fft(result["cleaned_data"])
        drifted_fft = fft_data * drift_modulation
        result["cleaned_data"] = np.real(np.fft.ifft(drifted_fft))

        # Add very subtle harmonic distortion (characteristic of analog systems)
        distortion_level = 0.0001
        second_harmonic = (
            distortion_level
            * np.sign(result["cleaned_data"])
            * (result["cleaned_data"] ** 2)
        )
        result["cleaned_data"] += second_harmonic

        # Only normalize if actual clipping occurred
        max_val = np.max(np.abs(result["cleaned_data"]))
        if max_val > 1.0:
            result["cleaned_data"] /= max_val

        return result

    def _apply_amplitude_normalization(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply amplitude normalization to match human audio characteristics"""
        normalized_data = audio_data.copy()

        # Dynamic range compression to human-like levels
        # RMS-based normalization
        rms = np.sqrt(np.mean(normalized_data**2))

        target_rms = self.target_rms
        if rms > 0:
            normalized_data = normalized_data * (target_rms / rms)

        # Soft clipping to prevent harsh digital artifacts
        clipping_threshold = 0.95
        clipped_indices = np.abs(normalized_data) > clipping_threshold

        if np.any(clipped_indices):
            # Apply soft clipping only to the samples that exceed threshold
            normalized_data[clipped_indices] = np.tanh(normalized_data[clipped_indices] * 2) / 2

        return normalized_data

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4)

    def _calculate_quality_metrics(
        self, original: np.ndarray, cleaned: np.ndarray, sample_rate: int = 44100
    ) -> Dict[str, Any]:
        """Calculate quality metrics comparing original and cleaned audio"""
        metrics = {}
        original_vector = self._quality_metric_vector(original)
        cleaned_vector = self._quality_metric_vector(cleaned)
        min_length = min(original_vector.size, cleaned_vector.size)
        if min_length == 0:
            return {
                "snr_db": float("inf"),
                "mfcc_distance": 0.0,
                "spectral_similarity": 1.0,
                "preservation_score": 1.0,
            }

        original_vector = original_vector[:min_length]
        cleaned_vector = cleaned_vector[:min_length]

        # Signal-to-Noise Ratio (SNR)
        noise = original_vector - cleaned_vector
        signal_power = np.mean(original_vector**2)
        noise_power = np.mean(noise**2)

        if noise_power > 0:
            metrics["snr_db"] = 10 * np.log10(signal_power / noise_power)
        else:
            metrics["snr_db"] = float("inf")

        # Perceptual similarity (simplified MFCC-based)
        orig_mfcc = librosa.feature.mfcc(
            y=original_vector, sr=sample_rate, n_mfcc=13
        )
        clean_mfcc = librosa.feature.mfcc(
            y=cleaned_vector, sr=sample_rate, n_mfcc=13
        )

        # Calculate distance between MFCCs (handle length mismatch)
        min_frames = min(orig_mfcc.shape[1], clean_mfcc.shape[1])
        mfcc_distance = np.mean(np.abs(orig_mfcc[:, :min_frames] - clean_mfcc[:, :min_frames]))
        metrics["mfcc_distance"] = float(mfcc_distance)

        # Spectral similarity
        orig_fft = np.abs(np.fft.fft(original_vector))
        clean_fft = np.abs(np.fft.fft(cleaned_vector))
        spectral_similarity = np.corrcoef(orig_fft, clean_fft)[0, 1]
        if not np.isfinite(spectral_similarity):
            spectral_similarity = 1.0 if np.allclose(orig_fft, clean_fft) else 0.0
        metrics["spectral_similarity"] = float(spectral_similarity)

        # Quality preservation score (0-1, higher is better)
        if metrics["snr_db"] > 40:
            preservation_score = 1.0
        elif metrics["snr_db"] > 20:
            preservation_score = 0.8
        elif metrics["snr_db"] > 10:
            preservation_score = 0.6
        else:
            preservation_score = 0.4

        # Adjust for spectral similarity
        preservation_score *= (1 + spectral_similarity) / 2

        metrics["quality_preservation"] = float(preservation_score)

        return metrics

    def _quality_metric_vector(self, audio_data: np.ndarray) -> np.ndarray:
        """Return a 1-D vector for quality metrics without accidental broadcasting."""
        audio_array = np.asarray(audio_data)
        if audio_array.ndim == 0:
            return audio_array.reshape(1).astype(float)
        if audio_array.ndim == 1:
            return audio_array.astype(float)
        return audio_array[:, 0].astype(float)

    def remove_machine_perfection_patterns(self, audio_data: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """Remove patterns typical of machine-perfect audio generation"""
        cleaned_data = audio_data.copy()

        # Detect and disrupt perfect rhythmic patterns
        # Compute onset detection
        onset_frames = librosa.onset.onset_detect(y=cleaned_data, sr=sample_rate)
        if len(onset_frames) > 3:
            onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
            intervals = np.diff(onset_times)

            # Check for too-perfect timing
            interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-10)

            if interval_cv < 0.05:  # Suspiciously perfect timing
                # Add subtle timing variations
                for i, onset_frame in enumerate(onset_frames[1:-1]):
                    if 50 < onset_frame < len(cleaned_data) - 50:
                        # Small random timing shift
                        shift_samples = np.random.randint(-10, 11)
                        if shift_samples != 0:
                            # Apply local shift
                            window = 20
                            start = max(0, onset_frame - window)
                            end = min(len(cleaned_data), onset_frame + window)

                            if end - start > 0:
                                window_data = cleaned_data[start:end]
                                shifted_data = np.interp(
                                    np.linspace(0, 1, len(window_data)),
                                    np.linspace(0, 1, len(window_data))
                                    + shift_samples / len(window_data),
                                    window_data,
                                )
                                cleaned_data[start:end] = shifted_data

        return cleaned_data
