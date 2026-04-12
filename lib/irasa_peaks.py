"""
IRASA Peak Extraction
=====================

Extracts discrete spectral peaks from EEG data using IRASA (Irregular
Resampling Auto-Spectral Analysis) for aperiodic removal, followed by
Gaussian fitting on the oscillatory residual.

Output format matches FOOOF/specparam: each peak is [center_freq, power,
bandwidth] where bandwidth = 2*sigma (same convention as specparam).

This enables method-independent replication of phi-lattice enrichment
analysis (Prediction P22).

hset selection follows Gerster et al. (2022) recommendations:
- h_max as small as possible to minimize evaluated frequency range
- But large enough to separate peaks of the expected logarithmic width
- Band-adaptive: lower h_max for high-frequency bands near the noise floor
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from shape_vs_resonance import irasa_psd


# Default hset per Gerster et al. (2022) recommendation:
# Keep h_max small to limit evaluated range, but large enough for peak removal.
# For scalp EEG resting-state ("easy" spectra with narrow peaks), h_max=1.9 is
# generally sufficient. For bands near the noise floor, reduce h_max.
HSET_DEFAULT = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)
HSET_REDUCED = (1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5)


def compute_safe_hset(fit_lo, fit_hi, fs, highpass_freq=1.0, noise_ceil=100.0):
    """Compute a band-adaptive hset following Gerster et al. (2022).

    The evaluated frequency range of IRASA extends beyond the fitting range:
        f_eval_min = fit_lo / h_max
        f_eval_max = fit_hi * h_max

    We constrain h_max so that:
        - f_eval_min stays above highpass_freq (avoid filter stopband)
        - f_eval_max stays below noise_ceil (avoid spectral plateau)
        - f_eval_max stays below Nyquist/2 (conservative headroom)

    Returns the largest safe hset from {HSET_DEFAULT, HSET_REDUCED}.
    """
    nyquist = fs / 2.0

    # Maximum safe h_max from each constraint
    h_max_lo = fit_lo / max(highpass_freq, 0.5)  # avoid highpass edge
    h_max_hi = min(noise_ceil, nyquist * 0.9) / fit_hi  # avoid noise floor / Nyquist

    safe_h_max = min(h_max_lo, h_max_hi)

    # Choose the largest hset that fits within the safe limit
    if safe_h_max >= 1.9:
        return HSET_DEFAULT
    elif safe_h_max >= 1.5:
        return HSET_REDUCED
    else:
        # Very constrained: build a minimal hset
        h_vals = []
        h = 1.1
        while h <= safe_h_max and h < 2.0:
            h_vals.append(round(h, 2))
            h += 0.1
        if len(h_vals) < 3:
            h_vals = [1.1, 1.15, 1.2]  # absolute minimum
        return tuple(h_vals)


def _gaussian(f, amplitude, center, sigma):
    """Single Gaussian peak model."""
    return amplitude * np.exp(-0.5 * ((f - center) / sigma) ** 2)


def _fit_gaussian(freqs, p_osc, peak_idx, freq_res, max_half_width_hz):
    """Fit a Gaussian to a single peak in the oscillatory spectrum.

    Returns (center_freq, amplitude, bandwidth) or None if fit fails.
    Bandwidth = 2*sigma, matching specparam convention.
    """
    center_guess = freqs[peak_idx]
    amp_guess = p_osc[peak_idx]

    # Local window: +/- max_half_width around peak
    window_mask = ((freqs >= center_guess - max_half_width_hz) &
                   (freqs <= center_guess + max_half_width_hz))
    if window_mask.sum() < 3:
        return None

    f_win = freqs[window_mask]
    p_win = p_osc[window_mask]

    try:
        popt, _ = curve_fit(
            _gaussian, f_win, p_win,
            p0=[amp_guess, center_guess, freq_res * 2],
            bounds=(
                [0, f_win[0], freq_res * 0.5],           # lower bounds
                [amp_guess * 10, f_win[-1], max_half_width_hz]  # upper bounds
            ),
            maxfev=2000,
        )
        amplitude, center, sigma = popt
        bandwidth = 2 * sigma
        return (center, amplitude, bandwidth)
    except (RuntimeError, ValueError):
        return None


def irasa_extract_peaks(data, fs, fit_lo, fit_hi, nperseg, noverlap,
                        hset=None, max_n_peaks=12, min_peak_height=0.0001,
                        peak_prominence=0.001, freq_res=None,
                        max_peak_width_hz=12.0):
    """Extract spectral peaks using IRASA aperiodic removal + Gaussian fitting.

    Parameters
    ----------
    data : 1-D array
        Single-channel EEG time series.
    fs : float
        Sampling frequency in Hz.
    fit_lo, fit_hi : float
        Frequency range for IRASA decomposition.
    nperseg : int
        Welch window length (samples).
    noverlap : int
        Welch overlap (samples). Not used by irasa_psd (uses nperseg//2).
    hset : tuple of float or None
        Resampling factors for IRASA. If None, automatically computed
        per Gerster et al. (2022) to avoid highpass edge and noise floor.
    max_n_peaks : int
        Maximum number of peaks to return.
    min_peak_height : float
        Minimum oscillatory power for a peak.
    peak_prominence : float
        Minimum prominence for scipy.signal.find_peaks.
    freq_res : float or None
        Frequency resolution in Hz. If None, computed as fs/nperseg.
    max_peak_width_hz : float
        Maximum Gaussian half-width for fitting.

    Returns
    -------
    peaks : ndarray, shape (N, 3)
        Each row: [center_freq, power, bandwidth]. Empty (0,3) if no peaks.
    quality : float
        Fractal consistency (1 - mean CV across h-values). Higher = better.
    osc_snr : float
        Fraction of power that is oscillatory.
    """
    if freq_res is None:
        freq_res = fs / nperseg

    # Band-adaptive hset per Gerster et al. (2022)
    if hset is None:
        hset = compute_safe_hset(fit_lo, fit_hi, fs)

    # Give IRASA the full evaluated range so all h-values contribute at every
    # frequency in [fit_lo, fit_hi]. Without this, the downsampled PSDs (1/h)
    # can't reach fit_hi and the fractal estimate degrades at the band edges.
    h_max = max(hset)
    irasa_fmax = min(fit_hi * h_max, fs / 2.0 * 0.95)

    # Run IRASA with per-h fractal estimates
    f, P_total, P_frac, P_osc, P_fracs_stack = irasa_psd(
        data, fs, hset=hset, nperseg=nperseg, fmax=irasa_fmax, return_per_h=True)

    # Crop to fit range (IRASA decomposed over wider range, we only need this slice)
    mask = (f >= fit_lo) & (f <= fit_hi)
    f = f[mask]
    P_total = P_total[mask]
    P_frac = P_frac[mask]
    P_osc = P_osc[mask]
    P_fracs_stack = P_fracs_stack[:, mask]

    if len(f) < 5:
        return np.empty((0, 3)), 0.0, 0.0

    # Quality metric: fractal consistency across h-values
    frac_mean = np.mean(P_fracs_stack, axis=0)
    frac_std = np.std(P_fracs_stack, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(frac_mean > 1e-20, frac_std / frac_mean, 0.0)
    quality = float(np.clip(1.0 - np.mean(cv), 0.0, 1.0))

    # Oscillatory SNR
    total_power = np.sum(P_total)
    if total_power > 0:
        osc_snr = float(1.0 - np.sum(P_frac) / total_power)
        osc_snr = max(0.0, osc_snr)
    else:
        osc_snr = 0.0

    # Find peaks in oscillatory spectrum.
    # IRASA P_osc is in linear power (V²/Hz or µV²/Hz) -- absolute thresholds
    # are meaningless because scale depends on EEG units. Convert to relative
    # thresholds based on the P_osc distribution within this band.
    p_osc_median = np.median(P_osc[P_osc > 0]) if np.any(P_osc > 0) else 0.0
    rel_height = max(p_osc_median * min_peak_height * 1e4, 0.0)  # scale-invariant
    rel_prominence = max(p_osc_median * peak_prominence * 1e3, 0.0)
    min_distance = max(1, int(np.ceil(2 * freq_res / (f[1] - f[0]))))
    peak_indices, properties = find_peaks(
        P_osc,
        height=rel_height,
        prominence=rel_prominence,
        distance=min_distance,
    )

    if len(peak_indices) == 0:
        return np.empty((0, 3)), quality, osc_snr

    # Sort by power (descending) and cap
    order = np.argsort(P_osc[peak_indices])[::-1]
    peak_indices = peak_indices[order[:max_n_peaks]]

    # Fit Gaussians to each peak
    max_half_width = max_peak_width_hz / 2.0
    fitted_peaks = []
    for idx in peak_indices:
        result = _fit_gaussian(f, P_osc, idx, freq_res, max_half_width)
        if result is not None:
            fitted_peaks.append(result)

    if not fitted_peaks:
        return np.empty((0, 3)), quality, osc_snr

    peaks = np.array(fitted_peaks)
    # Re-sort by power descending
    peaks = peaks[np.argsort(peaks[:, 1])[::-1]]

    return peaks, quality, osc_snr
