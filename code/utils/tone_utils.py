"""
Stimulus generation and display utilities for the tone discrimination task.

Functions are parameter-agnostic: all defaults come from the notebook's
params cell, which is expected to run before these functions are called.
Import with:
    from utils import *
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from IPython.display import Audio, display


# ── Waveform helpers ──────────────────────────────────────────────────────

def cosine_ramp(n_samples, ramp_samples):
    """Cosine-squared (raised-cosine) onset/offset envelope."""
    env = np.ones(n_samples)
    ramp = (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples)) / 2
    env[:ramp_samples]  = ramp
    env[-ramp_samples:] = ramp[::-1]
    return env


def db_to_amp(spl, ref=94):
    """Convert dB SPL to linear amplitude (ref=94 dB SPL → amplitude 1.0)."""
    return 10 ** ((spl - ref) / 20)


# ── Stimulus generators ───────────────────────────────────────────────────

def make_pure_tone(f0, duration, sr, ramp, amplitude):
    """Single pure-tone burst, RMS-normalised after ramp so amplitude = RMS level."""
    n    = int(duration * sr)
    t    = np.arange(n) / sr
    wave = np.sin(2 * np.pi * f0 * t)
    wave *= cosine_ramp(n, int(ramp * sr))
    wave /= np.sqrt(np.mean(wave ** 2))
    return wave * amplitude


def make_hct(f0, duration, sr, max_freq, ramp, amplitude, mask_f0=False):
    """
    Harmonic complex tone: equal-amplitude sinusoids at k*f0 for k*f0 <= max_freq.
    RMS-normalised after ramp so total loudness (sound-meter reading) = amplitude.
    mask_f0=True removes the fundamental (k=1).
    """
    n         = int(duration * sr)
    t         = np.arange(n) / sr
    harmonics = [k for k in range(1, int(max_freq / f0) + 1) if k * f0 <= max_freq]
    if mask_f0:
        harmonics = [k for k in harmonics if k != 1]
    wave = sum(np.sin(2 * np.pi * k * f0 * t) for k in harmonics)
    wave *= cosine_ramp(n, int(ramp * sr))
    wave /= np.sqrt(np.mean(wave ** 2))
    return wave * amplitude


def make_white_noise(duration, sr, ramp, amplitude, seed=None):
    """White noise burst, RMS-normalised after ramp so amplitude = RMS level."""
    rng  = np.random.default_rng(seed)
    n    = int(duration * sr)
    wave = rng.standard_normal(n)
    wave *= cosine_ramp(n, int(ramp * sr))
    wave /= np.sqrt(np.mean(wave ** 2))
    return wave * amplitude


def make_pulse_train(pulse_makers, pulse_dur, pulse_onset, n_pulses, sr):
    """
    Build a pulse train by cycling through pulse_makers.
    Alternating: pass [maker_A, maker_B]
    Repeating:   pass [maker_A]
    """
    total   = int((n_pulses - 1) * pulse_onset * sr) + int(pulse_dur * sr)
    train   = np.zeros(total)
    n_pulse = int(pulse_dur * sr)
    n_step  = int(pulse_onset * sr)
    for idx in range(n_pulses):
        onset = idx * n_step
        pulse = pulse_makers[idx % len(pulse_makers)]()
        train[onset:onset + n_pulse] += pulse
    return train


# ── Feature extraction ───────────────────────────────────────────────────

def tone_fft(f0, duration, sr, ramp, amplitude, max_freq=None, mask_f0=False, seed=None, round_to=None, theoretical=False):
    """
    Generate the stimulus waveform and return its FFT power spectrum.

      f0 given, max_freq=None  → pure tone
      f0 given, max_freq given → HCT
      f0=None                  → white noise (seed for reproducibility)
      round_to                 → round freq values to this resolution (e.g. 0.01 → nearest 0.01 Hz)
      theoretical              → for white noise (f0=None), return flat theoretical spectrum
                                 (power = amplitude² at every bin) instead of a random realisation

    Returns {'freq': np.ndarray (Hz), 'power': np.ndarray}
    """
    n    = int(duration * sr)
    freq = np.fft.rfftfreq(n, 1.0 / sr)

    if f0 is None and theoretical:
        # Theoretical white noise PSD: flat at amplitude² per bin
        power = np.full(len(freq), amplitude ** 2)
    else:
        if f0 is None:
            wave = make_white_noise(duration, sr, ramp, amplitude, seed=seed)
        elif max_freq is None:
            wave = make_pure_tone(f0, duration, sr, ramp, amplitude)
        else:
            wave = make_hct(f0, duration, sr, max_freq, ramp, amplitude, mask_f0=mask_f0)
        power = np.abs(np.fft.rfft(wave)) ** 2 / n

    if round_to is not None:
        freq = np.round(freq / round_to) * round_to
    return {'freq': freq, 'power': power}


def tone_spectrum(f0, duration, sr, ramp, amplitude, max_freq=None, mask_f0=False):
    """
    Return the spectral content of a tone as {frequency_hz: amplitude}.

    Mirrors make_pure_tone / make_hct inputs:
      max_freq=None  → pure tone  → {f0: amplitude}
      max_freq given → HCT        → {k*f0: amp_per_harmonic, ...}

    For HCT the per-harmonic amplitude matches what make_hct produces:
    each harmonic has equal weight 1.0 before peak normalisation, so
    amp_per_harmonic = amplitude / peak(sum_of_harmonics).
    """
    if max_freq is None:
        # Pure tone: single sinusoid at f0
        return {f0: amplitude}

    # HCT: build harmonic list (same logic as make_hct)
    harmonics = [k for k in range(1, int(max_freq / f0) + 1) if k * f0 <= max_freq]
    if mask_f0:
        harmonics = [k for k in harmonics if k != 1]

    # Compute peak of un-normalised sum to get per-harmonic amplitude
    n        = int(duration * sr)
    t        = np.arange(n) / sr
    sum_wave = sum(np.sin(2 * np.pi * k * f0 * t) for k in harmonics)
    peak     = np.max(np.abs(sum_wave))

    amp_per_harmonic = amplitude / peak
    return {k * f0: amp_per_harmonic for k in harmonics}


# ── Task utilities ────────────────────────────────────────────────────────

def extract_tonal_peaks(stimuli_specs, sr, pulse_dur, ramp):
    """
    Run FFT on each stimulus and extract peaks above mean+2SD.
    stimuli_specs: list of (label, f0, max_freq, mask_f0, amplitude)
    White noise (f0=None) entries get empty arrays.
    Returns dict: label → {'freqs': Hz array, 'power': array}
    """
    from scipy.signal import find_peaks as _find_peaks
    results = {}
    for label, f0, max_freq, mask_f0, amplitude in stimuli_specs:
        result = tone_fft(f0, pulse_dur, sr, ramp, amplitude,
                          max_freq=max_freq, mask_f0=mask_f0,
                          seed=42, theoretical=False, round_to=1)
        if f0 is None:
            results[label] = {'freqs': np.array([]), 'power': np.array([])}
        else:
            power = result['power']
            freq  = result['freq']
            thresh = power.mean() + 2 * power.std()
            peak_idxs, _ = _find_peaks(power, height=thresh)
            results[label] = {'freqs': freq[peak_idxs], 'power': power[peak_idxs]}
    return results


def build_log_freq_grid(peak_results, freq_thresh_pct=5):
    """
    Build a log-spaced frequency grid from tonal peaks and compute reference power.
    peak_results: output of extract_tonal_peaks
    Returns (grid_hz, ref_power) where grid_hz is in linear Hz.
    """
    tonal_freqs = np.concatenate(
        [v['freqs'] for v in peak_results.values() if len(v['freqs']) > 0])
    tonal_freqs = np.unique(tonal_freqs)
    freq_min    = tonal_freqs.min()
    freq_max    = tonal_freqs.max()
    ratio       = 1 + freq_thresh_pct / 100
    n_grid      = int(np.ceil(np.log(freq_max / freq_min) / np.log(ratio))) + 1
    grid_hz     = freq_min * ratio ** np.arange(n_grid)

    tonal_total = [v['power'].sum() for v in peak_results.values()
                   if len(v['power']) > 0]
    ref_power   = np.mean(tonal_total)
    return grid_hz, ref_power


def make_stimulus_vectors(stimuli_specs, peak_results, grid_hz, ref_power,
                           amplitude_ref, apply_log1p=True):
    """
    Align each stimulus's spectral peaks to the log-spaced grid.
    White noise gets uniform power scaled by (amp/amplitude_ref)².
    Returns (grid_hz, {label: power_vector}).
    """
    log_grid = np.log(grid_hz)
    n_grid   = len(grid_hz)
    vectors  = {}

    for label, f0, max_freq, mask_f0, amplitude in stimuli_specs:
        if f0 is None:
            total = ref_power * (amplitude / amplitude_ref) ** 2
            vec   = np.full(n_grid, total / n_grid)
        else:
            orig_freqs = peak_results[label]['freqs']
            orig_power = peak_results[label]['power']
            vec        = np.zeros(n_grid)
            snap_idx   = np.argmin(
                np.abs(log_grid[:, None] - np.log(orig_freqs)[None, :]), axis=0)
            for gi, pw in zip(snap_idx, orig_power):
                vec[gi] += pw

        if apply_log1p:
            vec = np.log1p(vec)
        vectors[label] = vec

    return grid_hz, vectors


# ── Display ───────────────────────────────────────────────────────────────

def show(title, wave, sr, ramp, f0=None):
    """
    4-panel plot: full waveform | fine-structure zoom | spectrogram (linear) | spectrogram (log freq).
    f0: if provided, zoom shows exactly 5 periods; otherwise shows 5 ms.
    """
    zoom_ms      = 5 / f0 * 1000 if f0 else 5.0
    t            = np.arange(len(wave)) / sr * 1e3
    zoom_samples = int(zoom_ms * sr / 1e3)
    start        = int(sr * ramp)           # skip onset ramp for zoom

    fig, axes = plt.subplots(1, 4, figsize=(20, 3),
                             gridspec_kw={'width_ratios': [2, 1.2, 2, 2]})

    # Panel 1: full waveform
    axes[0].plot(t, wave, lw=0.3, color='steelblue')
    axes[0].set(xlabel='Time (ms)', ylabel='Amplitude',
                title=title + ' -- full waveform', xlim=(0, t[-1]))

    # Panel 2: fine-structure zoom
    axes[1].plot(t[start:start + zoom_samples],
                 wave[start:start + zoom_samples], lw=0.8, color='steelblue')
    axes[1].axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    zoom_label = ('5 periods  ({:.2f} ms, f0={:.0f} Hz)'.format(zoom_ms, f0)
                  if f0 else '5 ms window (noise)')
    axes[1].set(xlabel='Time (ms)', ylabel='Amplitude', title=zoom_label,
                xlim=(t[start], t[start + zoom_samples]))

    # Shared spectrogram data
    f_ax, t_s, Sxx = scipy.signal.spectrogram(wave, fs=sr, nperseg=1024, noverlap=896)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    mask   = f_ax <= 55_000
    f_plot = f_ax[mask] / 1e3
    f_log  = np.maximum(f_plot, 1e-3)      # guard against zeros on log axis

    # Panel 3: spectrogram, linear frequency
    axes[2].pcolormesh(t_s * 1e3, f_plot, Sxx_db[mask],
                       shading='gouraud', cmap='inferno')
    axes[2].set(xlabel='Time (ms)', ylabel='Frequency (kHz)',
                title='Spectrogram (linear)')

    # Panel 4: spectrogram, log frequency
    axes[3].pcolormesh(t_s * 1e3, f_log, Sxx_db[mask],
                       shading='gouraud', cmap='inferno')
    axes[3].set_yscale('log')
    axes[3].set(xlabel='Time (ms)', ylabel='Frequency (kHz)',
                title='Spectrogram (log freq)')
    axes[3].yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: '{:.0f}'.format(x)))

    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()
    display(Audio(wave, rate=sr))
    print('Duration:', round(len(wave) / sr * 1000), 'ms | SR:', sr // 1000, 'kHz')
