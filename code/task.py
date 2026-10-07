"""
Auditory go/no-go task generator.

Stimulus types
--------------
  'pure'     : pure-tone Go (low or high F0) vs pure-tone No-Go
  'harmonic' : harmonic complex tone Go vs HCT No-Go
  'wn'       : white-noise Go (low or high dB) vs white-noise No-Go

Frequency grid
--------------
The log-spaced frequency grid is built from the tonal peaks of ALL task types
listed in `task_list`.  White-noise stimulus vectors span the entire grid at
uniform power (scaled to match total tonal power at the same dB level).

Trial structure (react style)
------------------------------
  pre-stim  :  input = 0,              output = 0
  stim      :  input = stimulus vector, output = 1 (Go) or 0 (No-Go)
  post-stim :  input = 0,              output = 1 (Go) or 0 (No-Go)

x shape : (tdim, batch_size, n_freq)
y shape : (tdim, batch_size, 1)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from utils.tone_utils import (
    db_to_amp,
    extract_tonal_peaks,
    build_log_freq_grid,
    make_stimulus_vectors,
)

DEFAULT_PARAMS = os.path.join(os.path.dirname(__file__), 'task_params.json')


# ── Parameter loading ──────────────────────────────────────────────────────

def load_params(path=None):
    with open(path or DEFAULT_PARAMS) as f:
        return json.load(f)


# ── Stimulus catalogue ─────────────────────────────────────────────────────

def _task_stimuli(task_type, params):
    """
    Return {'go': [...], 'nogo': [...]} lists of
    (label, f0, max_freq, mask_f0, amplitude) tuples for one task type.
    """
    s        = params['stimulus']
    amp      = db_to_amp(s['amplitude_db'])
    wn_hi    = db_to_amp(s['wn_go_db'])
    wn_nogo  = db_to_amp(s['wn_nogo_db'])
    max_freq = s['hct_max_freq']

    catalogue = {
        'pure': {
            'go': [
                ('PT Go low',  s['f0_low'],  None,     False, amp),
                ('PT Go high', s['f0_high'], None,     False, amp),
            ],
            'nogo': [
                ('PT No-Go',   s['f0_nogo'], None,     False, amp),
            ],
        },
        'harmonic': {
            'go': [
                ('HCT Go low',  s['f0_low'],  max_freq, False, amp),
                ('HCT Go high', s['f0_high'], max_freq, False, amp),
            ],
            'nogo': [
                ('HCT No-Go',   s['f0_nogo'], max_freq, False, amp),
            ],
        },
        'wn': {
            'go': [
                ('WN Go', None, None, False, wn_hi),
            ],
            'nogo': [
                ('WN No-Go', None, None, False, wn_nogo),
            ],
        },
    }
    if task_type not in catalogue:
        raise ValueError(f"Unknown task_type '{task_type}'. "
                         f"Choose from: {list(catalogue)}")
    return catalogue[task_type]


def _all_specs(task_list, params):
    """Deduplicated list of all stimulus specs across task_list."""
    seen  = set()
    specs = []
    for tt in task_list:
        for role in ('go', 'nogo'):
            for spec in _task_stimuli(tt, params)[role]:
                if spec[0] not in seen:
                    seen.add(spec[0])
                    specs.append(spec)
    return specs


# ── Grid and vector builders ───────────────────────────────────────────────

def build_freq_grid(task_list, params=None):
    """
    Build the shared log-spaced frequency grid from all tonal stimuli
    in task_list.

    Returns
    -------
    grid_hz   : 1-D array of grid centre frequencies (Hz)
    ref_power : mean total power of tonal stimuli (used to scale WN)
    """
    if params is None:
        params = load_params()
    s       = params['stimulus']
    specs   = _all_specs(task_list, params)
    peaks   = extract_tonal_peaks(specs, s['sr'], s['fft_pulse_dur'], s['ramp'])
    grid_hz, ref_power = build_log_freq_grid(peaks, s['freq_thresh_pct'])
    return grid_hz, ref_power


def _normalize_vectors(vectors, norm_mode, task_list, params):
    """
    Normalize stimulus vectors according to norm_mode.

    norm_mode : 'stimulus' — each vector divided by its own max
                'pair'     — vectors within each task type's go+nogo set
                             share the same scale (max across that set)
                'all'      — all vectors divided by the global max
                None/'none'— no normalization
    """
    if not norm_mode or norm_mode == 'none':
        return vectors

    vecs = {k: v.copy() for k, v in vectors.items()}

    if norm_mode == 'stimulus':
        for k, v in vecs.items():
            mx = v.max()
            if mx > 0:
                vecs[k] = v / mx

    elif norm_mode == 'pair':
        for task_type in task_list:
            specs  = _task_stimuli(task_type, params)
            labels = [s[0] for s in specs['go'] + specs['nogo']
                      if s[0] in vecs]
            if not labels:
                continue
            mx = max(vecs[l].max() for l in labels)
            if mx > 0:
                for l in labels:
                    vecs[l] = vecs[l] / mx

    elif norm_mode == 'all':
        mx = max(v.max() for v in vecs.values())
        if mx > 0:
            vecs = {k: v / mx for k, v in vecs.items()}

    else:
        raise ValueError(
            f"Unknown norm mode '{norm_mode}'. "
            "Choose from: 'stimulus', 'pair', 'all', 'none'.")

    return vecs


def get_stimulus_vectors(task_list, params=None):
    """
    Build log1p-power feature vectors for all stimuli across task_list,
    then normalize according to params['stimulus']['norm'].

    norm options
    ------------
    'stimulus' : each vector scaled to its own max  (max = 1 per stimulus)
    'pair'     : go + nogo of each task type share one scale
    'all'      : single global max across all stimuli  (default)
    'none'     : no normalization

    Returns
    -------
    grid_hz : 1-D array of grid frequencies (Hz)
    vectors : dict  label → np.ndarray of shape (n_freq,)
    """
    if params is None:
        params = load_params()
    s        = params['stimulus']
    amp_ref  = db_to_amp(s['amplitude_db'])
    specs    = _all_specs(task_list, params)
    peaks    = extract_tonal_peaks(specs, s['sr'], s['fft_pulse_dur'], s['ramp'])
    grid_hz, ref_power = build_log_freq_grid(peaks, s['freq_thresh_pct'])
    _, vectors = make_stimulus_vectors(
        specs, peaks, grid_hz, ref_power, amp_ref, apply_log1p=True)
    norm_mode = s.get('norm', 'all')
    vectors   = _normalize_vectors(vectors, norm_mode, task_list, params)
    return grid_hz, vectors


# ── Context encoding ──────────────────────────────────────────────────────────
# One-hot context vector per task type, constant across the whole trial.
# Channel layout in trial.x:  [freq (n_freq) | fixation (1) | context (3)]
CONTEXT_MAP = {
    'pure':     np.array([1, 0, 0], dtype='float32'),
    'wn':       np.array([0, 1, 0], dtype='float32'),
    'harmonic': np.array([0, 0, 1], dtype='float32'),
}
N_CONTEXT = 3   # number of context channels


# ── Trial class ────────────────────────────────────────────────────────────

class AuditoryGoNoGoTrial:
    """Batch of auditory go/no-go trials.

    Attributes
    ----------
    x        : (tdim, batch_size, n_freq + 1 + N_CONTEXT)
               freq channels | fixation (1) | context one-hot (3)
    y        : (tdim, batch_size, 1)   target response (0 or 1)
    stim_on  : int  time step of stimulus onset
    stim_off : int  time step of stimulus offset
    epochs   : dict with keys 'pre_stim', 'stim', 'post_stim'
    grid_hz  : 1-D array of grid frequencies (Hz)
    labels   : list of str, per-trial stimulus label
    is_go    : bool array of shape (batch_size,)
    n_freq   : int  number of frequency channels
    n_in     : int  total input channels (n_freq + 1 + N_CONTEXT)
    """

    def __init__(self, tdim, batch_size, n_freq):
        self.n_freq = n_freq
        self.n_in   = n_freq + 1 + N_CONTEXT
        self.x = np.zeros((tdim, batch_size, n_freq + 1 + N_CONTEXT), dtype='float32')
        self.y = np.zeros((tdim, batch_size, 1),                       dtype='float32')


# ── Trial generator ────────────────────────────────────────────────────────

def generate_trials(task_type, task_list, mode='random', batch_size=32,
                    params_path=None, rng=None):
    """
    Generate a batch of auditory go/no-go trials.

    Parameters
    ----------
    task_type  : 'pure' | 'harmonic' | 'wn'
        Which stimulus pair is used in this trial batch.
    task_list  : list of str
        All task types the network will ever learn; determines the freq grid.
    mode       : 'random' — 50/50 Go/No-Go, random stimulus drawn within role
                 'test'   — cycles through all Go and No-Go stimuli in order
    batch_size : int
    params_path: path to task_params.json (None → default next to task.py)
    rng        : np.random.Generator (None → new default_rng)

    Returns
    -------
    trial : AuditoryGoNoGoTrial
    """
    if rng is None:
        rng = np.random.default_rng()

    params = load_params(params_path)
    t      = params['timing']
    dt     = t['dt']

    stim_on           = int(t['pre_stim_dur_ms'] / dt)
    n_pulses          = t['n_pulses']
    pulse_steps       = max(1, int(np.ceil(t['pulse_dur_ms'] / dt)))
    stim_off          = stim_on + n_pulses * pulse_steps
    tdim              = stim_off + int(t['post_stim_dur_ms'] / dt)
    resp_onset_pulses = t.get('resp_onset_pulses', 2)

    grid_hz, vectors = get_stimulus_vectors(task_list, params)
    n_freq = len(grid_hz)

    task_specs = _task_stimuli(task_type, params)
    go_labels   = [s[0] for s in task_specs['go']]
    nogo_labels = [s[0] for s in task_specs['nogo']]

    trial        = AuditoryGoNoGoTrial(tdim, batch_size, n_freq)
    labels       = []
    is_go        = np.zeros(batch_size, dtype=bool)

    # fixation channel (index n_freq): 1 before stim onset, 0 after
    trial.x[:stim_on, :, n_freq] = 1.0

    # context one-hot (indices n_freq+1 : n_freq+1+N_CONTEXT): constant for whole trial
    ctx = CONTEXT_MAP.get(task_type, np.zeros(N_CONTEXT, dtype='float32'))
    trial.x[:, :, n_freq + 1 : n_freq + 1 + N_CONTEXT] = ctx

    def _fill_go(i):
        """Fill trial i with n_pulses alternating Go spectra, stim period only."""
        for step in range(stim_on, stim_off):
            pulse_idx = (step - stim_on) // pulse_steps
            lbl = go_labels[pulse_idx % len(go_labels)]
            trial.x[step, i, :n_freq] = vectors[lbl]
        trial.y[stim_on + resp_onset_pulses * pulse_steps:, i, 0] = 1.0

    def _fill_nogo(i, label):
        trial.x[stim_on:stim_off, i, :n_freq] = vectors[label]

    if mode == 'random':
        for i in range(batch_size):
            go = bool(rng.integers(0, 2))
            if go:
                _fill_go(i)
                labels.append('Go (' + '/'.join(go_labels) + ')')
            else:
                label = nogo_labels[rng.integers(len(nogo_labels))]
                _fill_nogo(i, label)
                labels.append(label)
            is_go[i] = go

    elif mode == 'test':
        # 1 go trial type (alternates internally) + len(nogo_labels) nogo types
        n_nogo = len(nogo_labels)
        cycle  = 1 + n_nogo
        for i in range(batch_size):
            pos = i % cycle
            go  = pos == 0
            if go:
                _fill_go(i)
                labels.append('Go (' + '/'.join(go_labels) + ')')
            else:
                label = nogo_labels[(pos - 1) % n_nogo]
                _fill_nogo(i, label)
                labels.append(label)
            is_go[i] = go

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'random' or 'test'.")

    trial.stim_on  = stim_on
    trial.stim_off = stim_off
    trial.tdim     = tdim
    trial.epochs   = {
        'pre_stim':  (0,        stim_on),
        'stim':      (stim_on,  stim_off),
        'post_stim': (stim_off, tdim),
    }
    trial.grid_hz = grid_hz
    trial.labels  = labels
    trial.is_go   = is_go
    return trial


# ── Quick sanity check ─────────────────────────────────────────────────────

if __name__ == '__main__':
    task_list = ['pure', 'harmonic', 'wn']
    for task_type in task_list:
        trial = generate_trials(task_type, task_list, mode='test', batch_size=6)
        print(f'{task_type:10s}  n_freq={len(trial.grid_hz):3d}  '
              f'tdim={trial.tdim}  '
              f'stim_on={trial.stim_on}  '
              f'go={trial.is_go.tolist()}')
        print(f'  labels: {trial.labels}')
        print(f'  x[stim_on, 0, :5] = {trial.x[trial.stim_on, 0, :5]}')
        print(f'  y[stim_on, :, 0]  = {trial.y[trial.stim_on, :, 0]}')
