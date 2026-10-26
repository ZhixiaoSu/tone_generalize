"""
Single-neuron Go / No-Go discrimination analysis for trained LeakyRNN checkpoints.

For every RNN unit and every task the checkpoint was trained on, we measure how
well that unit's activity separates Go from No-Go trials:

  * d'      : (mu_go - mu_nogo) / sqrt(0.5 * (var_go + var_nogo))
  * AUC     : ROC area (0.5 = chance), with a label-permutation p-value
  * per-epoch (stim / post-stim) trial-averaged activity, and per-timestep d'
  * selectivity class: Go-preferring, No-Go-preferring, or non-selective
  * for multi-task checkpoints, d' of each unit is compared across tasks

Hidden states are computed with a numpy re-implementation of LeakyRNN, with the
same input / recurrent noise used in training, so trial-to-trial variability is
non-zero.  Only the checkpoint unpickling needs jax.

Also provides `discrimination_vs_behavior(model_dir, ...)`: for every checkpoint in a
model directory, simulate behaviour (hit / false alarm / correct rate) and the
Euclidean distance between trial-averaged Go and No-Go population responses in a
time window, then plot distance vs. behaviour colour-coded by training step.

Usage:
    python neuron_discrimination.py <ckpt.pkl> [<ckpt2.pkl> ...] [options]
    python neuron_discrimination.py --model-dir <model_dir> [--window 0 300]

Outputs go to <ckpt dir>/checkpoint_<step>/neuron_discrimination/.
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).parent))
from task import generate_trials, load_params as load_task_params


# ── numpy LeakyRNN forward (mirrors network.LeakyRNN) ─────────────────────────

def run_rnn(params, xs, alpha, sigma_x, sigma_rec, rng, activation='relu', return_y=False):
    """xs (T, B, n_in) -> hs (T, B, n_rnn) [, ys (T, B, n_out)], with training-style noise."""
    W_in, W_rec = np.asarray(params['W_in']), np.asarray(params['W_rec'])
    b_rec = np.asarray(params['b_rec'])
    act = {'relu': lambda z: np.maximum(z, 0), 'tanh': np.tanh,
           'softplus': lambda z: np.logaddexp(0, z)}[activation]
    T, B, _ = xs.shape
    n = W_rec.shape[0]
    xs = xs + sigma_x * rng.standard_normal(xs.shape)
    eta = np.sqrt(2.0 / alpha) * sigma_rec * rng.standard_normal((T, B, n))
    h = np.zeros((B, n))
    hs = np.empty((T, B, n))
    for t in range(T):
        h = (1 - alpha) * h + alpha * act(xs[t] @ W_in + h @ W_rec + b_rec + eta[t])
        hs[t] = h
    if return_y:
        return hs, hs @ np.asarray(params['W_out']) + np.asarray(params['b_out'])
    return hs


# ── statistics ────────────────────────────────────────────────────────────────

def dprime(a, b):
    """d' per column. a: (n_go, N), b: (n_nogo, N). 0 where both variances are 0."""
    pooled = np.sqrt(0.5 * (a.var(0, ddof=1) + b.var(0, ddof=1)))
    diff = a.mean(0) - b.mean(0)
    return np.divide(diff, pooled, out=np.zeros_like(diff), where=pooled > 1e-12)


def behavioral_dprime(hit, fa, eps=1e-4):
    """
    Behavioural d' = Z(hit rate) - Z(false-alarm rate), Z = inverse Gaussian CDF
    (see dprime_test.ipynb).  Rates are clipped to [eps, 1 - eps] first, so a perfect
    performance (hit 1, FA 0) gives d' = 2 * Z(1 - eps) = 7.43 for eps = 1e-4, the same
    convention as the post-transfer evaluation in the transfer notebooks.
    Unlike the correct rate, d' does not saturate at 1.
    """
    h = np.clip(hit, eps, 1 - eps)
    f = np.clip(fa, eps, 1 - eps)
    return norm.ppf(h) - norm.ppf(f)


def auc(a, b):
    """ROC AUC per column (Mann-Whitney U), P(go > nogo) with ties = 0.5."""
    n1, n2 = len(a), len(b)
    ranks = rankdata(np.concatenate([a, b], 0), axis=0)
    u = ranks[:n1].sum(0) - n1 * (n1 + 1) / 2
    return u / (n1 * n2)


def auc_perm_p(a, b, n_perm, rng):
    """Two-sided permutation p-value for |AUC - 0.5| per column."""
    obs = np.abs(auc(a, b) - 0.5)
    pooled = np.concatenate([a, b], 0)
    n1 = len(a)
    ge = np.zeros(pooled.shape[1])
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        ge += np.abs(auc(pooled[idx[:n1]], pooled[idx[n1:]]) - 0.5) >= obs
    return (ge + 1) / (n_perm + 1)


# ── per-task analysis ─────────────────────────────────────────────────────────

def _geometry():
    """
    Go-cue-aligned frame for the timing generate_trials uses (task_params.json).

    Every trial has its own pre-stimulus time (trial.stim_on is a (batch,) array), so
    simulated arrays are cut to a common window around each trial's own stimulus
    onset, like train.evaluate_training: fixed_pre steps before the onset (the
    minimum guaranteed by pre_stim_range_ms) and fixed_post steps from the onset on.
    In the aligned frame the stimulus onset is index fixed_pre for every trial.
    """
    t = load_task_params()['timing']
    dt = t['dt']
    pulse_steps = max(1, int(np.ceil(t['pulse_dur_ms'] / dt)))
    pre_range = t.get('pre_stim_range_ms', [t['pre_stim_dur_ms']] * 2)
    fixed_pre = int(min(pre_range) / dt)
    stim_dur = t['n_pulses'] * pulse_steps
    fixed_post = stim_dur + int(t['post_stim_dur_ms'] / dt)
    return dict(dt=dt, fixed_pre=fixed_pre, fixed_post=fixed_post,
                aligned_tdim=fixed_pre + fixed_post,
                stim_on=fixed_pre, stim_off=fixed_pre + stim_dur)


def _align_to_cue(arr, stim_on, geo):
    """Cut (T, B, ...) to (aligned_tdim, B, ...) around each trial's own stimulus onset."""
    arr = np.asarray(arr)
    so = np.broadcast_to(np.asarray(stim_on), (arr.shape[1],))
    if so.min() < geo['fixed_pre'] or so.max() + geo['fixed_post'] > arr.shape[0]:
        raise ValueError(f'stim_on in [{so.min()}, {so.max()}] does not fit the aligned '
                         f'window (pre={geo["fixed_pre"]}, post={geo["fixed_post"]}, '
                         f'T={arr.shape[0]}); task_params.json differs from the one used '
                         f'to generate the trials?')
    idx = so[None, :] - geo['fixed_pre'] + np.arange(geo['aligned_tdim'])[:, None]
    return arr[idx, np.arange(arr.shape[1])[None, :]]


def analyse_task(params, task_type, task_list, cfg, args, rng):
    trial = generate_trials(task_type, task_list, mode='random',
                            batch_size=args['n_trials'], rng=rng)
    hs = run_rnn(params, trial.x, cfg['alpha'], cfg['sigma_x'], cfg['sigma_rec'],
                 rng, cfg['activation'])
    go, nogo = trial.is_go, ~trial.is_go
    geo = _geometry()
    hs = _align_to_cue(hs, trial.stim_on, geo)         # every trial on its own onset
    on, off, tdim = geo['stim_on'], geo['stim_off'], geo['aligned_tdim']
    # evaluate on the part of the stim epoch after the response-onset pulses begin,
    # and on the post-stim (response / hold) epoch
    epochs = {'stim': (on, off), 'post_stim': (off, tdim)}

    res = {'n_go': int(go.sum()), 'n_nogo': int(nogo.sum()), 'epochs': {}}
    for name, (t0, t1) in epochs.items():
        m = hs[t0:t1].mean(0)                       # (B, n_rnn)
        a, b = m[go], m[nogo]
        p = auc_perm_p(a, b, args['n_perm'], rng)
        res['epochs'][name] = dict(
            dprime=dprime(a, b), auc=auc(a, b), p=p,
            mean_go=a.mean(0), mean_nogo=b.mean(0))
    # time-resolved d'
    res['dprime_t'] = np.stack(
        [dprime(hs[t, go], hs[t, nogo]) for t in range(tdim)], 0)   # (T, n_rnn)
    res['stim_on'], res['stim_off'], res['tdim'] = on, off, tdim
    res['mean_h'] = hs.mean(1)                                       # (T, n_rnn)
    return res


def classify(res, alpha_p, d_min):
    """Selectivity from the stim-epoch stats: +1 Go-pref, -1 No-Go-pref, 0 none."""
    e = res['epochs']['stim']
    sel = (e['p'] < alpha_p) & (np.abs(e['dprime']) >= d_min)
    return np.where(sel, np.sign(e['dprime']), 0).astype(int)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_task(res, cls, task_type, step, out_dir, dt):
    d_t = res['dprime_t']
    T, n = d_t.shape
    t_ms = (np.arange(T) - res['stim_on']) * dt

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f'Neuron Go/No-Go discrimination — {task_type} | step {step}', fontsize=13)

    # (0,0) time-resolved d' heatmap, units sorted by signed time-averaged stim d'
    order = np.argsort(res['epochs']['stim']['dprime'])
    lim = max(1.0, np.percentile(np.abs(d_t), 99))
    im = ax[0, 0].imshow(d_t[:, order].T, aspect='auto', cmap='RdBu_r',
                         vmin=-lim, vmax=lim,
                         extent=[t_ms[0], t_ms[-1], 0, n])
    ax[0, 0].axvline(0, color='k', lw=0.8)
    ax[0, 0].axvline((res['stim_off'] - res['stim_on']) * dt, color='k', lw=0.8, ls='--')
    ax[0, 0].set(xlabel='time from stim onset (ms)', ylabel='unit (sorted)',
                 title="time-resolved d' (red = Go > No-Go)")
    fig.colorbar(im, ax=ax[0, 0], label="d'")

    # (0,1) histogram of d' per epoch
    for name, c in [('stim', 'C0'), ('post_stim', 'C1')]:
        ax[0, 1].hist(res['epochs'][name]['dprime'], bins=30, alpha=0.55, color=c, label=name)
    ax[0, 1].axvline(0, color='k', lw=0.8)
    ax[0, 1].set(xlabel="d'  (Go − No-Go)", ylabel='# units', title="d' distribution")
    ax[0, 1].legend()

    # (0,2) AUC sorted
    for name, c in [('stim', 'C0'), ('post_stim', 'C1')]:
        ax[0, 2].plot(np.sort(res['epochs'][name]['auc']), color=c, label=name)
    ax[0, 2].axhline(0.5, color='k', lw=0.8, ls=':')
    ax[0, 2].set(xlabel='unit rank', ylabel='ROC AUC', title='AUC (sorted)')
    ax[0, 2].legend()

    # (1,0) mean Go vs No-Go activity scatter, coloured by class
    e = res['epochs']['stim']
    colors = np.array(['0.6', 'tab:red', 'tab:blue'])[np.where(cls == 1, 1, np.where(cls == -1, 2, 0))]
    ax[1, 0].scatter(e['mean_nogo'], e['mean_go'], c=colors, s=18)
    mx = max(e['mean_go'].max(), e['mean_nogo'].max()) * 1.05
    ax[1, 0].plot([0, mx], [0, mx], 'k--', lw=0.8)
    ax[1, 0].set(xlabel='mean activity, No-Go', ylabel='mean activity, Go',
                 title='stim epoch  (red Go-pref, blue No-Go-pref)')

    # (1,1) mean time course per class
    mh = res['mean_h']
    for c, lab, col in [(1, 'Go-pref', 'tab:red'), (-1, 'No-Go-pref', 'tab:blue'), (0, 'non-selective', '0.5')]:
        if (cls == c).any():
            ax[1, 1].plot(t_ms, mh[:, cls == c].mean(1), color=col,
                          label=f'{lab} (n={(cls == c).sum()})')
    ax[1, 1].axvline(0, color='k', lw=0.8)
    ax[1, 1].set(xlabel='time from stim onset (ms)', ylabel='mean activity (all trials)',
                 title='population time course by class')
    ax[1, 1].legend()

    # (1,2) fraction selective over time
    sig = np.abs(d_t) >= 0.5
    ax[1, 2].plot(t_ms, (d_t >= 0.5).mean(1), color='tab:red', label="d' ≥ +0.5")
    ax[1, 2].plot(t_ms, (d_t <= -0.5).mean(1), color='tab:blue', label="d' ≤ −0.5")
    ax[1, 2].plot(t_ms, sig.mean(1), color='k', label='either')
    ax[1, 2].axvline(0, color='k', lw=0.8)
    ax[1, 2].set(xlabel='time from stim onset (ms)', ylabel='fraction of units',
                 title='fraction of discriminating units')
    ax[1, 2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / f'neuron_discrimination_{task_type}_{step}.png', dpi=140)
    plt.close(fig)


def plot_cross_task(results, step, out_dir):
    tasks = list(results)
    if len(tasks) < 2:
        return
    fig, axes = plt.subplots(1, len(tasks) - 1, figsize=(5.5 * (len(tasks) - 1), 5), squeeze=False)
    d0 = results[tasks[0]]['epochs']['stim']['dprime']
    for ax, t in zip(axes[0], tasks[1:]):
        d1 = results[t]['epochs']['stim']['dprime']
        ax.scatter(d0, d1, s=16)
        ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
        r = np.corrcoef(d0, d1)[0, 1]
        ax.set(xlabel=f"d' ({tasks[0]})", ylabel=f"d' ({t})", title=f'r = {r:.2f}')
    fig.suptitle(f"Per-unit stim-epoch d' across tasks | step {step}")
    fig.tight_layout()
    fig.savefig(out_dir / f'neuron_discrimination_cross_task_{step}.png', dpi=140)
    plt.close(fig)


# ── discrimination vs. behaviour across training checkpoints ──────────────────

def _list_checkpoints(model_dir):
    """[(step, path)] for checkpoint_<step>.pkl files, sorted by step."""
    out = []
    for p in Path(model_dir).glob('checkpoint_*.pkl'):
        m = re.fullmatch(r'checkpoint_(\d+)\.pkl', p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


DEFAULT_CRITERIA = Path(__file__).parent / 'responsive_criteria.json'


def load_criteria(path=None):
    with open(path or DEFAULT_CRITERIA, encoding='utf-8') as f:
        return json.load(f)


def responsive_units(hs, stim_on, dt, criteria=None):
    """
    Stimulus-responsive units: peak activity in the response window exceeds the
    pre-stimulus baseline mean by >= n_sd baseline SDs (criteria from
    responsive_criteria.json by default).

    Parameters
    ----------
    hs      : (T, B, n_rnn) hidden states, all trials of the task (Go and No-Go)
    stim_on : stimulus-onset time step
    dt      : ms per time step
    criteria: dict (see responsive_criteria.json) or path to a json file

    Returns
    -------
    dict with mask (n,) bool, idx (indices), peak, base_mean, base_sd (n,), criteria
    """
    if criteria is None or isinstance(criteria, (str, Path)):
        criteria = load_criteria(criteria)
    r0, r1 = (stim_on + int(round(w / dt)) for w in criteria['response_window_ms'])
    b0, b1 = (stim_on + int(round(w / dt)) for w in criteria['baseline_window_ms'])
    if not (0 <= b0 < b1 <= stim_on and r0 < r1 <= hs.shape[0]):
        raise ValueError('response/baseline windows fall outside the trial or overlap '
                         'the pre-stimulus period incorrectly')
    psth = hs.mean(1)                                          # (T, n)
    peak = psth[r0:r1].max(0)
    if criteria.get('sd_mode', 'psth') == 'psth':
        base = psth[b0:b1]                                     # (W, n)
    elif criteria['sd_mode'] == 'trial':
        base = hs[b0:b1].reshape(-1, hs.shape[2])              # (W*B, n)
    else:
        raise ValueError(f"Unknown sd_mode '{criteria['sd_mode']}'")
    base_mean, base_sd = base.mean(0), base.std(0, ddof=1)
    # strict '>' on the mean also rejects silent units (peak = mean = sd = 0)
    mask = (peak > base_mean) & (peak - base_mean >= criteria['n_sd'] * base_sd)
    return dict(mask=mask, idx=np.flatnonzero(mask), peak=peak,
                base_mean=base_mean, base_sd=base_sd, criteria=criteria)


def save_responsive_units(res, path):
    """Write the criteria and the responsive-unit indices of `responsive_units` to json."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict(criteria=res['criteria'], n_units=int(len(res['mask'])),
                       n_responsive=int(res['mask'].sum()),
                       responsive_units=res['idx'].tolist()), f, indent=2)


def select_units(hs, stim_on, dt, zscore='all', responsive_only=True, criteria=None):
    """
    Prepare responses for the Euclidean-distance analyses: keep responsive units
    only and Z-score each unit's activity.

    zscore : 'all'      -> per-unit mean / SD over all trials and time steps
             'baseline' -> per-unit mean / SD over the pre-stimulus baseline window
                           (criteria['baseline_window_ms'], all trials)
             None       -> raw activity
    Units are selected on the raw activity (Z-scoring is a positive affine map, so it
    would not change the selection).  Units with zero SD are set to 0 after Z-scoring.

    Returns (z (T, B, k), idx (k,)): selected hidden states and their unit indices.
    """
    if criteria is None or isinstance(criteria, (str, Path)):
        criteria = load_criteria(criteria)
    n = hs.shape[2]
    idx = (responsive_units(hs, stim_on, dt, criteria)['idx']
           if responsive_only else np.arange(n))
    if len(idx) == 0:
        raise ValueError('No responsive units under the given criteria')
    z = hs[:, :, idx]
    if zscore == 'all':
        ref = z.reshape(-1, len(idx))
    elif zscore == 'baseline':
        b0, b1 = (stim_on + int(round(w / dt)) for w in criteria['baseline_window_ms'])
        ref = z[b0:b1].reshape(-1, len(idx))
    elif zscore is None:
        return z, idx
    else:
        raise ValueError(f"Unknown zscore '{zscore}'; use 'all', 'baseline' or None")
    mu, sd = ref.mean(0), ref.std(0, ddof=1)
    z = np.divide(z - mu, sd, out=np.zeros_like(z), where=sd > 1e-12)
    return z, idx


def random_init_params(n_in, hp, seed=0):
    """Numpy equivalent of LeakyRNN's initialisers (see network.py)."""
    r = np.random.default_rng(seed)
    n = hp['n_rnn']
    q, rr = np.linalg.qr(r.standard_normal((n, n)))
    q = q * np.sign(np.diag(rr))                      # orthogonal, sign-fixed
    return dict(W_in=r.standard_normal((n_in, n)) / np.sqrt(n_in),
                W_rec=hp.get('w_rec_coeff', 0.8) * q,
                b_rec=np.zeros(n),
                W_out=r.standard_normal((n, 1)) / np.sqrt(n),
                b_out=np.zeros(1))


_STEP0_REPORTED = set()      # model dirs whose step-0 source was already printed


def _starting_point(model_dir, config, hp, seed):
    """Step-0 params: checkpoint_init.pkl if present, else a random network.
    The source used is printed once per model directory."""
    init = Path(model_dir) / 'checkpoint_init.pkl'
    first = str(model_dir) not in _STEP0_REPORTED
    _STEP0_REPORTED.add(str(model_dir))
    if init.exists():
        if first:
            print(f'  step 0: using checkpoint_init.pkl  [{model_dir}]')
        with open(init, 'rb') as f:
            return pickle.load(f)['params']
    if first:
        print(f'  step 0: NO checkpoint_init.pkl in {model_dir} -> '
              f'using a random network built from its config.json')
    return random_init_params(config['n_in'], hp, seed=config.get('seed', 0))


def population_distance(hs, is_go, t0, t1):
    """
    Euclidean distance between the class-mean vectors r_Go and r_NoGo, where each
    unit's activity is averaged over the window [t0, t1) and over Go / No-Go trials
    first, D = sqrt(sum_i (r_i,Go - r_i,NoGo)^2).  hs: (T, B, n) (Z-scored, selected
    units; see select_units).
    Also returns the mean |d'| across units (trial-wise, window-averaged activity).
    """
    m = hs[t0:t1].mean(0)                                      # (B, n) window means
    dist = float(np.linalg.norm(m[is_go].mean(0) - m[~is_go].mean(0)))
    mean_abs_d = float(np.abs(dprime(m[is_go], m[~is_go])).mean())
    return dist, mean_abs_d


def population_noise(hs, is_go, t0, t1):
    """
    Trial-to-trial noise of the population response in the window [t0, t1) and the
    Euclidean distance expressed in units of that noise.  hs: (T, B, n) Z-scored responses
    of the selected units (see select_units).

    Each trial's response is the window average X [B x n].  Noise = within-class scatter:
    residuals from the Go / No-Go class means, pooled over both classes,
        var_i = sum (X_i - class mean_i)^2 / (B - 2)          (per unit)
    Returns a dict:
      noise_var     sum_i var_i             total variance of the population (trial-to-trial)
      noise_sd      sqrt(noise_var)         population SD (same units as the distance);
                                            grows with sqrt(number of units)
      noise_sd_per_unit  noise_sd / sqrt(N) = sqrt(mean_i var_i)
                                            population SD normalised to the number of units
                                            (RMS single-unit SD), comparable across sessions
                                            with different numbers of units
      distance_per_unit  D / sqrt(N)        Euclidean distance normalised the same way
                                            (RMS single-unit distance)
      unit_sd_mean  mean_i sqrt(var_i)      average single-unit SD
      distance_norm D / noise_sd            Euclidean distance in units of the population SD
      dprime_axis   D / SD along the Go-No-Go axis
                                            the same distance, but with the noise measured
                                            only along the direction that separates the classes
    """
    X = hs[t0:t1].mean(0)                                      # (B, n) per-trial window means
    go, nogo = X[is_go], X[~is_go]
    d_vec = go.mean(0) - nogo.mean(0)
    D = float(np.linalg.norm(d_vec))
    res = np.concatenate([go - go.mean(0), nogo - nogo.mean(0)], axis=0)    # (B, n) residuals
    dof = max(len(X) - 2, 1)
    var_i = (res ** 2).sum(0) / dof
    noise_var = float(var_i.sum())
    noise_sd = float(np.sqrt(noise_var))
    axis_sd = float(np.sqrt(((res @ (d_vec / D)) ** 2).sum() / dof)) if D > 0 else np.nan
    n_units = X.shape[1]
    return dict(noise_var=noise_var, noise_sd=noise_sd,
                noise_sd_per_unit=noise_sd / np.sqrt(n_units),
                distance_per_unit=D / np.sqrt(n_units),
                unit_sd_mean=float(np.sqrt(var_i).mean()),
                distance_norm=D / noise_sd if noise_sd > 0 else np.nan,
                dprime_axis=D / axis_sd if axis_sd and axis_sd > 0 else np.nan)


def discrimination_vs_behavior(model_dir, window_ms=None, n_trials=512, seed=0,
                               save=True, show=False, zscore='all',
                               responsive_only=True, criteria=None):
    """
    Step 0 is checkpoint_init.pkl if it exists, otherwise a random network built from
    config.json (same initialisers as network.py).  Then, for each checkpoint_<step>.pkl
    in model_dir and each trained task:
      simulate trials with config.json settings (training-style noise), compute
      hit / false-alarm / correct rate (same definition as train.evaluate_training)
      and the Go-vs-No-Go population distance in the evaluation window.

    Parameters
    ----------
    model_dir  : directory with config.json and checkpoint_<step>.pkl files
    window_ms  : (start, end) in ms relative to stimulus onset.  Default: the whole
                 stimulus period, (0, stimulus duration).
                 e.g. (0, 300) to exclude lick-driven activity.
    n_trials   : trials per task per checkpoint (Go / No-Go ~50/50).  Every trial has
                 its own pre-stimulus time; trials are cut around their own stimulus
                 onset (see _geometry), so all windows are relative to the onset and the
                 responsive-unit baseline is limited to the guaranteed pre-stimulus time.
    zscore, responsive_only, criteria : see select_units.  The distance is computed
                 on Z-scored responses of responsive units, from the window-averaged
                 Go and No-Go class-mean vectors.

    Returns
    -------
    rows : list of dict (task, step, hit, fa, correct, distance, mean_abs_dprime,
           n_units)
    Saved to <model_dir>/neuron_discrimination_vs_behavior/ (JSON, PNG).
    """
    model_dir = Path(model_dir)
    config = _load_json(model_dir / 'config.json')
    hp = config['training']['model']
    tp = config['task_params']
    task_list = config.get('task_list', ['pure', 'harmonic', 'wn'])
    tasks = config.get('train_task', 'pure')
    tasks = [tasks] if isinstance(tasks, str) else tasks
    dt = tp['timing']['dt']
    geo = _geometry()
    stim_on, stim_off, tdim = geo['stim_on'], geo['stim_off'], geo['aligned_tdim']
    thresh = tp.get('response', {}).get('response_thresh', 0.5)
    reaction_steps = min(int(2000 / dt), geo['fixed_post'])

    if window_ms is None:
        window_ms = (0, (stim_off - stim_on) * dt)
    t0 = max(stim_on + int(round(window_ms[0] / dt)), 0)
    t1 = min(stim_on + int(round(window_ms[1] / dt)), tdim)
    if t1 <= t0:
        raise ValueError(f'Empty evaluation window {window_ms} ms')
    print(f'{model_dir.name}: tasks={tasks}  window={window_ms} ms -> steps [{t0}, {t1})')

    cfg = dict(alpha=dt / hp['tau'], sigma_x=hp['sigma_x'], sigma_rec=hp['sigma_rec'],
               activation=hp.get('activation', 'relu'))
    rng = np.random.default_rng(seed)
    rows = []
    ckpts = _list_checkpoints(model_dir)
    snapshots = []
    if not any(s == 0 for s, _ in ckpts):
        snapshots.append((0, _starting_point(model_dir, config, hp, seed)))
    for step, path in ckpts:
        with open(path, 'rb') as f:
            snapshots.append((step, pickle.load(f)['params']))
    for step, params in snapshots:
        for task in tasks:
            trial = generate_trials(task, task_list, mode='random',
                                    batch_size=n_trials, rng=rng)
            hs, ys = run_rnn(params, trial.x, cfg['alpha'], cfg['sigma_x'],
                             cfg['sigma_rec'], rng, cfg['activation'], return_y=True)
            hs = _align_to_cue(hs, trial.stim_on, geo)   # cut around each trial's own onset
            ys = _align_to_cue(ys, trial.stim_on, geo)
            go = trial.is_go
            pre = ys[:stim_on, :, 0].max(0) if stim_on > 0 else np.zeros(n_trials)
            post = ys[stim_on:stim_on + reaction_steps, :, 0].max(0)
            resp = (post > thresh) & (pre < 0.3)
            hit, fa = float(resp[go].mean()), float(resp[~go].mean())
            z, idx = select_units(hs, stim_on, dt, zscore, responsive_only, criteria)
            dist, mad = population_distance(z, go, t0, t1)
            nz = population_noise(z, go, t0, t1)
            rows.append(dict(task=task, step=step, hit=hit, fa=fa,
                             correct=0.5 * (hit + 1 - fa),
                             behav_dprime=float(behavioral_dprime(hit, fa)),
                             distance=dist,
                             mean_abs_dprime=mad, n_units=int(len(idx)), **nz))
            print(f'  step {step:6d} {task:9s} hit={hit:.2f} fa={fa:.2f} '
                  f"correct={rows[-1]['correct']:.2f} d'={rows[-1]['behav_dprime']:.2f} dist={dist:.3f} "
                  f'noise_sd={nz["noise_sd"]:.3f} dist/noise={nz["distance_norm"]:.2f} '
                  f'units={len(idx)}')
    if len(snapshots) < 2:
        raise FileNotFoundError(f'No checkpoint_<step>.pkl files in {model_dir}')

    out_dir = model_dir / 'neuron_discrimination_vs_behavior'
    if save:
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / f'perf_vs_distance_{window_ms[0]}_{window_ms[1]}ms.json', 'w') as f:
            json.dump(dict(model_dir=str(model_dir), window_ms=list(window_ms),
                           n_trials=n_trials,
                           definition=dict(zscore=zscore, responsive_only=responsive_only,
                                           aligned_to_stim_onset=True,
                                           noise_normalised=True, behav_dprime=True,
                                           per_unit_normalised=True,
                                           criteria=criteria if isinstance(criteria, dict)
                                           else load_criteria(criteria)),
                           rows=rows), f, indent=2)
    plot_distance_vs_behavior(rows, tasks, window_ms, model_dir.name,
                              out_dir if save else None, show)
    plot_distance_vs_behavior(rows, tasks, window_ms, model_dir.name,
                              out_dir if save else None, show, xkey='distance_norm',
                              xlabel='distance / population noise SD', tag='_norm')
    return rows


def _simulate_checkpoint(model_dir, step, task=None, n_trials=512, seed=0, return_y=False):
    """Load one checkpoint (step 0 falls back as in _starting_point) and simulate a
    noisy batch of `task` trials, cut around each trial's own stimulus onset.
    Returns dict(hs (T_aligned, B, n), trial, task, dt, stim_on, stim_off); stim_on /
    stim_off are indices in the aligned frame (see _geometry).  With return_y the dict
    also has ys (T_aligned, B, 1), the network output, and geo (the aligned geometry)."""
    model_dir = Path(model_dir)
    config = _load_json(model_dir / 'config.json')
    hp = config['training']['model']
    task_list = config.get('task_list', ['pure', 'harmonic', 'wn'])
    if task is None:
        task = config.get('train_task', 'pure')
        task = task if isinstance(task, str) else task[0]
    dt = config['task_params']['timing']['dt']

    path = model_dir / f'checkpoint_{step}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            params = pickle.load(f)['params']
    elif step == 0:
        params = _starting_point(model_dir, config, hp, seed)
    else:
        raise FileNotFoundError(path)

    trial = generate_trials(task, task_list, mode='random', batch_size=n_trials,
                            rng=np.random.default_rng(seed))
    out = run_rnn(params, trial.x, dt / hp['tau'], hp['sigma_x'], hp['sigma_rec'],
                  np.random.default_rng(seed + 1), hp.get('activation', 'relu'),
                  return_y=return_y)
    hs, ys = out if return_y else (out, None)
    geo = _geometry()
    res = dict(hs=_align_to_cue(hs, trial.stim_on, geo), trial=trial, task=task, dt=dt,
               stim_on=geo['stim_on'], stim_off=geo['stim_off'])
    if return_y:
        res['ys'] = _align_to_cue(ys, trial.stim_on, geo)
        res['geo'] = geo
    return res


def _draw_unit_panel(ax, u, name, t, psth, psth_go, psth_nogo, res,
                     show_xlabel=True, show_legend=False):
    """One unit's trial-averaged activity with baseline / response windows and threshold."""
    crit = res['criteria']
    ax.axvspan(*crit['baseline_window_ms'], color='tab:blue', alpha=0.12)
    ax.axvspan(*crit['response_window_ms'], color='tab:orange', alpha=0.15)
    ax.plot(t, psth_go[:, u], color='tab:red', lw=0.8, alpha=0.6, label='Go')
    ax.plot(t, psth_nogo[:, u], color='tab:green', lw=0.8, alpha=0.6, label='No-Go')
    ax.plot(t, psth[:, u], color='k', lw=1.5, label='all trials')
    thr = res['base_mean'][u] + crit['n_sd'] * res['base_sd'][u]
    ax.axhline(res['base_mean'][u], color='tab:blue', ls=':', lw=1)
    ax.axhline(thr, color='tab:blue', ls='--', lw=1, label=f'baseline + {crit["n_sd"]:g} SD')
    ax.set_title(f'{name}  unit {u}\npeak {res["peak"][u]:.2f} vs thr {thr:.2f}', fontsize=9)
    ax.set_xlim(-300, 400)
    if show_xlabel:
        ax.set_xlabel('time from stimulus onset (ms)')
    if show_legend:
        ax.set_ylabel('activity')
        ax.legend(fontsize=7, loc='upper left')


def plot_responsive_examples(model_dir, step, task=None, n_examples=5, n_trials=512,
                             seed=0, criteria=None, save_path=None, show=True):
    """
    Plot randomly chosen responsive (top row) and non-responsive (bottom row) units.

    Each panel shows the unit's trial-averaged activity (all trials; thin lines: Go and
    No-Go separately), the baseline window with its mean and the mean + n_sd * SD
    threshold, and the response window.  Titles give peak, threshold and the unit index.
    The criterion is the one in responsive_criteria.json (see responsive_units).

    Returns the responsive_units result dict.
    """
    sim = _simulate_checkpoint(model_dir, step, task, n_trials, seed)
    hs, trial, dt, stim_on = sim['hs'], sim['trial'], sim['dt'], sim['stim_on']
    res = responsive_units(hs, stim_on, dt, criteria)
    go = trial.is_go
    t = (np.arange(hs.shape[0]) - stim_on) * dt
    psth, psth_go, psth_nogo = hs.mean(1), hs[:, go].mean(1), hs[:, ~go].mean(1)

    rng = np.random.default_rng(seed)
    groups = [('responsive', res['idx']), ('non-responsive', np.flatnonzero(~res['mask']))]
    ncol = n_examples
    fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 6.4), sharex=True, squeeze=False)
    for row, (name, pool) in enumerate(groups):
        pick = rng.choice(pool, size=min(ncol, len(pool)), replace=False) if len(pool) else []
        for col in range(ncol):
            ax = axes[row, col]
            if col >= len(pick):
                ax.axis('off')
                continue
            _draw_unit_panel(ax, pick[col], name, t, psth, psth_go, psth_nogo, res,
                             show_xlabel=row == 1, show_legend=col == 0)
    fig.suptitle(f'{Path(model_dir).name} | step {step} | {sim["task"]} | '
                 f'{int(res["mask"].sum())}/{len(res["mask"])} units responsive  '
                 f'(blue = baseline window, orange = response window)', fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return res


def responsive_units_report(model_dir, step, task=None, n_examples=10, n_trials=512,
                            seed=0, criteria=None, pdf_path=None, ncols=5, rows_per_page=4):
    """
    Full responsive-unit report as one multi-page PDF.

      page 1      summary: model / criteria / counts, distribution of the response
                  z-score, peak vs threshold scatter, mean activity of both classes
      next pages  table of every unit (peak, baseline mean / SD, z, responsive?)
      last pages  n_examples random responsive and n_examples random non-responsive
                  units (trial-averaged activity, windows and threshold)

    n_examples : examples per class (capped at the number of units in that class)
    pdf_path   : default <model_dir>/responsive_units/responsive_units_<task>_<step>.pdf
                 (a .json with the criteria and responsive-unit indices is saved beside it)
    Returns (pdf_path, responsive_units result).
    """
    from matplotlib.backends.backend_pdf import PdfPages

    model_dir = Path(model_dir)
    sim = _simulate_checkpoint(model_dir, step, task, n_trials, seed)
    hs, trial, dt, stim_on = sim['hs'], sim['trial'], sim['dt'], sim['stim_on']
    task = sim['task']
    res = responsive_units(hs, stim_on, dt, criteria)
    crit, mask, n = res['criteria'], res['mask'], len(res['mask'])
    go = trial.is_go
    t = (np.arange(hs.shape[0]) - stim_on) * dt
    psth, psth_go, psth_nogo = hs.mean(1), hs[:, go].mean(1), hs[:, ~go].mean(1)
    thr = res['base_mean'] + crit['n_sd'] * res['base_sd']
    z = np.divide(res['peak'] - res['base_mean'], res['base_sd'],
                  out=np.full(n, np.inf), where=res['base_sd'] > 1e-12)

    if pdf_path is None:
        pdf_path = model_dir / 'responsive_units' / f'responsive_units_{task}_{step}.pdf'
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    resp_idx, non_idx = res['idx'], np.flatnonzero(~mask)
    title = f'{model_dir.name} | step {step} | {task}'

    with PdfPages(pdf_path) as pdf:
        # ── page 1: summary ───────────────────────────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.3))
        gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.4])
        ax = fig.add_subplot(gs[0, :]); ax.axis('off')
        lines = [
            f'Responsive-unit report', f'{title}', '',
            f'trials simulated: {n_trials} ({int(go.sum())} Go / {int((~go).sum())} No-Go), seed {seed}',
            f'response window {crit["response_window_ms"]} ms, baseline window '
            f'{crit["baseline_window_ms"]} ms, threshold = baseline mean + {crit["n_sd"]:g} SD '
            f'(sd_mode = {crit.get("sd_mode", "psth")})',
            '',
            f'responsive units: {int(mask.sum())} / {n} ({100 * mask.mean():.0f}%)',
            f'peak - baseline mean:  responsive median {np.median((res["peak"] - res["base_mean"])[mask]) if mask.any() else float("nan"):.3f}'
            f'   non-responsive median {np.median((res["peak"] - res["base_mean"])[~mask]) if (~mask).any() else float("nan"):.3f}',
            f'silent units (zero baseline SD): {int((res["base_sd"] <= 1e-12).sum())}',
        ]
        ax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=10)

        zc = np.clip(z, -10, 20)
        ax = fig.add_subplot(gs[1, 0])
        bins = np.linspace(-10, 20, 46)
        ax.hist(zc[mask], bins=bins, color='tab:orange', alpha=0.8, label='responsive')
        ax.hist(zc[~mask], bins=bins, color='tab:blue', alpha=0.6, label='non-responsive')
        ax.axvline(crit['n_sd'], color='k', ls='--', lw=1)
        ax.set(xlabel='(peak - baseline mean) / baseline SD  (clipped)', ylabel='# units',
               title='response z-score')
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[1, 1])
        ax.scatter(thr[~mask], res['peak'][~mask], s=14, color='tab:blue', label='non-responsive')
        ax.scatter(thr[mask], res['peak'][mask], s=14, color='tab:orange', label='responsive')
        lim = [0, max(thr.max(), res['peak'].max()) * 1.05]
        ax.plot(lim, lim, 'k--', lw=0.8)
        ax.set(xlabel='threshold (baseline mean + n SD)', ylabel='peak in response window',
               title='peak vs threshold')
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[1, 2])
        for sel, col, lab in [(mask, 'tab:orange', 'responsive'), (~mask, 'tab:blue', 'non-responsive')]:
            if sel.any():
                m = psth[:, sel].mean(1)
                s = psth[:, sel].std(1, ddof=1) / np.sqrt(sel.sum()) if sel.sum() > 1 else 0 * m
                ax.plot(t, m, color=col, label=f'{lab} (n={int(sel.sum())})')
                ax.fill_between(t, m - s, m + s, color=col, alpha=0.25)
        ax.axvspan(*crit['baseline_window_ms'], color='tab:blue', alpha=0.1)
        ax.axvspan(*crit['response_window_ms'], color='tab:orange', alpha=0.12)
        ax.set(xlabel='time from stimulus onset (ms)', ylabel='mean activity (± SEM across units)',
               title='population average', xlim=(-300, 400))
        ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── table pages: every unit ───────────────────────────────────────────
        header = f'{"unit":>5} {"resp":>5} {"peak":>9} {"base mean":>10} {"base SD":>9} {"z":>8} {"peak-thr":>9}'
        per_page = 44
        for p0 in range(0, n, per_page):
            fig = plt.figure(figsize=(8.3, 11.7))
            rows = [f'{title}   (units {p0}-{min(p0 + per_page, n) - 1})', '', header,
                    '-' * len(header)]
            for u in range(p0, min(p0 + per_page, n)):
                zs = f'{z[u]:8.2f}' if np.isfinite(z[u]) else f'{"inf":>8}'
                rows.append(f'{u:5d} {"yes" if mask[u] else "no":>5} {res["peak"][u]:9.3f} '
                            f'{res["base_mean"][u]:10.3f} {res["base_sd"][u]:9.4f} {zs} '
                            f'{res["peak"][u] - thr[u]:9.3f}')
            fig.text(0.05, 0.96, '\n'.join(rows), va='top', family='monospace', fontsize=9)
            pdf.savefig(fig); plt.close(fig)

        # ── example pages ─────────────────────────────────────────────────────
        for name, pool in [('responsive', resp_idx), ('non-responsive', non_idx)]:
            k = min(n_examples, len(pool))
            pick = rng.choice(pool, size=k, replace=False) if k else []
            per = ncols * rows_per_page
            for p0 in range(0, max(k, 1), per):
                fig, axes = plt.subplots(rows_per_page, ncols, figsize=(3.4 * ncols, 2.6 * rows_per_page + 0.8),
                                         squeeze=False)
                chunk = pick[p0:p0 + per]
                for i, ax in enumerate(axes.ravel()):
                    if i >= len(chunk):
                        ax.axis('off')
                        continue
                    _draw_unit_panel(ax, chunk[i], name, t, psth, psth_go, psth_nogo, res,
                                     show_xlabel=i // ncols == rows_per_page - 1 or p0 + i + ncols >= k,
                                     show_legend=i == 0)
                fig.suptitle(f'{title} | {name} examples ({min(p0 + per, k)}/{k} of {len(pool)} units)'
                             if k else f'{title} | no {name} units', fontsize=11)
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)

    save_responsive_units(res, pdf_path.with_suffix('.json'))
    print(f'Saved report: {pdf_path}')
    return pdf_path, res


def time_resolved_distance(model_dir, step, task=None, n_trials=512, seed=0,
                           zscore='all', responsive_only=True, criteria=None):
    """
    Time-resolved Go vs No-Go distance for one checkpoint and one task.

    Parameters
    ----------
    model_dir : model directory (config.json + checkpoint_<step>.pkl)
    step      : checkpoint step; 0 -> checkpoint_0.pkl, else checkpoint_init.pkl,
                else a random network from config.json (same rule as
                discrimination_vs_behavior)
    task      : stimulus set to simulate ('pure' | 'harmonic' | 'wn');
                default = first task the model was trained on
    seed      : seeds the trial draw (identical across models for the same seed,
                so two models see the same trials) and, separately, the noise
    zscore, responsive_only, criteria : see select_units (defaults: Z-score each
                unit over all trials/time; responsive units only)

    Returns
    -------
    dict with
      t_ms       (T,)    time relative to stimulus onset (ms), whole trial
      dist       (T,)    population Euclidean distance D(t) between the class-mean
                         Go and No-Go vectors at each time step (selected units)
      diff       (T, k)  signed r_Go - r_NoGo of each selected unit
      per_neuron (T, k)  |diff|: single-neuron Euclidean distance at each time step
      noise_sd   (T,)    population SD of the trial-to-trial scatter at each time step:
                         sqrt(sum_i var_i(t)), var_i = within-class (Go / No-Go pooled)
                         variance of unit i across trials
      dist_norm  (T,)    dist / noise_sd: the distance in units of the population noise
      noise_sd_per_unit (T,)  noise_sd / sqrt(k) and
      dist_per_unit     (T,)  dist / sqrt(k): the same two quantities normalised to the number
                         k of selected units (comparable across sessions / models)
      units      (k,)    indices of the selected units
      stim_ms    (on, off) stimulus onset / offset, ms from stimulus onset
      step, task, model_dir
    """
    sim = _simulate_checkpoint(model_dir, step, task, n_trials, seed)
    hs, trial, task = sim['hs'], sim['trial'], sim['task']
    dt, stim_on, stim_off = sim['dt'], sim['stim_on'], sim['stim_off']
    go = trial.is_go
    z, idx = select_units(hs, stim_on, dt, zscore, responsive_only, criteria)
    diff = z[:, go].mean(1) - z[:, ~go].mean(1)               # (T, k)
    dist = np.linalg.norm(diff, axis=1)
    # trial-to-trial noise at each time step: residuals from the class means, pooled over classes
    ss = (((z[:, go] - z[:, go].mean(1, keepdims=True)) ** 2).sum(1)
          + ((z[:, ~go] - z[:, ~go].mean(1, keepdims=True)) ** 2).sum(1))      # (T, k)
    noise_sd = np.sqrt(ss.sum(1) / max(z.shape[1] - 2, 1))                      # (T,)
    return dict(t_ms=(np.arange(hs.shape[0]) - stim_on) * dt,
                dist=dist, noise_sd=noise_sd,
                noise_sd_per_unit=noise_sd / np.sqrt(z.shape[2]),
                dist_per_unit=dist / np.sqrt(z.shape[2]),
                dist_norm=np.divide(dist, noise_sd, out=np.full_like(dist, np.nan),
                                    where=noise_sd > 0),
                diff=diff, per_neuron=np.abs(diff), units=idx,
                stim_ms=((0, (stim_off - stim_on) * dt)),
                step=step, task=task, model_dir=str(model_dir))


def load_saved_rows(model_dir, window_ms=None):
    """
    Load rows previously saved by discrimination_vs_behavior (no recomputation).
    window_ms : (start, end) as used when computing; None = default stimulus window.
    """
    model_dir = Path(model_dir)
    out_dir = model_dir / 'neuron_discrimination_vs_behavior'
    if window_ms is None:
        geo = _geometry()
        window_ms = (0, (geo['stim_off'] - geo['stim_on']) * geo['dt'])
    for p in sorted(out_dir.glob('perf_vs_distance_*.json')):
        with open(p) as f:
            saved = json.load(f)
        if np.allclose(saved['window_ms'], window_ms):
            d_ = saved.get('definition', {})
            if not (d_.get('aligned_to_stim_onset') and d_.get('noise_normalised')
                    and d_.get('behav_dprime') and d_.get('per_unit_normalised')):
                raise FileNotFoundError(
                    f'{p} was saved with an old definition (no Z-scoring / unit selection, '
                    f'trials not cut around their own stimulus onset, or without the '
                    f"noise-normalised distance / behavioural d' columns). "
                    f'Recompute with reload = False.')
            print(f'{model_dir.name}: loaded {p.name}  (definition: '
                  f'zscore={saved["definition"]["zscore"]}, '
                  f'responsive_only={saved["definition"]["responsive_only"]})')
            return saved['rows']
    raise FileNotFoundError(
        f'No saved result for window {tuple(window_ms)} ms in {out_dir}. '
        f'Run with reload = False first.')


def plot_distance_vs_behavior(rows, tasks, window_ms, title, out_dir=None, show=False,
                              xkey='distance', xlabel='Euclidean distance, Go vs No-Go population',
                              tag=''):
    """Scatter of behaviour vs a distance column (xkey) of the rows, coloured by step.
    The file is perf_vs_distance<tag>_<window>ms.png."""
    metrics = [('correct', 'correct rate'), ('behav_dprime', "behavioural d'"),
               ('hit', 'hit rate'), ('fa', 'false-alarm rate')]
    steps = [r['step'] for r in rows]
    norm = matplotlib.colors.Normalize(min(steps), max(steps))
    fig, axes = plt.subplots(len(tasks), len(metrics), figsize=(5 * len(metrics), 4.3 * len(tasks)),
                             squeeze=False)
    for i, task in enumerate(tasks):
        tr = [r for r in rows if r['task'] == task]
        d = np.array([r[xkey] for r in tr], dtype=float)
        for j, (k, lab) in enumerate(metrics):
            y = np.array([r[k] for r in tr])
            ax = axes[i, j]
            sc = ax.scatter(d, y, c=[r['step'] for r in tr], cmap='viridis',
                            norm=norm, s=55, edgecolor='k', lw=0.4)
            ax.set(xlabel=xlabel, ylabel=lab)
            if k != 'behav_dprime':                 # rates live in [0, 1]; d' has its own range
                ax.set_ylim(-0.05, 1.05)
            r_txt = (f'r = {np.corrcoef(d, y)[0, 1]:.2f}'
                     if len(tr) > 2 and d.std() > 0 and y.std() > 0 else 'r = n/a')
            ax.set_title(f'{task}: {lab}   ({r_txt})')
    fig.colorbar(sc, ax=axes, label='checkpoint step (training time)', fraction=0.02)
    fig.suptitle(f'{title}  |  window {window_ms[0]}–{window_ms[1]} ms from stim onset')
    if out_dir is not None:
        fig.savefig(out_dir / f'perf_vs_distance{tag}_{window_ms[0]}_{window_ms[1]}ms.png',
                    dpi=140, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ── driver ────────────────────────────────────────────────────────────────────

def _load_json(path):
    text = Path(path).read_text(encoding='utf-8')
    return json.loads(re.sub(r',\s*([}\]])', r'\1', text))


def analyse(ckpt_path, args):
    ckpt_path = Path(ckpt_path)
    print(f'\n{"=" * 70}\nAnalysing: {ckpt_path}\n{"=" * 70}')
    config = _load_json(ckpt_path.parent / 'config.json')
    hp = config['training']
    task_params = config['task_params']
    task_list = config.get('task_list', config.get('tasks', ['pure', 'harmonic', 'wn']))
    train_tasks = config.get('train_task', config.get('task_type', 'pure'))
    if isinstance(train_tasks, str):
        train_tasks = [train_tasks]
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    params, step = ckpt['params'], int(ckpt['step'])
    dt = ckpt.get('dt', task_params['timing']['dt'])
    m = hp['model']
    cfg = dict(alpha=dt / m['tau'], sigma_x=m['sigma_x'], sigma_rec=m['sigma_rec'],
               activation=m.get('activation', 'relu'))

    out_dir = ckpt_path.parent / f'checkpoint_{step}' / 'neuron_discrimination'
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    a = dict(n_trials=args.n_trials, n_perm=args.n_perm)

    results, summary = {}, {}
    for task_type in train_tasks:
        res = analyse_task(params, task_type, task_list, cfg, a, rng)
        cls = classify(res, args.p_thresh, args.d_min)
        results[task_type] = res
        summary[task_type] = dict(
            n_go=res['n_go'], n_nogo=res['n_nogo'],
            n_units=int(len(cls)),
            n_go_pref=int((cls == 1).sum()),
            n_nogo_pref=int((cls == -1).sum()),
            n_nonselective=int((cls == 0).sum()),
            median_abs_dprime_stim=float(np.median(np.abs(res['epochs']['stim']['dprime']))),
            median_abs_dprime_post=float(np.median(np.abs(res['epochs']['post_stim']['dprime']))),
        )
        print(f'{task_type:10s} {summary[task_type]}')
        plot_task(res, cls, task_type, step, out_dir, dt)

        # per-unit table
        cols = ['unit', 'class', 'dprime_stim', 'auc_stim', 'p_stim',
                'dprime_post', 'auc_post', 'p_post']
        es, ep = res['epochs']['stim'], res['epochs']['post_stim']
        table = np.column_stack([np.arange(len(cls)), cls, es['dprime'], es['auc'], es['p'],
                                 ep['dprime'], ep['auc'], ep['p']])
        np.savetxt(out_dir / f'units_{task_type}_{step}.csv', table, delimiter=',',
                   header=','.join(cols), comments='',
                   fmt=['%d', '%d'] + ['%.5f'] * 6)
        np.save(out_dir / f'dprime_t_{task_type}_{step}.npy', res['dprime_t'])

    plot_cross_task(results, step, out_dir)
    with open(out_dir / f'summary_{step}.json', 'w') as f:
        json.dump(dict(ckpt=str(ckpt_path), step=step, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'checkpoints'},
                       tasks=summary), f, indent=2)
    print(f'Saved to {out_dir}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='*', type=Path)
    p.add_argument('--model-dir', type=Path,
                   help='run discrimination_vs_behavior on this model directory')
    p.add_argument('--window', type=float, nargs=2, metavar=('START_MS', 'END_MS'),
                   help='evaluation window in ms from stim onset (default: whole stimulus period)')
    p.add_argument('--n-trials', type=int, default=512, help='trials per task (default 512)')
    p.add_argument('--n-perm', type=int, default=500, help='AUC permutations (default 500)')
    p.add_argument('--p-thresh', type=float, default=0.01, help='permutation p cutoff (default 0.01)')
    p.add_argument('--d-min', type=float, default=0.5, help="min |d'| to call a unit selective (default 0.5)")
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.model_dir:
        discrimination_vs_behavior(
            args.model_dir, window_ms=tuple(args.window) if args.window else None,
            n_trials=args.n_trials, seed=args.seed)
    for c in args.checkpoints:
        analyse(c, args)


if __name__ == '__main__':
    main()
