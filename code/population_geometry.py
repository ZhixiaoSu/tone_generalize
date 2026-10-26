"""
Population geometry of Go / No-Go responses in trained LeakyRNNs: PCA trajectories and
dimensionality (participation ratio), following the A1 analysis in the mouse study.

  * PCA / manifold geometry: Go and No-Go condition-mean trajectories (Z-scored,
    responsive units, 0-300 ms after stimulus onset) are stacked into a [2T x N] matrix;
    PCA is fitted on it and the top 3 PCs are kept.
  * Participation ratio  PR = (sum_i lam_i)^2 / sum_i lam_i^2, where lam_i are the
    eigenvalues of the unit x unit covariance matrix across trials, each trial being the
    population response averaged over the window.  nPR = PR / N responsive units.
  * Two further readouts of the same eigenspectrum: fraction of variance in the first
    covariance PC, and an entropy-based effective dimension exp(H(lam / sum lam)).
  * Behaviour of each session (hit / false alarm / lick fraction) and outcome-matched
    PR comparisons (trials subsampled so that two groups have the same numbers of
    Hit / Miss / False Alarm / Correct Reject trials, or of lick / no-lick trials).

A "session" is one independent simulation of a batch of trials (own trial draw and own
noise) of a fixed checkpoint, standing in for a recording session.  Sessions of the
same network are repeated measurements of one network, not independent animals.

Only numpy is needed for the analysis; checkpoints are unpickled by neuron_discrimination.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from neuron_discrimination import (_simulate_checkpoint, _list_checkpoints,
                                   select_units)
from task import load_params as load_task_params


def participation_ratio(X):
    """PR, normalised PR, PC1 fraction and entropy effective dimension of X [trials x N]."""
    C = np.atleast_2d(np.cov(X, rowvar=False))                 # (N, N) across trials
    lam = np.clip(np.linalg.eigvalsh(C), 0.0, None)[::-1]      # descending
    tot = lam.sum()
    if tot <= 0:
        return dict(pr=np.nan, npr=np.nan, pc1_frac=np.nan, eff_dim=np.nan)
    pr = tot ** 2 / np.sum(lam ** 2)
    p = lam[lam > 0] / tot
    return dict(pr=float(pr), npr=float(pr / C.shape[0]), pc1_frac=float(lam[0] / tot),
                eff_dim=float(np.exp(-np.sum(p * np.log(p)))))


def trajectory_pca(go_mean, nogo_mean, k=3):
    """
    PCA on the stacked condition-mean trajectories [2T x N] (columns centred).
    Returns dict(proj_go (T,k), proj_nogo (T,k), var_ratio (k,)).
    """
    M = np.concatenate([go_mean, nogo_mean], axis=0)
    mu = M.mean(0)
    U, S, Vt = np.linalg.svd(M - mu, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    W = Vt[:k].T                                               # (N, k)
    return dict(proj_go=(go_mean - mu) @ W, proj_nogo=(nogo_mean - mu) @ W,
                var_ratio=var[:k])


def _pca_samples(M, k=3):
    """PCA of samples M [n x N] (columns centred). Returns (mean, W [N x k], var ratio (k,),
    participation ratio of the full eigenspectrum)."""
    mu = M.mean(0)
    _, S, Vt = np.linalg.svd(M - mu, full_matrices=False)
    lam = S ** 2
    return mu, Vt[:k].T, lam[:k] / lam.sum(), float(lam.sum() ** 2 / np.sum(lam ** 2))


def _var_along(M, w):
    """Fraction of the total variance of samples M [n x N] that lies along unit vector w."""
    Mc = M - M.mean(0)
    return float(((Mc @ w) ** 2).sum() / (Mc ** 2).sum())


def separating_direction(z_post, is_go):
    """
    Unit vector that best separates Go from No-Go in the window z_post (P, B, N):
    the difference of the Go and No-Go mean population vectors (the coding direction).
    Returns None if the classes coincide.
    """
    d = z_post[:, is_go].mean((0, 1)) - z_post[:, ~is_go].mean((0, 1))
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def separating_variance(zw, z_post, is_go):
    """
    Variance explained by the Go / No-Go separating dimension, for mean and for noisy
    individual trajectories, over the analysis window zw (W, B, N) and over the post-offset
    window z_post (P, B, N).  The direction is estimated on the post-offset window of one
    half of the trials and evaluated on the other half (2-fold cross-fitting, averaged), so
    the number is not inflated by fitting and testing on the same trials.
    Returns dict(sep_var_mean, sep_var_all, sep_var_mean_post, sep_var_all_post,
    sep_w): the first four are fractions of total variance, sep_w the direction from all trials.
    """
    B = zw.shape[1]
    half = np.arange(B) % 2                                    # alternate trials -> two folds
    vals = {k: [] for k in ('sep_var_mean', 'sep_var_all', 'sep_var_mean_post', 'sep_var_all_post')}
    for fit, test in ((half == 0, half == 1), (half == 1, half == 0)):
        if not (is_go[fit].any() and (~is_go[fit]).any() and is_go[test].any() and (~is_go[test]).any()):
            continue
        w = separating_direction(z_post[:, fit], is_go[fit])
        if w is None:
            continue
        for tag, z_ in (('', zw), ('_post', z_post)):
            zt, gt = z_[:, test], is_go[test]
            mean_traj = np.concatenate([zt[:, gt].mean(1), zt[:, ~gt].mean(1)], axis=0)   # [2T x N]
            vals['sep_var_mean' + tag].append(_var_along(mean_traj, w))
            vals['sep_var_all' + tag].append(_var_along(zt.reshape(-1, zt.shape[2]), w))
    out = {k: (float(np.mean(v)) if v else np.nan) for k, v in vals.items()}
    out['sep_w'] = separating_direction(z_post, is_go)
    return out


def session_geometry(z, is_go, stim_on, dt, window_ms=(0, 300), stim_off=None,
                     sep_after_off_ms=(0, None), n_show_trials=20, cutoff_ms=None):
    """
    Geometry of one session.  z: (T, B, N) Z-scored responses of the selected units,
    aligned so that index stim_on is the stimulus onset; is_go: (B,) bool.

    Mean-trajectory analysis (as in the paper):  PCA of the stacked Go / No-Go condition-mean
    trajectories -> var_ratio, proj_go, proj_nogo.
    Noisy-trajectory analysis: PCA of ALL individual trial trajectories (every trial x time
    sample of the window as one observation) -> var_ratio_all, pr_all (participation ratio of
    that spectrum), proj_go_all / proj_nogo_all (the mean trajectories in that PC space) and
    proj_trials_go / proj_trials_nogo (n_show_trials single trials).
    Per-trial window-mean covariance: pr, npr, pc1_frac, eff_dim (as before).

    If stim_off (aligned index of stimulus offset) is given, the dimension that best
    separates Go from No-Go after stimulus offset (mean-difference direction over
    [stim_off + sep_after_off_ms[0], stim_off + sep_after_off_ms[1]], None = end of trial) is
    extracted and the variance along it is computed: sep_var_mean / sep_var_all (analysis
    window; mean / noisy trajectories) and *_post (same, within the post-offset window),
    plus sep_pc1_cos (|cos| between it and PC1 of the noisy-trajectory PCA).
    cutoff_ms : if given, only the first cutoff_ms after stimulus onset are analysed: the
    analysis window and the post-offset window are both truncated there.
    Returns the geometry dict plus 'X', the per-trial window-mean responses [B x N].
    """
    if cutoff_ms is not None:
        z = z[:stim_on + int(round(cutoff_ms / dt))]
    t0 = stim_on + int(round(window_ms[0] / dt))
    t1 = stim_on + int(round(window_ms[1] / dt))
    zw = z[t0:t1]                                              # (W, B, N)
    X = zw.mean(0)                                             # per-trial window mean
    out = participation_ratio(X)
    out.update(trajectory_pca(zw[:, is_go].mean(1), zw[:, ~is_go].mean(1)))
    out['n_units'] = int(z.shape[2])
    out['X'] = X

    # PCA of all individual trajectories
    mu, Wp, var_all, pr_all = _pca_samples(zw.reshape(-1, zw.shape[2]))
    out.update(var_ratio_all=var_all, pr_all=pr_all,
               proj_go_all=(zw[:, is_go].mean(1) - mu) @ Wp,
               proj_nogo_all=(zw[:, ~is_go].mean(1) - mu) @ Wp,
               proj_trials_go=np.stack([(zw[:, i] - mu) @ Wp for i in np.flatnonzero(is_go)[:n_show_trials]]),
               proj_trials_nogo=np.stack([(zw[:, i] - mu) @ Wp for i in np.flatnonzero(~is_go)[:n_show_trials]]))

    if stim_off is not None:
        p0 = stim_off + int(round(sep_after_off_ms[0] / dt))
        p1 = z.shape[0] if sep_after_off_ms[1] is None else stim_off + int(round(sep_after_off_ms[1] / dt))
        sep = separating_variance(zw, z[p0:p1], is_go)
        w = sep.pop('sep_w')
        out.update(sep)
        out['sep_pc1_cos'] = float(abs(w @ Wp[:, 0])) if w is not None else np.nan
    return out


def trial_responses(ys, geo, dt, thresh=None):
    """
    Did the network respond (lick) on each trial?  Same rule as train.evaluate_training:
    output peak above `thresh` within 2 s after onset, and below 0.3 before onset.
    ys: (T_aligned, B, 1) aligned output.  Returns bool (B,).
    """
    if thresh is None:
        thresh = load_task_params().get('response', {}).get('response_thresh', 0.5)
    pre = geo['fixed_pre']
    reaction = min(int(2000 / dt), geo['fixed_post'])
    pre_peak = ys[:pre, :, 0].max(0) if pre > 0 else np.zeros(ys.shape[1])
    post_peak = ys[pre:pre + reaction, :, 0].max(0)
    return (post_peak > thresh) & (pre_peak < 0.3)


def pick_steps(model_dir, n=8, include_zero=True):
    """Up to n checkpoint steps spread evenly over the available ones (+ step 0)."""
    steps = [s for s, _ in _list_checkpoints(model_dir)]
    if len(steps) > n:
        steps = [steps[i] for i in np.unique(np.round(np.linspace(0, len(steps) - 1, n)).astype(int))]
    return ([0] if include_zero and 0 not in steps else []) + steps


def common_steps(model_dirs, n=8, include_zero=True, max_step=None):
    """
    Up to n checkpoint steps that exist in every model dir (so models can be compared at
    the same training step), spread evenly; step 0 is added if include_zero.
    """
    sets = [set(s for s, _ in _list_checkpoints(d)) for d in model_dirs]
    steps = sorted(set.intersection(*sets))
    if max_step is not None:
        steps = [s for s in steps if s <= max_step]
    if len(steps) > n:
        steps = [steps[i] for i in np.unique(np.round(np.linspace(0, len(steps) - 1, n)).astype(int))]
    return ([0] if include_zero and 0 not in steps else []) + steps


def run_condition(label, model_dir, task, steps, n_sessions=10, n_trials=200, seed0=0,
                  window_ms=(0, 300), zscore='all', responsive_only=True, criteria=None,
                  keep_trials=False, verbose=True, sep_after_off_ms=(0, None),
                  n_show_trials=20, cutoff_ms=None):
    """
    Geometry of every (step, session) for one model / task.

    Returns a list of dict rows: label, task, step, session, pr, npr, pc1_frac, eff_dim,
    n_units, var_ratio (3,), proj_go / proj_nogo (T, 3), hit, fa, lick_frac, and the
    noisy-trajectory PCA / separating-dimension fields described in session_geometry
    (var_ratio_all, pr_all, proj_*_all, proj_trials_*, sep_var_*, sep_pc1_cos).
    sep_after_off_ms : window after stimulus offset (ms from offset; end None = end of
                       trial) used to find the dimension that separates Go from No-Go
    n_show_trials    : number of single-trial trajectories kept per class for plotting
    cutoff_ms        : analyse only this many ms after stimulus onset (None = whole trial);
                       truncates both the analysis window and the post-offset window
    With keep_trials the rows also hold X [trials x units], resp (lick, bool) and is_go,
    needed for outcome-matched comparisons.
    Sessions with no responsive unit are skipped.
    """
    rows = []
    for step in steps:
        for s in range(n_sessions):
            sim = _simulate_checkpoint(model_dir, step, task, n_trials, seed=seed0 + s,
                                       return_y=True)
            try:
                z, idx = select_units(sim['hs'], sim['stim_on'], sim['dt'], zscore,
                                      responsive_only, criteria)
            except ValueError:                                 # no responsive units
                continue
            is_go = sim['trial'].is_go
            g = session_geometry(z, is_go, sim['stim_on'], sim['dt'], window_ms,
                                 stim_off=sim['stim_off'], sep_after_off_ms=sep_after_off_ms,
                                 n_show_trials=n_show_trials, cutoff_ms=cutoff_ms)
            resp = trial_responses(sim['ys'], sim['geo'], sim['dt'])
            X = g.pop('X')
            g.update(hit=float(resp[is_go].mean()), fa=float(resp[~is_go].mean()),
                     lick_frac=float(resp.mean()))
            if keep_trials:
                g.update(X=X, resp=resp, is_go=is_go)
            rows.append(dict(label=label, task=task, step=int(step), session=s, **g))
        if verbose:
            r = [x for x in rows if x['step'] == step]
            if r:
                print(f'{label:>10s} step {step:6d} {task:9s}  PR={np.mean([x["pr"] for x in r]):5.2f}  '
                      f'nPR={np.mean([x["npr"] for x in r]):.3f}  units={np.mean([x["n_units"] for x in r]):.0f}  '
                      f'hit={np.mean([x["hit"] for x in r]):.2f}  FA={np.mean([x["fa"] for x in r]):.2f}')
    return rows


def matched_pr(rows_a, rows_b, by='outcome', n_rep=20, seed=0, min_trials=20):
    """
    PR of two groups after subsampling trials to equate their composition, session by
    session (session i of group A with session i of group B; rows need keep_trials=True).

    by='outcome': equal numbers of Hit / Miss / False Alarm / Correct Reject trials
    by='lick'   : equal numbers of lick (response) and no-lick trials
    For every category the number of trials is the minimum over the two groups; the
    subsampling is repeated n_rep times and PR averaged.  Sessions with fewer than
    min_trials matched trials are skipped.

    Returns (pr_a, pr_b, n_matched): arrays over the retained sessions.
    """
    rng = np.random.default_rng(seed)
    by_sess_b = {r['session']: r for r in rows_b}

    def category(r):
        if by == 'lick':
            return r['resp'].astype(int)
        return r['is_go'].astype(int) * 2 + r['resp'].astype(int)  # 0 CR,1 FA,2 Miss,3 Hit

    out_a, out_b, n_m = [], [], []
    for ra in rows_a:
        rb = by_sess_b.get(ra['session'])
        if rb is None:
            continue
        ca, cb = category(ra), category(rb)
        counts = {c: min((ca == c).sum(), (cb == c).sum()) for c in np.unique(np.concatenate([ca, cb]))}
        total = sum(counts.values())
        if total < min_trials:
            continue
        pa, pb = [], []
        for _ in range(n_rep):
            ia = np.concatenate([rng.choice(np.flatnonzero(ca == c), n, replace=False)
                                 for c, n in counts.items() if n > 0])
            ib = np.concatenate([rng.choice(np.flatnonzero(cb == c), n, replace=False)
                                 for c, n in counts.items() if n > 0])
            pa.append(participation_ratio(ra['X'][ia])['pr'])
            pb.append(participation_ratio(rb['X'][ib])['pr'])
        out_a.append(np.nanmean(pa)); out_b.append(np.nanmean(pb)); n_m.append(total)
    return np.array(out_a), np.array(out_b), np.array(n_m)


# ── single-neuron statistics and hierarchical bootstrap ────────────────────────

def single_neuron_stats(z, is_go, stim_on, dt, window_ms=(0, 300)):
    """
    Per-unit Go vs No-Go separation of one session.  z: (T, B, N) Z-scored responses of the
    selected units, aligned so index stim_on is the stimulus onset; is_go: (B,) bool.

    Each trial's response is the window average X [B x N].  For every unit i
      dist_i      = |mean_Go,i - mean_NoGo,i|     single-neuron Euclidean distance
      sd_i        = pooled within-class SD of X_i across trials (noise of that unit)
      dist_norm_i = dist_i / sd_i                 the distance in units of the unit's noise
    Returns dict(dist, sd, dist_norm), each (N,).  dist_norm is NaN where sd is 0.
    """
    t0 = stim_on + int(round(window_ms[0] / dt))
    t1 = stim_on + int(round(window_ms[1] / dt))
    X = z[t0:t1].mean(0)                                       # (B, N)
    go, nogo = X[is_go], X[~is_go]
    dist = np.abs(go.mean(0) - nogo.mean(0))
    res = np.concatenate([go - go.mean(0), nogo - nogo.mean(0)], axis=0)
    sd = np.sqrt((res ** 2).sum(0) / max(len(X) - 2, 1))
    dist_norm = np.divide(dist, sd, out=np.full_like(dist, np.nan), where=sd > 1e-12)
    return dict(dist=dist, sd=sd, dist_norm=dist_norm)


def run_neuron_condition(model_dirs, task, step, n_sessions=10, n_trials=200, seed0=0,
                         window_ms=(0, 300), zscore='all', responsive_only=True,
                         criteria=None):
    """
    Per-neuron statistics for one condition = (models, task, checkpoint step).

    model_dirs : one model directory or a list of them (all are used).  Every
                 (model dir, session) is one 'animal' in the hierarchical analysis; a
                 session is an independent simulation (own trials and noise).
    step       : checkpoint step; 0 = checkpoint_init.pkl if present, else a random network.
    Returns a list (one entry per animal) of dict(model_dir, session, units, dist, sd,
    dist_norm); sessions without responsive units are skipped.
    """
    if not isinstance(model_dirs, (list, tuple)):
        model_dirs = [model_dirs]
    out = []
    for d in model_dirs:
        for s in range(n_sessions):
            sim = _simulate_checkpoint(d, step, task, n_trials, seed=seed0 + s)
            try:
                z, idx = select_units(sim['hs'], sim['stim_on'], sim['dt'], zscore,
                                      responsive_only, criteria)
            except ValueError:
                continue
            st = single_neuron_stats(z, sim['trial'].is_go, sim['stim_on'], sim['dt'], window_ms)
            out.append(dict(model_dir=str(d), session=s, units=idx, **st))
    return out


def hierarchical_bootstrap(animals_a, animals_b, key, n_boot=10000, stat='median', seed=0):
    """
    Hierarchical bootstrap of the difference (B - A) of a per-neuron statistic.

    animals_*  : lists from run_neuron_condition (animal -> its neurons)
    Each iteration resamples animals with replacement and, within every resampled animal,
    resamples its neurons with replacement; the group statistic ('median' or 'mean') is
    taken over the pooled neurons.  Two-sided p = 2 * min(P(diff <= 0), P(diff >= 0)),
    with the +1 correction.
    Returns dict(diff (observed B - A), p, ci (2.5 / 97.5 %), boot (n_boot,)).
    """
    f = np.median if stat == 'median' else np.mean
    pools = []
    for animals in (animals_a, animals_b):
        arrs = []
        for a in animals:
            v = np.asarray(a[key], dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                arrs.append(v)
        pools.append(arrs)
    rng = np.random.default_rng(seed)

    def one(arrs):
        pick = rng.integers(0, len(arrs), len(arrs))
        return f(np.concatenate([arrs[i][rng.integers(0, len(arrs[i]), len(arrs[i]))] for i in pick]))

    boot = np.array([one(pools[1]) - one(pools[0]) for _ in range(n_boot)])
    obs = f(np.concatenate(pools[1])) - f(np.concatenate(pools[0]))
    p = min(1.0, (2 * min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (n_boot + 1))
    return dict(diff=float(obs), p=float(p), ci=tuple(np.percentile(boot, [2.5, 97.5])), boot=boot)
