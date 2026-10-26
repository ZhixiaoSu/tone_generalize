"""
Bifurcation analysis script: alpha sweep between any two task inputs.

Blends inputs linearly from task A (alpha=0) to task B (alpha=1), finding
fixed points at each alpha and phase.

Outputs (in <ckpt_dir>/bifurcation_<task_a>_<task_b>/):
  bifur_3d_<phase>.html   — Plotly 3D scatter coloured by alpha
  bifur_1d.png            — PC1 of stable FPs vs alpha per phase
  bifur_count.png         — # stable FPs vs alpha per phase
  bifur_params.json       — parameters used in this run

Usage:
    python bifurcation_analysis.py <ckpt.pkl> --tasks TASK_A TASK_B [options]

Examples:
    python bifurcation_analysis.py .../checkpoint.pkl --tasks pure harmonic
    python bifurcation_analysis.py .../checkpoint.pkl --tasks wn harmonic
    python bifurcation_analysis.py .../checkpoint.pkl --tasks pure wn --n-alpha 11 --show
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
import jax
import jax.numpy as jnp
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from network import LeakyRNN
from task import generate_trials
from utils.fixed_points import find_fixed_points, get_phase_step, phase_label


# ── Plotly colour helper ──────────────────────────────────────────────────────

def _pc(c, alpha=None):
    r, g, b, *_ = mcolors.to_rgba(c)
    if alpha is not None:
        return f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.2f})'
    return f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'

def _fp3d(fig, x, y, z, fill_col, ec, sz, name, stable, showlegend=True, opacity=1.0):
    """Double-circle trick: WebGL Scatter3d ignores marker.line.width in the viewport.
    Paint a larger ec-coloured background circle first, then the fill-coloured one on top."""
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(color=ec, size=sz + 2, symbol='circle', opacity=opacity),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(color=fill_col, size=sz, symbol='circle', opacity=opacity),
        name=name, showlegend=showlegend,
    ))
    if not stable:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(color='black', size=max(sz - 3, 3), symbol='x',
                        line=dict(color='black', width=1.5), opacity=opacity),
            showlegend=False,
        ))

def _make_scene(ax):
    return dict(xaxis_title=ax[0], yaxis_title=ax[1], zaxis_title=ax[2],
                aspectmode='data')

GO_COL   = _pc('tab:blue')
NOGO_COL = _pc('tab:orange')


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse(ckpt_path: Path, args) -> None:
    ckpt_path = Path(ckpt_path)
    task_a, task_b = args.tasks
    blend_label = f'{task_a} → {task_b}'

    print(f'\n{"="*70}')
    print(f'Bifurcation analysis: {blend_label}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'{"="*70}')

    # ── load config + checkpoint ──────────────────────────────────────────────
    with open(ckpt_path.parent / 'config.json') as f:
        config = json.load(f)

    hp          = config['training']
    task_params = config['task_params']

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    params = ckpt['params']
    step   = ckpt.get('step', 0)
    dt     = ckpt.get('dt', task_params['timing']['dt'])
    stim_on = ckpt.get('stim_on', config.get('stim_on',
                int(task_params['timing']['pre_stim_dur_ms'] / dt)))
    stim_off = ckpt.get('stim_off', config.get('stim_off',
                stim_on + task_params['timing']['n_pulses'] *
                max(1, int(np.ceil(task_params['timing']['pulse_dur_ms'] / dt)))))
    pulse_steps = config.get('pulse_steps',
                  max(1, int(np.ceil(task_params['timing']['pulse_dur_ms'] / dt))))
    n_rnn = hp['model']['n_rnn']
    alpha = dt / hp['model']['tau']

    print(f'n_rnn={n_rnn}  alpha={alpha:.3f}  stim_on={stim_on}  stim_off={stim_off}')

    # ── run model on both tasks ───────────────────────────────────────────────
    model = LeakyRNN(n_rnn=n_rnn, n_out=1, alpha=alpha,
                     sigma_rec=hp['model']['sigma_rec'],
                     sigma_x=hp['model']['sigma_x'])
    task_list = config.get('task_list', config.get('tasks', ['pure', 'harmonic', 'wn']))
    fp_sz = 8
    def _run_task(task_type, seed):
        trial = generate_trials(task_type, task_list, mode='test',
                                batch_size=args.n_trials,
                                rng=np.random.default_rng(seed))
        _, hs_jax = model.apply({'params': params}, jnp.array(trial.x),
                                noise_input=False, noise_rec=False,
                                rngs={'noise': jax.random.PRNGKey(0)})
        return trial, np.array(hs_jax)   # (T, batch, n_rnn)

    print(f'\nRunning task A ({task_a}) trials…')
    trial_a, hs_a = _run_task(task_a, seed=1)
    print(f'Running task B ({task_b}) trials…')
    trial_b, hs_b = _run_task(task_b, seed=2)

    go_a   = trial_a.is_go
    nogo_a = ~trial_a.is_go
    go_b   = trial_b.is_go
    nogo_b = ~trial_b.is_go
    print(f'Task A ({task_a}): {go_a.sum()} Go, {nogo_a.sum()} No-Go')
    print(f'Task B ({task_b}): {go_b.sum()} Go, {nogo_b.sum()} No-Go')

    out_dir = ckpt_path.parent / f'bifurcation_{task_a}_{task_b}' / f'checkpoint_{step}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {out_dir}')

    # ── PCA on pooled hidden states ───────────────────────────────────────────
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([hs_a.reshape(-1, n_rnn),
                             hs_b.reshape(-1, n_rnn)], axis=0))
    var_exp = pca.explained_variance_ratio_ * 100
    axis_labels = [f'PC1 ({var_exp[0]:.1f}%)',
                   f'PC2 ({var_exp[1]:.1f}%)',
                   f'PC3 ({var_exp[2]:.1f}%)']

    # ── alpha sweep ───────────────────────────────────────────────────────────
    alphas = np.linspace(0, 1, args.n_alpha)
    phases = args.phases
    cmap_coolwarm = plt.cm.coolwarm
    norm_alpha    = plt.Normalize(0, 1)
    cmap_phase    = plt.cm.tab10

    # sweep_results[phase_key] = list of {go: {h,is_stable}, nogo: ...} per alpha
    sweep_results = {}

    for ph_idx, (phase, pulse_num) in enumerate(phases):
        t_step = get_phase_step(phase, stim_on, stim_off, pulse_steps,
                                pulse_num=pulse_num if pulse_num else 2)
        key = phase_label(phase, pulse_num)
        print(f'\n── Phase: {key}  (t={t_step})')

        x_a_go   = trial_a.x[t_step, np.where(go_a)[0][0],   :].copy()
        x_a_nogo = trial_a.x[t_step, np.where(nogo_a)[0][0], :].copy()
        x_b_go   = trial_b.x[t_step, np.where(go_b)[0][0],   :].copy()
        x_b_nogo = trial_b.x[t_step, np.where(nogo_b)[0][0], :].copy()

        h_go   = hs_a[t_step, go_a,   :]
        h_nogo = hs_a[t_step, nogo_a, :]

        alpha_results = []
        for a in alphas:
            x_go_a   = (1.0 - a) * x_a_go   + a * x_b_go
            x_nogo_a = (1.0 - a) * x_a_nogo + a * x_b_nogo
            cond_res = {}
            for cond, x_frozen, h_inits in [('go', x_go_a, h_go),
                                             ('nogo', x_nogo_a, h_nogo)]:
                cond_res[cond] = find_fixed_points(
                    params, x_frozen, h_inits, alpha,
                    n_iters=args.n_iters, lr=args.lr,
                    tol=args.tol, unique_tol=args.unique_tol,
                    verbose=False)
            alpha_results.append(cond_res)
            n_go_s   = int(cond_res['go']['is_stable'].sum())   if len(cond_res['go']['h'])   > 0 else 0
            n_nogo_s = int(cond_res['nogo']['is_stable'].sum()) if len(cond_res['nogo']['h']) > 0 else 0
            print(f'  α={a:.2f}  go={n_go_s} stable  nogo={n_nogo_s} stable')

        sweep_results[key] = alpha_results

    # ── save parameters ───────────────────────────────────────────────────────
    bifur_params_out = {
        'ckpt_path':  str(ckpt_path),
        'step':       int(step),
        'task_a':     task_a,
        'task_b':     task_b,
        'n_alpha':    args.n_alpha,
        'n_trials':   args.n_trials,
        'n_iters':    args.n_iters,
        'lr':         args.lr,
        'tol':        args.tol,
        'unique_tol': args.unique_tol,
        'phases':     [[ph, pn] for ph, pn in phases],
    }
    params_path = out_dir / 'bifur_params.json'
    with open(params_path, 'w') as f:
        json.dump(bifur_params_out, f, indent=2)
    print(f'\nSaved params: {params_path}')

    # ── Plotly: 3D scatter per phase, coloured by alpha ───────────────────────
    def _save_plotly(fig, path):
        if args.show:
            fig.show()
        fig.write_html(str(path))
        print(f'Saved: {path}')

    for ph_idx, (key, alpha_results) in enumerate(sweep_results.items()):
        fig = go.Figure()
        for a_idx, (a, cond_res) in enumerate(zip(alphas, alpha_results)):
            col  = _pc(cmap_coolwarm(norm_alpha(a)))
            lbl  = f'α={a:.2f}'
            show = bool(a_idx == 0 or a_idx == len(alphas) - 1
                        or a_idx == len(alphas) // 2)
            for cond, ec in [('go', GO_COL), ('nogo', NOGO_COL)]:
                if cond not in cond_res or len(cond_res[cond]['h']) == 0:
                    continue
                fp  = cond_res[cond]
                pc  = pca.transform(fp['h'])
                s   = fp['is_stable']
                if s.any():
                    _fp3d(fig, pc[s, 0], pc[s, 1], pc[s, 2], col, ec, fp_sz,
                          f'{lbl} {cond} stable', stable=True,
                          showlegend=show, opacity=0.85)
                if (~s).any():
                    _fp3d(fig, pc[~s, 0], pc[~s, 1], pc[~s, 2], col, ec, fp_sz,
                          f'{lbl} {cond} unstable', stable=False,
                          showlegend=show, opacity=0.55)
        fig.update_layout(
            title=f'Bifurcation 3D — {blend_label} — {key}<br>'
                  f'<sup>fill: {task_a}(blue)→{task_b}(red)  '
                  f'go edge=blue  no-go edge=orange  circle=stable  circle+×=unstable</sup>',
            scene=_make_scene(axis_labels),
            height=650,
            paper_bgcolor='white',
        )
        fname = key.replace(' ', '_').replace('/', '-')
        _save_plotly(fig, out_dir / f'bifur_3d_{fname}.html')

    # ── Plotly: 3D alpha-axis scatter per phase (PC1, PC2, Z=alpha) ───────────
    alpha_axis_labels = [axis_labels[0], axis_labels[1],
                         f'α  ({task_a} → {task_b})']

    # Scale alpha to span the same range as PC1 so the 3-D aspect is balanced
    _all_pc1 = [pca.transform(cr[cnd]['h'])[:, 0]
                for ar in sweep_results.values()
                for cr in ar
                for cnd in ('go', 'nogo')
                if cnd in cr and len(cr[cnd]['h']) > 0]
    _pc1_lo, _pc1_hi = (float(np.concatenate(_all_pc1).min()),
                         float(np.concatenate(_all_pc1).max())) if _all_pc1 else (0.0, 1.0)
    def _a2z(a):
        return _pc1_lo + float(a) * (_pc1_hi - _pc1_lo)
    _a_tickvals = [_a2z(v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
    _a_ticktext  = ['0', '0.25', '0.5', '0.75', '1']
    _alpha_scene = dict(
        xaxis_title=alpha_axis_labels[0],
        yaxis_title=alpha_axis_labels[1],
        zaxis=dict(title=alpha_axis_labels[2],
                   tickvals=_a_tickvals, ticktext=_a_ticktext),
        aspectmode='cube',
    )
    for ph_idx, (key, alpha_results) in enumerate(sweep_results.items()):
        fig_a = go.Figure()
        for a_idx, (a, cond_res) in enumerate(zip(alphas, alpha_results)):
            col  = _pc(cmap_coolwarm(norm_alpha(a)))
            lbl  = f'α={a:.2f}'
            show = bool(a_idx == 0 or a_idx == len(alphas) - 1
                        or a_idx == len(alphas) // 2)
            for cond, ec in [('go', GO_COL), ('nogo', NOGO_COL)]:
                if cond not in cond_res or len(cond_res[cond]['h']) == 0:
                    continue
                fp  = cond_res[cond]
                pc  = pca.transform(fp['h'])
                s   = fp['is_stable']
                z   = np.full(len(fp['h']), _a2z(a))
                if s.any():
                    _fp3d(fig_a, pc[s, 0], pc[s, 1], z[s], col, ec, fp_sz,
                          f'{lbl} {cond} stable', stable=True,
                          showlegend=show, opacity=0.85)
                if (~s).any():
                    _fp3d(fig_a, pc[~s, 0], pc[~s, 1], z[~s], col, ec, fp_sz,
                          f'{lbl} {cond} unstable', stable=False,
                          showlegend=show, opacity=0.55)
        fig_a.update_layout(
            title=f'Bifurcation 3D (α-axis) — {blend_label} — {key}<br>'
                  f'<sup>fill: {task_a}(blue)→{task_b}(red)  '
                  f'go edge=blue  no-go edge=orange  circle=stable  circle+×=unstable</sup>',
            scene=_alpha_scene,
            height=650,
            paper_bgcolor='white',
        )
        fname = key.replace(' ', '_').replace('/', '-')
        _save_plotly(fig_a, out_dir / f'bifur_3d_alpha_{fname}.html')

    # ── Plotly: combined alpha-axis plot, all pulse_on phases, viridis by pulse ─
    pulse_entries = [(key, sweep_results[key])
                     for (phase, pn), key in zip(phases, sweep_results)
                     if phase == 'pulse_on']
    if pulse_entries:
        cmap_viridis = plt.cm.viridis
        n_pulses     = len(pulse_entries)
        fig_comb     = go.Figure()
        for p_idx, (key, alpha_results) in enumerate(pulse_entries):
            pulse_col = _pc(cmap_viridis(p_idx / max(n_pulses - 1, 1)))
            shown_stable   = {cond: False for cond in ('go', 'nogo')}
            shown_unstable = {cond: False for cond in ('go', 'nogo')}
            for a, cond_res in zip(alphas, alpha_results):
                for cond, ec in [('go', GO_COL), ('nogo', NOGO_COL)]:
                    if cond not in cond_res or len(cond_res[cond]['h']) == 0:
                        continue
                    fp  = cond_res[cond]
                    pc  = pca.transform(fp['h'])
                    s   = fp['is_stable']
                    z   = np.full(len(fp['h']), _a2z(a))
                    if s.any():
                        show = not shown_stable[cond]
                        _fp3d(fig_comb, pc[s, 0], pc[s, 1], z[s], pulse_col, ec, fp_sz,
                              f'{key} {cond} stable', stable=True,
                              showlegend=show, opacity=0.85)
                        shown_stable[cond] = True
                    if (~s).any():
                        show = not shown_unstable[cond]
                        _fp3d(fig_comb, pc[~s, 0], pc[~s, 1], z[~s], pulse_col, ec, fp_sz,
                              f'{key} {cond} unstable', stable=False,
                              showlegend=show, opacity=0.55)
                        shown_unstable[cond] = True
        fig_comb.update_layout(
            title=f'Bifurcation 3D — all pulses (α-axis) — {blend_label}<br>'
                  f'<sup>fill colour=pulse number (viridis)  '
                  f'go edge=blue  no-go edge=orange  circle=stable  circle+×=unstable</sup>',
            scene=_alpha_scene,
            height=650,
            paper_bgcolor='white',
        )
        _save_plotly(fig_comb, out_dir / 'bifur_3d_alpha_combined.html')

    # ── matplotlib: PC1 of FPs vs alpha per phase ────────────────────────────
    from matplotlib.lines import Line2D as _L2D
    _DASHES = [(None, None), (6, 2), (2, 2), (6, 2, 2, 2), (1, 1)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    n_phases = len(phases)
    for ph_idx, (key, alpha_results) in enumerate(sweep_results.items()):
        col = cmap_phase(ph_idx / max(n_phases - 1, 1))
        dash = _DASHES[ph_idx % len(_DASHES)]
        for cond_idx, cond in enumerate(['go', 'nogo']):
            ax = axes[cond_idx]
            s_pc1, s_alpha = [], []
            u_pc1, u_alpha = [], []
            for a, cond_res in zip(alphas, alpha_results):
                if cond not in cond_res or len(cond_res[cond]['h']) == 0:
                    continue
                fp = cond_res[cond]; s = fp['is_stable']
                if s.any():
                    s_pc1.append(pca.transform(fp['h'][s])[:, 0].mean())
                    s_alpha.append(a)
                if (~s).any():
                    for v in pca.transform(fp['h'][~s])[:, 0]:
                        u_pc1.append(v); u_alpha.append(a)
            if s_alpha:
                ln, = ax.plot(s_alpha, s_pc1, color=col, lw=2,
                              label=key, marker='o', ms=4)
                if dash[0] is not None:
                    ln.set_dashes(dash)
            if u_alpha:
                ax.scatter(u_alpha, u_pc1, color=col, s=40, marker='x',
                           linewidths=1.5, zorder=3)

    _stab   = _L2D([], [], color='k', marker='o', ms=5, lw=1.5, label='stable')
    _unstab = _L2D([], [], color='k', marker='x', ms=6, lw=0,
                   markeredgewidth=1.5, label='unstable')
    for ax, title in zip(axes, ['Go', 'No-Go']):
        ax.set_xlabel(f'α  ({task_a}  →  {task_b})')
        ax.set_ylabel('PC1 of FPs')
        ax.set_title(f'{title}')
        phase_leg = ax.legend(fontsize=8, ncol=2, loc='best')
        ax.add_artist(phase_leg)
        ax.legend(handles=[_stab, _unstab], fontsize=8, loc='upper right')
        ax.set_xlim(0, 1)
    fig.suptitle(f'Bifurcation diagram  |  {blend_label}', fontsize=12)
    plt.tight_layout()
    out = out_dir / 'bifur_1d.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # ── matplotlib: FP count vs alpha ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ph_idx, (key, alpha_results) in enumerate(sweep_results.items()):
        col = cmap_phase(ph_idx / max(n_phases - 1, 1))
        for cond_idx, cond in enumerate(['go', 'nogo']):
            ax = axes[cond_idx]
            counts = []
            for cond_res in alpha_results:
                if cond not in cond_res or len(cond_res[cond]['h']) == 0:
                    counts.append(0)
                else:
                    counts.append(int(cond_res[cond]['is_stable'].sum()))
            ax.plot(alphas, counts, color=col, lw=2, label=key, marker='o', ms=4)

    for ax, title in zip(axes, ['Go', 'No-Go']):
        ax.set_xlabel(f'α  ({task_a}  →  {task_b})')
        ax.set_ylabel('# stable fixed points')
        ax.set_title(f'{title}')
        ax.legend(fontsize=8, ncol=2, loc='best')
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
    fig.suptitle(f'Fixed-point count vs α  |  {blend_label}', fontsize=12)
    plt.tight_layout()
    out = out_dir / 'bifur_count.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    print(f'\nDone: {ckpt_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

_DEFAULT_PHASES = [
    ('fixation', None),
    ('pulse_on', 1), ('pulse_on', 2), ('pulse_on', 3), ('pulse_on', 4),
    ('pulse_on', 5), ('pulse_on', 6), ('pulse_on', 7), ('pulse_on', 8),
    ('stim_off', None),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ckpt_path', type=Path,
                   help='Path to checkpoint.pkl')
    p.add_argument('--tasks', nargs=2, metavar=('TASK_A', 'TASK_B'), required=True,
                   help='Two task names: TASK_A (alpha=0) blends to TASK_B (alpha=1), '
                        'e.g. --tasks pure harmonic')
    _default_params = Path(__file__).parent / 'bifur_params.json'
    p.add_argument('--params', type=Path, default=_default_params,
                   help='Bifurcation parameter JSON (default: bifur_params.json next to this script); CLI flags override')
    p.add_argument('--n-alpha',    type=int,   default=5,   help='# alpha steps (default: 21)')
    p.add_argument('--n-trials',   type=int,   default=128,  help='Batch size for test trials (default: 128)')
    p.add_argument('--n-iters',    type=int,   default=8000, help='Adam steps for FP optimisation (default: 8000)')
    p.add_argument('--lr',         type=float, default=1e-2, help='Adam learning rate (default: 0.01)')
    p.add_argument('--tol',        type=float, default=1e-6, help='q-value convergence threshold (default: 1e-6)')
    p.add_argument('--unique-tol', type=float, default=1e-2, help='L2 deduplication distance (default: 0.01)')
    p.add_argument('--show',       action='store_true',      help='Open Plotly plots in browser')

    # Two-pass parse: load JSON defaults first, then let CLI flags override
    pre, _ = p.parse_known_args()
    phases_from_json = None
    params_path = pre.params
    if params_path is not None and Path(params_path).exists():
        with open(params_path) as f:
            loaded = json.load(f)
        _key_map = {
            'n_alpha':    'n_alpha',
            'n_trials':   'n_trials',
            'n_iters':    'n_iters',
            'lr':         'lr',
            'tol':        'tol',
            'unique_tol': 'unique_tol',
        }
        p.set_defaults(**{dest: loaded[src]
                          for src, dest in _key_map.items() if src in loaded})
        if 'phases' in loaded:
            phases_from_json = [(ph, pn) for ph, pn in loaded['phases']]
        print(f'Loaded bifurcation params from: {params_path}')

    args = p.parse_args()
    args.phases = phases_from_json or _DEFAULT_PHASES

    analyse(args.ckpt_path, args)


if __name__ == '__main__':
    main()
