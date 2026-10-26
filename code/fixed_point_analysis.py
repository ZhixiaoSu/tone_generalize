"""
Fixed-point analysis script.

Runs the full analysis pipeline for one or more checkpoints and saves all
outputs (PNG + HTML) next to each checkpoint in a checkpoint_<step>/ folder.

Usage:
    python fixed_point_analysis.py <ckpt.pkl> [<ckpt2.pkl> ...] [options]

Examples:
    python fixed_point_analysis.py $TONE_RESULTS_DIR/pure/.../checkpoint.pkl
    python fixed_point_analysis.py results/*/checkpoint.pkl --n-iters 12000
    python fixed_point_analysis.py results/pure/.../checkpoint.pkl --show
"""
import argparse
import json
import re
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def _make_scene(ax):
    return dict(xaxis_title=ax[0], yaxis_title=ax[1], zaxis_title=ax[2],
                aspectmode='data')


def _fp3d(fig, x, y, z, fill_col, ec, sz, name, stable,
          showlegend=True, opacity=1.0, row=None, col=None):
    """Add a Plotly 3D fixed-point marker using the double-circle trick.

    Plotly WebGL does not render marker.line.width in 3D — only the legend swatch.
    Workaround: add a slightly larger background circle filled with the edge colour,
    then paint the foreground circle (condition/pulse colour) on top.
    Unstable FPs get an additional black × overlay.
    """
    kw = dict(row=row, col=col) if row is not None else {}
    # 1) background circle → edge colour, +5 pixels larger, hidden from legend
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(color=ec, size=sz + 5, symbol='circle', opacity=opacity),
        showlegend=False,
    ), **kw)
    # 2) foreground circle → fill colour
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(color=fill_col, size=sz, symbol='circle', opacity=opacity),
        name=name, showlegend=showlegend,
    ), **kw)
    if not stable:
        # 3) black × overlay for unstable FPs
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(color='black', size=max(sz - 3, 3), symbol='x',
                        line=dict(color='black', width=1.5), opacity=opacity),
            showlegend=False,
        ), **kw)


def _compute_flow_field(pca, params, x_frozen, alpha, h_segment, n_grid=6, pad_std=0.5):
    """
    Evaluate the RNN velocity field Δh on a grid around the given trajectory segment.

    h_segment : (n_time, n_rnn) — mean hidden states for the relevant phase window.
                The grid spans the bounding box of h_segment in PCA space, padded
                by pad_std × the global PC standard deviation on each axis.

    Returns
    -------
    grid_pc : (N, 3)  — grid point coordinates in PCA space
    vel_pc  : (N, 3)  — velocity vectors in PCA space
    speed   : (N,)    — L2 norm of vel_pc
    """
    W_in  = np.array(params['W_in'])
    W_rec = np.array(params['W_rec'])
    b_rec = np.array(params['b_rec'])

    seg_pc = pca.transform(h_segment)  # (n_time, 3)
    pad    = np.sqrt(pca.explained_variance_) * pad_std
    axes = [np.linspace(seg_pc[:, i].min() - pad[i],
                        seg_pc[:, i].max() + pad[i], n_grid)
            for i in range(3)]
    g0, g1, g2 = np.meshgrid(*axes, indexing='ij')
    grid_pc = np.stack([g0.ravel(), g1.ravel(), g2.ravel()], axis=1)  # (N, 3)

    h      = pca.inverse_transform(grid_pc).astype(np.float32)
    x      = x_frozen.astype(np.float32)
    pre    = h @ W_rec + x @ W_in + b_rec
    h_next = (1.0 - alpha) * h + alpha * np.maximum(0.0, pre)
    dh     = h_next - h
    vel_pc = dh @ pca.components_.T
    speed  = np.linalg.norm(vel_pc, axis=1)
    return grid_pc, vel_pc, speed


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse(ckpt_path: Path, args) -> None:
    ckpt_path = Path(ckpt_path)
    print(f'\n{"="*70}')
    print(f'Analysing: {ckpt_path}')
    print(f'{"="*70}')

    # ── load config + checkpoint ──────────────────────────────────────────────
    def _load_json(path):
        text = Path(path).read_text(encoding='utf-8')
        # strip trailing commas before ] or } (invalid JSON but easy to produce)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return json.loads(text)

    config = _load_json(ckpt_path.parent / 'config.json')

    hp          = config['training']
    task_params = config['task_params']
    task_list   = config.get('task_list', config.get('tasks', ['pure', 'harmonic', 'wn']))
    train_tasks = config.get('train_task', config.get('task_type', 'pure'))
    if isinstance(train_tasks, str):
        train_tasks = [train_tasks]
    if args.task is not None:
        train_tasks = [t.strip() for t in args.task.split(',')]

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    params = ckpt['params']
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

    print(f'train_tasks={train_tasks}  task_list={task_list}')
    print(f'n_rnn={n_rnn}  alpha={alpha:.3f}  stim_on={stim_on}  stim_off={stim_off}')

    model = LeakyRNN(n_rnn=n_rnn, n_out=1, alpha=alpha,
                     sigma_rec=hp['model']['sigma_rec'],
                     sigma_x=hp['model']['sigma_x'])

    for task_type in train_tasks:
        print('\n' + '=' * 60)
        print(f'Task: {task_type}')

        trial = generate_trials(task_type, task_list, mode='test',
                                batch_size=args.n_trials,
                                rng=np.random.default_rng(1))
        xs = jnp.array(trial.x)
        _, hs_jax = model.apply({'params': params}, xs,
                                 noise_input=False, noise_rec=False,
                                 rngs={'noise': jax.random.PRNGKey(0)})
        hs = np.array(hs_jax)   # (T, batch, n_rnn)

        go_mask   = trial.is_go
        nogo_mask = ~trial.is_go
        print(f'Trials: {go_mask.sum()} Go, {nogo_mask.sum()} No-Go')

        out_dir = ckpt_path.parent / f'checkpoint_{ckpt["step"]}' / task_type
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Output dir: {out_dir}')

        # ── build phases list ─────────────────────────────────────────────────────
        phases = [('fixation', None), ('stim_off', None)]
        for p in range(1, args.n_pulses + 1):
            phases.append(('pulse_on', p))

        # ── find fixed points ─────────────────────────────────────────────────────
        STIM_ON_PHASES = {'stim_on', 'pulse_on'}
        results = {}

        for phase, pulse_num in phases:
            t_step = get_phase_step(phase, stim_on, stim_off, pulse_steps,
                                    pulse_num=pulse_num if pulse_num else 2)
            key = phase_label(phase, pulse_num)
            print(f'\n── {key}  (t={t_step})')

            x_go   = trial.x[t_step, np.where(go_mask)[0][0],   :].copy() if go_mask.any()   else None
            x_nogo = trial.x[t_step, np.where(nogo_mask)[0][0], :].copy() if nogo_mask.any() else None

            if phase in STIM_ON_PHASES:
                n_freq = trial.n_freq
                if x_go   is not None: x_go[n_freq]   = 1.0
                if x_nogo is not None: x_nogo[n_freq] = 1.0

            h_go   = hs[t_step, go_mask,   :]
            h_nogo = hs[t_step, nogo_mask, :]

            cond_results = {}
            for cond, x_frozen, h_inits in [('go', x_go, h_go), ('nogo', x_nogo, h_nogo)]:
                if x_frozen is None:
                    continue
                print(f'  [{cond}] {len(h_inits)} initial states')
                cond_results[cond] = find_fixed_points(
                    params, x_frozen, h_inits, alpha,
                    n_iters=args.n_iters, lr=args.lr,
                    tol=args.tol, unique_tol=args.unique_tol,
                    verbose=True)

            results[key] = dict(conds=cond_results, t_step=t_step,
                                x_go=x_go, x_nogo=x_nogo, h_go=h_go, h_nogo=h_nogo)

        # ── save FP parameters ────────────────────────────────────────────────────
        fp_params = {
            'ckpt_path':   str(ckpt_path),
            'step':        int(ckpt['step']),
            'n_trials':    args.n_trials,
            'n_iters':     args.n_iters,
            'lr':          args.lr,
            'tol':         args.tol,
            'unique_tol':  args.unique_tol,
            'freeze_dur_ms': args.freeze_dur,
            'n_pulses':    args.n_pulses,
            'phases':      [[ph, pn] for ph, pn in phases],
        }
        params_path = out_dir / 'fp_params.json'
        with open(params_path, 'w') as f:
            json.dump(fp_params, f, indent=2)
        print(f'Saved params: {params_path}')

        # ── PCA ───────────────────────────────────────────────────────────────────
        pca = PCA(n_components=3)
        pca.fit(hs.reshape(-1, n_rnn))
        var_exp = pca.explained_variance_ratio_ * 100

        pulse_results = {k: v for k, v in results.items() if k.startswith('pulse_on')}
        other_results = {k: v for k, v in results.items() if not k.startswith('pulse_on')}
        n_pulses_found = len(pulse_results)

        cmap = plt.cm.viridis
        norm = plt.Normalize(1, max(n_pulses_found, 1))

        axis_labels = [f'PC1 ({var_exp[0]:.1f}%)', f'PC2 ({var_exp[1]:.1f}%)', f'PC3 ({var_exp[2]:.1f}%)']
        GO_COL   = _pc('tab:blue')
        NOGO_COL = _pc('tab:orange')

        # ── helpers ───────────────────────────────────────────────────────────────
        COND_MKR     = {'go': '^', 'nogo': '^'}
        STABLE_MKR   = 'o'
        UNSTABLE_MKR = 'x'

        def _draw_traj(ax, a=0.12, lw=0.7):
            for i in np.where(go_mask)[0]:
                p = pca.transform(hs[:, i, :])
                ax.plot(p[:, 0], p[:, 1], p[:, 2], color='tab:blue', alpha=a, lw=lw)
            for i in np.where(nogo_mask)[0]:
                p = pca.transform(hs[:, i, :])
                ax.plot(p[:, 0], p[:, 1], p[:, 2], color='tab:orange', alpha=a, lw=lw)

        def _scatter_fp(ax, px, py, pz, stable, color, edge_color='black'):
            if stable:
                ax.scatter(px, py, pz, s=120, marker=STABLE_MKR,
                           color=color, edgecolors=edge_color, linewidths=0.8, depthshade=False)
            else:
                ax.scatter(px, py, pz, s=120, marker=STABLE_MKR,
                           color=color, edgecolors=edge_color, linewidths=1.2, depthshade=False)
                ax.scatter(px, py, pz, s=80, marker='x', color='black', linewidths=1.5, depthshade=False)

        def _set_labels(ax, title):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', labelpad=4)
            ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', labelpad=4)
            ax.set_zlabel(f'PC3 ({var_exp[2]:.1f}%)', labelpad=4)

        def _save_show_plotly(fig, path):
            if args.show:
                fig.show()
            fig.write_html(str(path))
            print(f'Saved: {path}')

        # ── matplotlib: FP overview ───────────────────────────────────────────────
        n_cols = len(other_results) + (1 if pulse_results else 0)
        fig = plt.figure(figsize=(6 * n_cols, 5))

        for col, (key, res) in enumerate(other_results.items()):
            ax = fig.add_subplot(1, n_cols, col + 1, projection='3d')
            _draw_traj(ax)
            t_step = res['t_step']
            for cond, mask, color in [('go', go_mask, 'tab:blue'), ('nogo', nogo_mask, 'tab:orange')]:
                p = pca.transform(hs[t_step, mask, :])
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=20, color=color,
                           alpha=0.6, marker=COND_MKR[cond], depthshade=False)
            for cond, color in [('go', 'tab:blue'), ('nogo', 'tab:orange')]:
                if cond not in res['conds']: continue
                fp = res['conds'][cond]
                if len(fp['h']) == 0: continue
                p = pca.transform(fp['h'])
                for (px, py, pz, stable) in zip(p[:, 0], p[:, 1], p[:, 2], fp['is_stable']):
                    _scatter_fp(ax, px, py, pz, stable, color,
                                edge_color='tab:blue' if cond == 'go' else 'tab:orange')
            _set_labels(ax, key)

        if pulse_results:
            ax = fig.add_subplot(1, n_cols, n_cols, projection='3d')
            _draw_traj(ax, a=0.06)
            for pidx, (key, res) in enumerate(pulse_results.items()):
                color_p = cmap(norm(pidx + 1))
                t_step  = res['t_step']
                for cond, mask in [('go', go_mask), ('nogo', nogo_mask)]:
                    p = pca.transform(hs[t_step, mask, :])
                    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=20, color=color_p,
                               alpha=0.6, marker=COND_MKR[cond], depthshade=False)
                for cond in ('go', 'nogo'):
                    if cond not in res['conds']: continue
                    fp = res['conds'][cond]
                    if len(fp['h']) == 0: continue
                    p = pca.transform(fp['h'])
                    for (px, py, pz, stable) in zip(p[:, 0], p[:, 1], p[:, 2], fp['is_stable']):
                        _scatter_fp(ax, px, py, pz, stable, color_p,
                                    edge_color='tab:blue' if cond == 'go' else 'tab:orange')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.12, label='Pulse #').set_ticks(range(1, n_pulses_found + 1))
            _set_labels(ax, f'pulse_on 1–{n_pulses_found}')

        fig.legend(handles=[
            mpatches.Patch(color='tab:blue',   label='Go'),
            mpatches.Patch(color='tab:orange', label='No-Go'),
            plt.Line2D([0], [0], marker=STABLE_MKR, color='tab:blue',   linestyle='None', markersize=10, markeredgecolor='tab:blue', label='Stable FP (Go)'),
            plt.Line2D([0], [0], marker=STABLE_MKR, color='tab:orange', linestyle='None', markersize=10, markeredgecolor='tab:orange', label='Stable FP (No-Go)'),
            plt.Line2D([0], [0], marker='x',         color='black',      linestyle='None', markersize=10, markeredgewidth=1.5,    label='Unstable FP (+ ×)'),
        ], loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f'Fixed points — {task_type}  |  step {ckpt["step"]}', fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / 'fixed_points.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_dir}/fixed_points.png')

        # ── Plotly: FP overview per phase ─────────────────────────────────────────
        def _add_mean_traj_traces(fig):
            go_pc   = pca.transform(hs[:, go_mask,   :].mean(axis=1))
            nogo_pc = pca.transform(hs[:, nogo_mask, :].mean(axis=1))
            fig.add_trace(go.Scatter3d(x=go_pc[:,0],   y=go_pc[:,1],   z=go_pc[:,2],   mode='lines',
                                       line=dict(color=GO_COL,   width=3), name='Go mean traj'))
            fig.add_trace(go.Scatter3d(x=nogo_pc[:,0], y=nogo_pc[:,1], z=nogo_pc[:,2], mode='lines',
                                       line=dict(color=NOGO_COL, width=3), name='No-Go mean traj'))

        for key, res in other_results.items():
            fig = go.Figure()
            _add_mean_traj_traces(fig)
            t_step = res['t_step']
            for cond, mask, col in [('go', go_mask, GO_COL), ('nogo', nogo_mask, NOGO_COL)]:
                p = pca.transform(hs[t_step, mask, :])
                fig.add_trace(go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2], mode='markers',
                                           marker=dict(color=col, size=3, opacity=0.7), name=f'{key} {cond}'))
                if cond in res['conds'] and len(res['conds'][cond]['h']) > 0:
                    fp = res['conds'][cond]; pfp = pca.transform(fp['h']); s = fp['is_stable']
                    ec = GO_COL if cond == 'go' else NOGO_COL
                    if s.any():
                        _fp3d(fig, pfp[s,0], pfp[s,1], pfp[s,2], col, ec, 10,
                              f'Stable FP {key} {cond}', stable=True)
                    if (~s).any():
                        _fp3d(fig, pfp[~s,0], pfp[~s,1], pfp[~s,2], col, ec, 7,
                              f'Unstable FP {key} {cond}', stable=False)
            fig.update_layout(title=f'Fixed points — {key}', scene=_make_scene(axis_labels),
                              height=600, paper_bgcolor='white')
            fname = key.replace(' ', '_').replace('/', '-')
            _save_show_plotly(fig, out_dir / f'fp_overview_{fname}_{ckpt["step"]}.html')

        if pulse_results:
            fig = go.Figure()
            _add_mean_traj_traces(fig)
            cmap_vir = plt.cm.viridis; norm_vir = plt.Normalize(1, max(len(pulse_results), 1))
            for pidx, (key, res) in enumerate(pulse_results.items()):
                col_str = _pc(cmap_vir(norm_vir(pidx + 1))); t_step = res['t_step']
                for cond, mask in [('go', go_mask), ('nogo', nogo_mask)]:
                    p = pca.transform(hs[t_step, mask, :])
                    fig.add_trace(go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2], mode='markers',
                                               marker=dict(color=col_str, size=3, opacity=0.6),
                                               name=f'{key} {cond}'))
                    if cond in res['conds'] and len(res['conds'][cond]['h']) > 0:
                        fp = res['conds'][cond]; pfp = pca.transform(fp['h']); s = fp['is_stable']
                        ec = GO_COL if cond == 'go' else NOGO_COL
                        if s.any():
                            _fp3d(fig, pfp[s,0], pfp[s,1], pfp[s,2], col_str, ec, 10,
                                  f'Stable FP {key} {cond}', stable=True)
                        if (~s).any():
                            _fp3d(fig, pfp[~s,0], pfp[~s,1], pfp[~s,2], col_str, ec, 7,
                                  f'Unstable FP {key} {cond}', stable=False)
            fig.update_layout(title=f'Fixed points — pulse_on 1-{len(pulse_results)}',
                              scene=_make_scene(axis_labels), height=600, paper_bgcolor='white')
            _save_show_plotly(fig, out_dir / f'fp_overview_pulses_{ckpt["step"]}.html')

        # ── frozen dynamics setup ─────────────────────────────────────────────────
        n_auto = int(args.freeze_dur / dt)
        W_in  = np.array(params['W_in'])
        W_rec = np.array(params['W_rec'])
        b_rec = np.array(params['b_rec'])

        def run_frozen(h0, x_frozen, n_steps):
            h = h0.astype(np.float32).copy()
            x = x_frozen.astype(np.float32)
            traj = [h.copy()]
            for _ in range(n_steps):
                pre = x @ W_in + h @ W_rec + b_rec
                h   = (1.0 - alpha) * h + alpha * np.maximum(0.0, pre)
                traj.append(h.copy())
            return np.stack(traj)   # (n_steps+1, batch, n_rnn)

        points = [('pulse_on', p) for p in range(1, args.n_pulses + 1)]

        # ── matplotlib: frozen traj (PC1 vs time) ────────────────────────────────
        ODD_COLOR  = 'tab:red'
        EVEN_COLOR = 'tab:blue'
        fig, ax = plt.subplots(figsize=(13, 4))
        xform = ax.get_xaxis_transform()

        for phase, pulse_num in points:
            key = phase_label(phase, pulse_num)
            res = results.get(key)
            if res is None: continue
            color = ODD_COLOR if pulse_num % 2 == 1 else EVEN_COLOR
            t_step      = res['t_step']
            t_offset_ms = (t_step - stim_on) * dt
            t_axis      = t_offset_ms + np.arange(n_auto + 1) * dt
            traj_go   = run_frozen(res['h_go'],   res['x_go'],   n_auto)
            traj_nogo = run_frozen(res['h_nogo'], res['x_nogo'], n_auto)
            pc1_go   = pca.transform(traj_go.reshape(-1, n_rnn))[:, 0].reshape(n_auto + 1, -1)
            pc1_nogo = pca.transform(traj_nogo.reshape(-1, n_rnn))[:, 0].reshape(n_auto + 1, -1)
            for j in range(pc1_go.shape[1]):
                ax.plot(t_axis, pc1_go[:, j],   color=color, alpha=0.03, lw=0.6, ls='-')
            for j in range(pc1_nogo.shape[1]):
                ax.plot(t_axis, pc1_nogo[:, j], color=color, alpha=0.03, lw=0.6, ls='--')
            ax.plot(t_axis, pc1_go.mean(1),   color=color, lw=2.0, ls='-',  alpha=0.9)
            ax.plot(t_axis, pc1_nogo.mean(1), color=color, lw=2.0, ls='--', alpha=0.9)
            ax.axvline(t_offset_ms, color=color, lw=0.8, ls=':', alpha=0.7)
            ax.text(t_offset_ms, 1.02, f'P{pulse_num}', fontsize=8, ha='center', color=color, transform=xform)

        ax.axvspan(0, (stim_off - stim_on) * dt, alpha=0.07, color='gray')
        ax.set_xlabel('Time from stimulus onset (ms)')
        ax.set_ylabel('PC1')
        ax.legend(handles=[
            plt.Line2D([0], [0], color=ODD_COLOR,  lw=1.8, label='Odd pulses (1,3,5,7)'),
            plt.Line2D([0], [0], color=EVEN_COLOR, lw=1.8, label='Even pulses (2,4,6,8)'),
            plt.Line2D([0], [0], color='k', lw=1.8, ls='-',  label='Go'),
            plt.Line2D([0], [0], color='k', lw=1.8, ls='--', label='No-Go'),
            mpatches.Patch(color='gray', alpha=0.2, label='Stimulus window'),
        ], fontsize=9, loc='upper right')
        ax.set_title(f'Autonomous dynamics after freeze — {task_type}  |  step {ckpt["step"]}  |  freeze_dur={args.freeze_dur} ms', fontsize=11)
        plt.tight_layout()
        out = out_dir / f'frozen_traj_{ckpt["step"]}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f'Saved: {out}')

        # ── matplotlib: frozen dynamics 3D (3 panels) ────────────────────────────
        for fig_title, conds_show, fname_base in [
            ('Go + No-Go', ('go', 'nogo'), 'frozen_pca'),
        ]:
            fig = plt.figure(figsize=(21, 7))
            axes3 = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
            titles3 = ['Go + No-Go', 'Go only', 'No-Go only']
            show3   = [('go', 'nogo'), ('go',), ('nogo',)]
            ls_c    = {'go': '-',  'nogo': '--'}
            mkr_c   = {'go': '^', 'nogo': '^'}

            for phase, pulse_num in points:
                key = phase_label(phase, pulse_num)
                res = results.get(key)
                if res is None: continue
                color    = cmap(norm(pulse_num))
                traj_go  = run_frozen(res['h_go'],   res['x_go'],   n_auto)
                traj_nogo= run_frozen(res['h_nogo'], res['x_nogo'], n_auto)
                pc_go    = pca.transform(traj_go.reshape(-1, n_rnn)).reshape(n_auto+1, -1, 3)
                pc_nogo  = pca.transform(traj_nogo.reshape(-1, n_rnn)).reshape(n_auto+1, -1, 3)
                mean_go  = pc_go.mean(1); mean_nogo = pc_nogo.mean(1)
                pc_c     = {'go': pc_go,    'nogo': pc_nogo}
                mean_c   = {'go': mean_go,  'nogo': mean_nogo}
                fp_c = {}
                for cond in ('go', 'nogo'):
                    if cond in res['conds'] and len(res['conds'][cond]['h']) > 0:
                        fp_c[cond]          = pca.transform(res['conds'][cond]['h'])
                        fp_c[cond+'_stable']= res['conds'][cond]['is_stable']

                for ax, conds in zip(axes3, show3):
                    for cond in conds:
                        pc   = pc_c[cond]; mean = mean_c[cond]
                        for j in range(pc.shape[1]):
                            ax.plot(pc[:, j, 0], pc[:, j, 1], pc[:, j, 2], color=color, alpha=0.04, lw=0.6, ls=ls_c[cond])
                        ax.plot(mean[:, 0], mean[:, 1], mean[:, 2], color=color, lw=2.0, ls=ls_c[cond], alpha=0.9)
                        ax.scatter(*mean[0, :3], s=50, color=color, marker=mkr_c[cond],
                                   edgecolors='black', linewidths=0.8, depthshade=False, zorder=5)
                        if cond in fp_c:
                            p = fp_c[cond]; stable = fp_c[cond+'_stable']
                            for (px, py, pz, s) in zip(p[:, 0], p[:, 1], p[:, 2], stable):
                                if s:
                                    ax.scatter(px, py, pz, s=140, marker='o', color=color,
                                               edgecolors='tab:blue' if cond == 'go' else 'tab:orange',
                                               linewidths=1.2, depthshade=False, zorder=6)
                                else:
                                    ax.scatter(px, py, pz, s=140, marker='o', color=color,
                                               edgecolors='tab:blue' if cond == 'go' else 'tab:orange',
                                               linewidths=1.2, depthshade=False, zorder=6)
                                    ax.scatter(px, py, pz, s=100, marker='x', color='black',
                                               linewidths=1.5, depthshade=False, zorder=7)

            for ax, t in zip(axes3, titles3):
                ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', labelpad=4)
                ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', labelpad=4)
                ax.set_zlabel(f'PC3 ({var_exp[2]:.1f}%)', labelpad=4)
                ax.set_title(t, fontsize=11)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
            fig.colorbar(sm, ax=axes3[-1], shrink=0.5, pad=0.12, label='Pulse #').set_ticks(range(1, n_pulses_found + 1))
            axes3[0].legend(handles=[
                plt.Line2D([0], [0], color='k', lw=1.8, ls='-',  label='Go (mean)'),
                plt.Line2D([0], [0], color='k', lw=1.8, ls='--', label='No-Go (mean)'),
                plt.Line2D([0], [0], color='k', marker='^', linestyle='None', markersize=7, markeredgecolor='black', label='Start (Go)'),
                plt.Line2D([0], [0], color='k', marker='^', linestyle='None', markersize=7, markeredgecolor='black', label='Start (No-Go)'),
                plt.Line2D([0], [0], color='tab:blue',   marker='o', linestyle='None', markersize=10, markeredgecolor='tab:blue', label='Stable FP (Go)'),
                plt.Line2D([0], [0], color='tab:orange', marker='o', linestyle='None', markersize=10, markeredgecolor='tab:orange', label='Stable FP (No-Go)'),
                plt.Line2D([0], [0], color='black',      marker='x', linestyle='None', markersize=10, markeredgewidth=1.5,    label='Unstable FP (+ ×)'),
            ], fontsize=8, loc='upper left')
            fig.suptitle(f'Frozen dynamics in PCA space — {task_type}  |  step {ckpt["step"]}  |  freeze_dur={args.freeze_dur} ms', fontsize=11)
            plt.tight_layout()
            out = out_dir / f'frozen_pca_{ckpt["step"]}.png'
            plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
            print(f'Saved: {out}')

        # ── Plotly: frozen dynamics ───────────────────────────────────────────────
        scene_labels = axis_labels
        for panel_title, conds_show, fname_base in [
            ('Go + No-Go', ('go', 'nogo'), 'frozen_pca_all'),
            ('Go only',    ('go',),        'frozen_pca_go'),
            ('No-Go only', ('nogo',),      'frozen_pca_nogo'),
        ]:
            fig = go.Figure()
            for phase, pulse_num in points:
                key = phase_label(phase, pulse_num); res = results.get(key)
                if res is None: continue
                col_str = _pc(cmap(norm(pulse_num)))
                traj_go  = run_frozen(res['h_go'],   res['x_go'],   n_auto)
                traj_nogo= run_frozen(res['h_nogo'], res['x_nogo'], n_auto)
                pc_go    = pca.transform(traj_go.reshape(-1, n_rnn)).reshape(n_auto+1, -1, 3)
                pc_nogo  = pca.transform(traj_nogo.reshape(-1, n_rnn)).reshape(n_auto+1, -1, 3)
                mean_go  = pc_go.mean(1); mean_nogo = pc_nogo.mean(1)
                mean_by  = {'go': mean_go, 'nogo': mean_nogo}
                fp_by = {}
                for cond in ('go', 'nogo'):
                    if cond in res['conds'] and len(res['conds'][cond]['h']) > 0:
                        fp_by[cond]          = pca.transform(res['conds'][cond]['h'])
                        fp_by[cond+'_stable']= res['conds'][cond]['is_stable']
                for cond in conds_show:
                    mean = mean_by[cond]; lbl = f'P{pulse_num} {cond}'
                    fig.add_trace(go.Scatter3d(x=mean[:,0], y=mean[:,1], z=mean[:,2], mode='lines',
                                               line=dict(color=col_str, width=6), opacity=0.9, name=f'{lbl} mean'))
                    fig.add_trace(go.Scatter3d(x=mean[0:1,0], y=mean[0:1,1], z=mean[0:1,2], mode='markers',
                                               marker=dict(color=col_str, size=8), name=f'{lbl} start'))
                    if cond in fp_by:
                        pfp = fp_by[cond]; s = fp_by[cond+'_stable']
                        ec = GO_COL if cond == 'go' else NOGO_COL
                        if s.any():
                            _fp3d(fig, pfp[s,0], pfp[s,1], pfp[s,2], col_str, ec, 9,
                                  f'Stable FP {lbl}', stable=True)
                        if (~s).any():
                            _fp3d(fig, pfp[~s,0], pfp[~s,1], pfp[~s,2], col_str, ec, 6,
                                  f'Unstable FP {lbl}', stable=False)
            fig.update_layout(title=f'Frozen dynamics — {panel_title}', scene=_make_scene(scene_labels),
                              height=600, paper_bgcolor='white')
            _save_show_plotly(fig, out_dir / f'{fname_base}_{ckpt["step"]}.html')

        # ── matplotlib: FPs PC1 vs PC2 vs pulse# ─────────────────────────────────
        cmap_fp = plt.cm.viridis; norm_fp = plt.Normalize(1, args.n_pulses)
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection='3d')
        for phase, pulse_num in points:
            key = phase_label(phase, pulse_num); res = results.get(key)
            if res is None: continue
            color = cmap_fp(norm_fp(pulse_num))
            for cond, mkr in COND_MKR.items():
                if cond not in res['conds'] or len(res['conds'][cond]['h']) == 0: continue
                fps = pca.transform(res['conds'][cond]['h'])
                stable = res['conds'][cond]['is_stable']
                z = np.full(len(fps), pulse_num)
                if stable.any():
                    ax.scatter(fps[stable, 0], fps[stable, 1], z[stable], s=120, marker=mkr, color=color,
                               edgecolors='tab:blue' if cond == 'go' else 'tab:orange',
                               linewidths=0.9, depthshade=False)
                if (~stable).any():
                    ax.scatter(fps[~stable, 0], fps[~stable, 1], z[~stable], s=120, marker=mkr,
                               color=color, edgecolors='tab:blue' if cond == 'go' else 'tab:orange',
                               linewidths=0.9, depthshade=False)
                    ax.scatter(fps[~stable, 0], fps[~stable, 1], z[~stable], s=80, marker='x',
                               color='black', linewidths=1.5, depthshade=False)
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', labelpad=4)
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', labelpad=4)
        ax.set_zlabel('Pulse #', labelpad=4)
        ax.set_zticks(range(1, args.n_pulses + 1))
        sm = plt.cm.ScalarMappable(cmap=cmap_fp, norm=norm_fp); sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.12, label='Pulse #').set_ticks(range(1, args.n_pulses + 1))
        ax.legend(handles=[
            plt.Line2D([0], [0], color='k', marker='o', linestyle='None', markersize=8, markeredgecolor='tab:blue',   label='Go stable'),
            plt.Line2D([0], [0], color='k', marker='^', linestyle='None', markersize=8, markeredgecolor='tab:orange', label='No-Go stable'),
            plt.Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=8, markeredgewidth=1.5, label='Unstable FP (+ ×)'),
        ], fontsize=9, loc='upper left')
        ax.set_title(f'Fixed points across pulses — {task_type}  |  step {ckpt["step"]}', fontsize=11)
        plt.tight_layout()
        out = out_dir / f'fps_pulse_z_{ckpt["step"]}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f'Saved: {out}')

        # ── Plotly: FPs PC1 vs PC2 vs pulse# ─────────────────────────────────────
        fig = go.Figure()
        for phase, pulse_num in points:
            key = phase_label(phase, pulse_num); res = results.get(key)
            if res is None: continue
            col_str = _pc(cmap_fp(norm_fp(pulse_num)))
            for cond, sz_s, sz_u in [('go', 8, 5), ('nogo', 5, 3)]:
                if cond not in res['conds'] or len(res['conds'][cond]['h']) == 0: continue
                fps = pca.transform(res['conds'][cond]['h'])
                s   = res['conds'][cond]['is_stable']
                z   = np.full(len(fps), float(pulse_num))
                ec = GO_COL if cond == 'go' else NOGO_COL
                if s.any():
                    _fp3d(fig, fps[s,0], fps[s,1], z[s], col_str, ec, sz_s,
                          f'Stable {cond} P{pulse_num}', stable=True)
                if (~s).any():
                    _fp3d(fig, fps[~s,0], fps[~s,1], z[~s], col_str, ec, sz_u,
                          f'Unstable {cond} P{pulse_num}', stable=False)
        fig.update_layout(title='Fixed points: PC1 vs PC2 vs pulse#',
                          scene=dict(xaxis_title=f'PC1 ({var_exp[0]:.1f}%)', yaxis_title=f'PC2 ({var_exp[1]:.1f}%)',
                                     zaxis_title='Pulse #', aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.7)),
                          height=600, paper_bgcolor='white')
        _save_show_plotly(fig, out_dir / f'fps_pulse_z_{ckpt["step"]}.html')

        # ── silent-input dynamics ─────────────────────────────────────────────────
        n_freq_chan = trial.n_freq

        def make_silent_x(x_with_stim, pulse_num):
            x = x_with_stim.copy()
            x[:n_freq_chan] = 0.0
            x[n_freq_chan]  = 1.0 if pulse_num == 1 else 0.0
            return x

        # matplotlib: PC1 vs time
        fig, ax = plt.subplots(figsize=(13, 4))
        xform = ax.get_xaxis_transform()
        for phase, pulse_num in points:
            key = phase_label(phase, pulse_num); res = results.get(key)
            if res is None: continue
            color = ODD_COLOR if pulse_num % 2 == 1 else EVEN_COLOR
            t_step      = res['t_step']
            t_offset_ms = (t_step - stim_on) * dt
            t_axis      = t_offset_ms + np.arange(n_auto + 1) * dt
            x_go   = make_silent_x(res['x_go'],   pulse_num)
            x_nogo = make_silent_x(res['x_nogo'], pulse_num)
            traj_go  = run_frozen(res['h_go'],   x_go,   n_auto)
            traj_nogo= run_frozen(res['h_nogo'], x_nogo, n_auto)
            pc1_go   = pca.transform(traj_go.reshape(-1, n_rnn))[:, 0].reshape(n_auto+1, -1)
            pc1_nogo = pca.transform(traj_nogo.reshape(-1, n_rnn))[:, 0].reshape(n_auto+1, -1)
            for j in range(pc1_go.shape[1]):
                ax.plot(t_axis, pc1_go[:, j],   color=color, alpha=0.03, lw=0.6, ls='-')
            for j in range(pc1_nogo.shape[1]):
                ax.plot(t_axis, pc1_nogo[:, j], color=color, alpha=0.03, lw=0.6, ls='--')
            ax.plot(t_axis, pc1_go.mean(1),   color=color, lw=2.0, ls='-',  alpha=0.9)
            ax.plot(t_axis, pc1_nogo.mean(1), color=color, lw=2.0, ls='--', alpha=0.9)
            ax.axvline(t_offset_ms, color=color, lw=0.8, ls=':', alpha=0.7)
            ax.text(t_offset_ms, 1.02, f'P{pulse_num}', fontsize=8, ha='center', color=color, transform=xform)
        ax.axvspan(0, (stim_off - stim_on) * dt, alpha=0.07, color='gray')
        ax.set_xlabel('Time from stimulus onset (ms)'); ax.set_ylabel('PC1')
        ax.legend(handles=[
            plt.Line2D([0], [0], color=ODD_COLOR,  lw=1.8, label='Odd pulses — silent'),
            plt.Line2D([0], [0], color=EVEN_COLOR, lw=1.8, label='Even pulses — silent'),
            plt.Line2D([0], [0], color='k', lw=1.8, ls='-',  label='Go'),
            plt.Line2D([0], [0], color='k', lw=1.8, ls='--', label='No-Go'),
            mpatches.Patch(color='gray', alpha=0.2, label='Original stim window'),
        ], fontsize=9, loc='upper right')
        ax.set_title(f'Autonomous dynamics — SILENT input  |  {task_type}  step {ckpt["step"]}  |  freeze_dur={args.freeze_dur} ms', fontsize=11)
        plt.tight_layout()
        out = out_dir / f'frozen_silent_traj_{ckpt["step"]}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f'Saved: {out}')

        # matplotlib: silent 3D
        fig = plt.figure(figsize=(21, 7))
        axes_s = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
        for phase, pulse_num in points:
            key = phase_label(phase, pulse_num); res = results.get(key)
            if res is None: continue
            color  = cmap(norm(pulse_num))
            x_go   = make_silent_x(res['x_go'],   pulse_num)
            x_nogo = make_silent_x(res['x_nogo'], pulse_num)
            traj_go  = run_frozen(res['h_go'],   x_go,   n_auto)
            traj_nogo= run_frozen(res['h_nogo'], x_nogo, n_auto)
            pc_go    = pca.transform(traj_go.reshape(-1, n_rnn)).reshape(n_auto+1, -1, 3)
            pc_nogo  = pca.transform(traj_nogo.reshape(-1, n_rnn)).reshape(n_auto+1, -1, 3)
            mean_go  = pc_go.mean(1); mean_nogo = pc_nogo.mean(1)
            pc_cs    = {'go': pc_go, 'nogo': pc_nogo}
            mean_cs  = {'go': mean_go, 'nogo': mean_nogo}
            fp_cs = {}
            for cond in ('go', 'nogo'):
                if cond in res['conds'] and len(res['conds'][cond]['h']) > 0:
                    fp_cs[cond]          = pca.transform(res['conds'][cond]['h'])
                    fp_cs[cond+'_stable']= res['conds'][cond]['is_stable']
            for ax, conds in zip(axes_s, [('go', 'nogo'), ('go',), ('nogo',)]):
                for cond in conds:
                    pc = pc_cs[cond]; mean = mean_cs[cond]; ls = ls_c[cond]; mkr = mkr_c[cond]
                    for j in range(pc.shape[1]):
                        ax.plot(pc[:, j, 0], pc[:, j, 1], pc[:, j, 2], color=color, alpha=0.04, lw=0.6, ls=ls)
                    ax.plot(mean[:, 0], mean[:, 1], mean[:, 2], color=color, lw=2.0, ls=ls, alpha=0.9)
                    ax.scatter(*mean[0, :3], s=50, color=color, marker=mkr,
                               edgecolors='black', linewidths=0.8, depthshade=False, zorder=5)
                    if cond in fp_cs:
                        p = fp_cs[cond]; stable = fp_cs[cond+'_stable']
                        for (px, py, pz, s) in zip(p[:, 0], p[:, 1], p[:, 2], stable):
                            if s:
                                ax.scatter(px, py, pz, s=140, marker='o', color=color,
                                           edgecolors='tab:blue' if cond == 'go' else 'tab:orange',
                                           linewidths=1.2, depthshade=False, zorder=6)
                            else:
                                ax.scatter(px, py, pz, s=140, marker='o', color=color,
                                           edgecolors='tab:blue' if cond == 'go' else 'tab:orange',
                                           linewidths=1.2, depthshade=False, zorder=6)
                                ax.scatter(px, py, pz, s=100, marker='x', color='black',
                                           linewidths=1.5, depthshade=False, zorder=7)
        for ax, t in zip(axes_s, ['Go + No-Go', 'Go only', 'No-Go only']):
            ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', labelpad=4)
            ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', labelpad=4)
            ax.set_zlabel(f'PC3 ({var_exp[2]:.1f}%)', labelpad=4)
            ax.set_title(t, fontsize=11)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=axes_s[-1], shrink=0.5, pad=0.12, label='Pulse #').set_ticks(range(1, n_pulses_found + 1))
        axes_s[0].legend(handles=[
            plt.Line2D([0], [0], color='k', lw=1.8, ls='-',  label='Go (mean)'),
            plt.Line2D([0], [0], color='k', lw=1.8, ls='--', label='No-Go (mean)'),
            plt.Line2D([0], [0], color='k', marker='^', linestyle='None', markersize=7, markeredgecolor='black', label='Start (Go)'),
            plt.Line2D([0], [0], color='k', marker='^', linestyle='None', markersize=7, markeredgecolor='black', label='Start (No-Go)'),
            plt.Line2D([0], [0], color='tab:blue',   marker='o', linestyle='None', markersize=10, markeredgecolor='tab:blue', label='Stable FP (Go)'),
            plt.Line2D([0], [0], color='tab:orange', marker='o', linestyle='None', markersize=10, markeredgecolor='tab:orange', label='Stable FP (No-Go)'),
            plt.Line2D([0], [0], color='black',      marker='x', linestyle='None', markersize=10, markeredgewidth=1.5,    label='Unstable FP (+ ×)'),
        ], fontsize=8, loc='upper left')
        fig.suptitle(f'Silent-input dynamics in PCA space — {task_type}  |  step {ckpt["step"]}  |  freeze_dur={args.freeze_dur} ms', fontsize=11)
        plt.tight_layout()
        out = out_dir / f'frozen_silent_pca_{ckpt["step"]}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f'Saved: {out}')

        # Plotly: silent dynamics
        for panel_title, conds_show, fname_base in [
            ('Go + No-Go', ('go', 'nogo'), 'frozen_silent_pca_all'),
            ('Go only',    ('go',),        'frozen_silent_pca_go'),
            ('No-Go only', ('nogo',),      'frozen_silent_pca_nogo'),
        ]:
            fig = go.Figure()
            for phase, pulse_num in points:
                key = phase_label(phase, pulse_num); res = results.get(key)
                if res is None: continue
                col_rgba = cmap(norm(pulse_num)); col_str = _pc(col_rgba)
                x_go   = make_silent_x(res['x_go'],   pulse_num)
                x_nogo = make_silent_x(res['x_nogo'], pulse_num)
                traj_go  = run_frozen(res['h_go'],   x_go,   n_auto)
                traj_nogo= run_frozen(res['h_nogo'], x_nogo, n_auto)
                pc_go    = pca.transform(traj_go.reshape(-1, n_rnn)).reshape(n_auto+1, -1, 3)
                pc_nogo  = pca.transform(traj_nogo.reshape(-1, n_rnn)).reshape(n_auto+1, -1, 3)
                mean_go  = pc_go.mean(1); mean_nogo = pc_nogo.mean(1)
                mean_by  = {'go': mean_go, 'nogo': mean_nogo}
                fp_by = {}
                for cond in ('go', 'nogo'):
                    if cond in res['conds'] and len(res['conds'][cond]['h']) > 0:
                        fp_by[cond]          = pca.transform(res['conds'][cond]['h'])
                        fp_by[cond+'_stable']= res['conds'][cond]['is_stable']
                for cond in conds_show:
                    mean = mean_by[cond]; lbl = f'P{pulse_num} {cond}'
                    fig.add_trace(go.Scatter3d(x=mean[:,0], y=mean[:,1], z=mean[:,2], mode='lines',
                                               line=dict(color=col_str, width=6), opacity=0.9, name=f'{lbl} mean'))
                    fig.add_trace(go.Scatter3d(x=mean[0:1,0], y=mean[0:1,1], z=mean[0:1,2], mode='markers',
                                               marker=dict(color=col_str, size=10), name=f'{lbl} start'))
                    if cond in fp_by:
                        pfp = fp_by[cond]; s = fp_by[cond+'_stable']
                        ec = GO_COL if cond == 'go' else NOGO_COL
                        if s.any():
                            _fp3d(fig, pfp[s,0], pfp[s,1], pfp[s,2], col_str, ec, 9,
                                  f'Stable FP {lbl}', stable=True)
                        if (~s).any():
                            _fp3d(fig, pfp[~s,0], pfp[~s,1], pfp[~s,2], col_str, ec, 6,
                                  f'Unstable FP {lbl}', stable=False)
            fig.update_layout(title=f'Silent-input dynamics — {panel_title}', scene=_make_scene(scene_labels),
                              height=600, paper_bgcolor='white')
            _save_show_plotly(fig, out_dir / f'{fname_base}_{ckpt["step"]}.html')

        # ── Plotly: flow field in PCA space per phase ─────────────────────────────
        # Cones show Δh per step on a regular grid; colour = speed.
        # Stable FPs appear as sinks (converging cones), unstable as sources/saddles.

        def _make_flow_fig(key, res):
            # Determine the phase time window so the flow field is sampled
            # only around the trajectory segment relevant to this phase.
            T = hs.shape[0]
            if key.startswith('pulse_on'):
                pnum    = int(re.search(r'\d+', key).group())
                t_start = stim_on + (pnum - 1) * pulse_steps
                t_end   = min(stim_on + pnum * pulse_steps, stim_off)
            elif 'stim_off' in key or 'stim off' in key:
                t_start, t_end = stim_off, T
            else:  # fixation
                t_start, t_end = 0, stim_on
            t_end = max(t_end, t_start + 1)  # guard against zero-length windows

            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=['Go', 'No-Go'],
            )
            for col_idx, (cond, mask, x_frozen, col_traj) in enumerate(
                [('go',   go_mask,   res['x_go'],   GO_COL),
                 ('nogo', nogo_mask, res['x_nogo'], NOGO_COL)], start=1):

                h_seg = hs[t_start:t_end, mask, :].mean(axis=1)  # (n_time, n_rnn)
                grid_pc, vel_pc, speed = _compute_flow_field(
                    pca, params, x_frozen, alpha, h_seg, n_grid=6)

                # scale cones so median-speed arrow ≈ 25 % of the PC1 data range
                data_range = float(np.sqrt(pca.explained_variance_[0])) * 5.0
                pos_speed  = speed[speed > 0]
                med_speed  = float(np.median(pos_speed)) if len(pos_speed) else 1.0
                sizeref    = 0.0025 * data_range / med_speed

                fig.add_trace(go.Cone(
                    x=grid_pc[:, 0].tolist(), y=grid_pc[:, 1].tolist(),
                    z=grid_pc[:, 2].tolist(),
                    u=vel_pc[:, 0].tolist(),  v=vel_pc[:, 1].tolist(),
                    w=vel_pc[:, 2].tolist(),
                    colorscale='Viridis', cmin=0, cmax=float(speed.max()),
                    sizemode='scaled', sizeref=sizeref,
                    opacity=0.5, showscale=(col_idx == 1),
                    colorbar=dict(title='Speed', len=0.5, x=0.46) if col_idx == 1 else {},
                    name=f'Flow ({cond})',
                ), row=1, col=col_idx)

                # mean trajectory
                mean_pc = pca.transform(hs[:, mask, :].mean(axis=1))
                fig.add_trace(go.Scatter3d(
                    x=mean_pc[:, 0], y=mean_pc[:, 1], z=mean_pc[:, 2], mode='lines',
                    line=dict(color=col_traj, width=6), opacity=0.9,
                    name=f'{cond} mean traj',
                ), row=1, col=col_idx)
                # start / end markers and phase timepoint marker
                for pt, sym, lbl in [(0, 'circle', 'start'), (-1, 'square', 'end')]:
                    fig.add_trace(go.Scatter3d(
                        x=[mean_pc[pt, 0]], y=[mean_pc[pt, 1]], z=[mean_pc[pt, 2]],
                        mode='markers',
                        marker=dict(color=col_traj, size=8, symbol=sym,
                                    line=dict(color='white', width=1)),
                        name=f'{cond} traj {lbl}', showlegend=True,
                    ), row=1, col=col_idx)
                # diamond marker at the phase timepoint (e.g. start of pulse 1)
                t_step = res['t_step']
                fig.add_trace(go.Scatter3d(
                    x=[mean_pc[t_step, 0]], y=[mean_pc[t_step, 1]], z=[mean_pc[t_step, 2]],
                    mode='markers+text',
                    marker=dict(color='yellow', size=10, symbol='diamond',
                                line=dict(color='black', width=1)),
                    text=[key], textposition='top center',
                    textfont=dict(size=11, color='black'),
                    name=f'{cond} @ {key}', showlegend=True,
                ), row=1, col=col_idx)

                # fixed points
                if cond in res['conds'] and len(res['conds'][cond]['h']) > 0:
                    fp = res['conds'][cond]
                    p  = pca.transform(fp['h'])
                    s  = fp['is_stable']
                    ec = GO_COL if cond == 'go' else NOGO_COL
                    if s.any():
                        _fp3d(fig, p[s, 0], p[s, 1], p[s, 2], col_traj, ec, 14,
                              f'Stable FP ({cond})', stable=True, row=1, col=col_idx)
                    if (~s).any():
                        _fp3d(fig, p[~s, 0], p[~s, 1], p[~s, 2], col_traj, ec, 8,
                              f'Unstable FP ({cond})', stable=False, row=1, col=col_idx)

            scene_kw = dict(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1],
                            zaxis_title=axis_labels[2], aspectmode='data')
            fig.update_layout(
                title=f'Flow field in PCA space — {key}<br>'
                      f'<sup>cones = Δh/step · colour = speed (viridis) · '
                      f'◉ stable FP · × unstable FP</sup>',
                scene=scene_kw, scene2=scene_kw,
                height=660, paper_bgcolor='white',
            )
            return fig

        for key, res in {**other_results, **pulse_results}.items():
            fig = _make_flow_fig(key, res)
            fname = key.replace(' ', '_').replace('/', '-')
            _save_show_plotly(fig, out_dir / f'flow_field_{fname}_{ckpt["step"]}.html')

    print(f'\nDone: {ckpt_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+', type=Path,
                   help='Path(s) to checkpoint.pkl file(s)')
    _default_params = Path(__file__).parent / 'fix_point_params.json'
    p.add_argument('--params',     type=Path,  default=_default_params,
                   help=f'FP parameter JSON file (default: fix_point_params.json next to this script); CLI flags override individual values')
    p.add_argument('--n-trials',   type=int,   default=256,   help='Batch size for test trials (default: 256)')
    p.add_argument('--n-iters',    type=int,   default=24000, help='Adam steps for FP optimisation (default: 24000)')
    p.add_argument('--lr',         type=float, default=1e-2,  help='Adam learning rate (default: 0.01)')
    p.add_argument('--tol',        type=float, default=1e-6,  help='q-value convergence threshold (default: 1e-6)')
    p.add_argument('--unique-tol', type=float, default=1e-3,  help='L2 deduplication distance (default: 1e-3)')
    p.add_argument('--freeze-dur', type=int,   default=2000,  help='Frozen dynamics duration in ms (default: 2000)')
    p.add_argument('--n-pulses',   type=int,   default=8,     help='Number of pulses to analyse (default: 8)')
    p.add_argument('--task',       type=str,   default=None,  help='Override task(s) to analyse, comma-separated (e.g. pure,harmonic); default: use training task from config')
    p.add_argument('--show',       action='store_true',       help='Open Plotly plots in browser')

    # First pass: pick up --params only, then inject its values as new defaults
    # so that any explicit CLI flag still wins.
    pre, _ = p.parse_known_args()
    if pre.params is not None:
        with open(pre.params) as f:
            loaded = json.load(f)
        # map JSON keys → argparse dest names
        _key_map = {
            'n_trials':    'n_trials',
            'n_iters':     'n_iters',
            'lr':          'lr',
            'tol':         'tol',
            'unique_tol':  'unique_tol',
            'freeze_dur_ms': 'freeze_dur',
            'n_pulses':    'n_pulses',
        }
        p.set_defaults(**{dest: loaded[src] for src, dest in _key_map.items() if src in loaded})
        print(f'Loaded FP params from: {pre.params}')

    args = p.parse_args()

    for ckpt_path in args.checkpoints:
        analyse(ckpt_path, args)


if __name__ == '__main__':
    main()