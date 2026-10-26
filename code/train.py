"""
Training entry point for the auditory go/no-go task.

Usage
-----
    from train import train
    state, log = train(['pure', 'harmonic', 'wn'], save_dir='results/')

    # or from the command line:
    python train.py --tasks pure harmonic wn --save_dir results/
"""

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task import generate_trials, get_stimulus_vectors, load_params as load_task_params
from network import LeakyRNN, create_train_state
from paths import results_dir

DEFAULT_TRAINING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training.json')


# ── Parameter loading ──────────────────────────────────────────────────────────

def load_training_params(path=None):
    with open(path or DEFAULT_TRAINING) as f:
        return json.load(f)


# ── JIT-compiled train step ────────────────────────────────────────────────────

@jax.jit
def _train_step(state, xs, ys_target, mask, noise_key):
    """Single gradient-descent step. Returns (new_state, loss).

    mask : (tdim, 1) float array — 1.0 for pre-stim/grace, 5.0 for response period.
    """
    def loss_fn(params):
        ys_pred, _ = state.apply_fn(
            {'params': params}, xs,
            noise_input=True, noise_rec=True,
            rngs={'noise': noise_key},
        )
        return jnp.mean(mask * (ys_pred - ys_target) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


# ── Transfer learning helpers ─────────────────────────────────────────────────

def _make_grad_mask(params, trainable, n_freq):
    """
    Return a pytree of float masks with the same structure as params.
    1.0 = gradient passes through, 0.0 = frozen.

    trainable options
    -----------------
    'all'      : train every parameter
    'context'  : only W_in rows [n_freq+1:]  (3 context one-hot channels)
    'sensory'  : only W_in rows [:n_freq]    (frequency input channels)
    list       : any mix of the above strings plus top-level param names,
                 e.g. ['context', 'W_rec'] or ['sensory', 'b_rec']
    """
    if trainable == 'all':
        return jax.tree_util.tree_map(jnp.ones_like, params)

    # Normalise to list so both 'context' and ['context'] work identically
    items = [trainable] if isinstance(trainable, str) else list(trainable)

    mask     = jax.tree_util.tree_map(jnp.zeros_like, params)
    win_mask = jnp.zeros_like(params['W_in'])

    for item in items:
        if item == 'context':
            win_mask = win_mask.at[n_freq + 1:].set(1.0)
        elif item == 'sensory':
            win_mask = win_mask.at[:n_freq].set(1.0)
        elif item in params:
            mask = dict(mask, **{item: jnp.ones_like(params[item])})
        else:
            raise ValueError(
                f"Unknown trainable item '{item}'. "
                f"Use 'all', 'context', 'sensory', or a param name: {list(params)}")

    if jnp.any(win_mask > 0):
        mask = dict(mask, W_in=win_mask)

    return mask


def _make_transfer_train_step(grad_mask):
    """Return a JIT-compiled train step that zeroes gradients for frozen params."""
    @jax.jit
    def _step(state, xs, ys_target, mask, noise_key):
        def loss_fn(params):
            ys_pred, _ = state.apply_fn(
                {'params': params}, xs,
                noise_input=True, noise_rec=True,
                rngs={'noise': noise_key},
            )
            return jnp.mean(mask * (ys_pred - ys_target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        masked_grads = jax.tree_util.tree_map(
            lambda g, m: g * m, grads, grad_mask)
        return state.apply_gradients(grads=masked_grads), loss

    return _step


def transfer_train(task_type, task_list, init_params, trainable='all',
                   save_dir=None, training_params_path=None,
                   task_params_path=None, seed=0):
    """
    Fine-tune a pre-trained LeakyRNN on a new task with optional weight freezing.

    Parameters
    ----------
    task_type    : str — 'pure' | 'harmonic' | 'wn'
    task_list    : list[str] — all tasks (determines shared freq grid)
    init_params  : dict — initial parameter pytree, e.g. from ckpt['params']
    trainable    : 'all' | 'context' | 'sensory' | list[str]
        'all'     : train every weight
        'context' : only W_in rows for the 3 context one-hot channels
        'sensory' : only W_in rows for the n_freq frequency channels
        list      : names of top-level param keys to train, e.g. ['W_in', 'W_out']
    save_dir     : str — directory for checkpoints
    training_params_path : str | None
    task_params_path     : str | None
    seed         : int

    Returns
    -------
    state : flax TrainState
    log   : dict with 'step', 'loss', 'hit_rate', 'fa_rate'
    """
    hp          = load_training_params(training_params_path)
    task_params = load_task_params(task_params_path)

    m_hp = hp['model']
    o_hp = hp['optimizer']
    t_hp = hp['training']

    timing      = task_params['timing']
    dt          = timing['dt']
    alpha       = dt / m_hp['tau']
    n_pulses    = timing['n_pulses']
    pulse_steps = max(1, int(np.ceil(timing['pulse_dur_ms'] / dt)))
    stim_on     = int(timing['pre_stim_dur_ms'] / dt)
    stim_off    = stim_on + n_pulses * pulse_steps
    tdim        = stim_off + int(timing['post_stim_dur_ms'] / dt)

    grid_hz, _  = get_stimulus_vectors(task_list, task_params)
    n_freq      = len(grid_hz)
    n_in        = n_freq + 4   # freq + fixation(1) + context(3)

    n_steps         = t_hp['n_steps']
    display_step    = t_hp['display_step']
    batch_size      = t_hp['batch_size']
    response_thresh = task_params.get('response', {}).get('response_thresh', 0.5)
    resp_onset_pulses = timing.get('resp_onset_pulses', 2)
    resp_onset      = stim_on + resp_onset_pulses * pulse_steps
    reaction_steps  = min(int(2000 / dt), tdim - stim_on)

    model = LeakyRNN(
        n_rnn       = m_hp['n_rnn'],
        n_out       = 1,
        alpha       = alpha,
        sigma_rec   = m_hp['sigma_rec'],
        sigma_x     = m_hp['sigma_x'],
        activation  = m_hp['activation'],
        w_rec_coeff = m_hp['w_rec_coeff'],
    )
    state = create_train_state(
        model,
        n_in          = n_in,
        batch_size    = batch_size,
        tdim          = tdim,
        learning_rate = o_hp['learning_rate'],
        grad_clip     = o_hp['grad_clip'],
        seed          = seed,
    )
    # Replace random init with provided weights
    state = state.replace(params=init_params)

    grad_mask  = _make_grad_mask(state.params, trainable, n_freq)
    train_step = _make_transfer_train_step(grad_mask)

    # Count trainable scalars for the info line
    n_trainable = sum(int(jnp.sum(m > 0)) for m in jax.tree_util.tree_leaves(grad_mask))
    n_total     = sum(int(p.size) for p in jax.tree_util.tree_leaves(state.params))

    save_path = Path(save_dir or results_dir())
    save_path.mkdir(parents=True, exist_ok=True)

    config = {
        'task_type': task_type, 'task_list': task_list,
        'trainable': trainable if isinstance(trainable, str) else list(trainable),
        'n_in': n_in, 'n_freq': n_freq, 'tdim': tdim,
        'stim_on': stim_on, 'stim_off': stim_off, 'seed': seed,
        'training': hp, 'task_params': task_params,
    }
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f'task_type : {task_type}   trainable: {trainable}')
    print(f'n_freq={n_freq}  n_in={n_in}  n_rnn={m_hp["n_rnn"]}  alpha={alpha:.3f}')
    print(f'trainable params: {n_trainable}/{n_total}')

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)

    log = {'step': [], 'loss': [], 'hit_rate': [], 'fa_rate': [],
           'resp_go': [], 'resp_nogo': [], 'traces_go': [], 'traces_nogo': []}

    for step in range(n_steps):
        trial = generate_trials(task_type, task_list, mode='random',
                                batch_size=batch_size, rng=rng)
        key, noise_key = jax.random.split(key)
        state, loss = train_step(
            state, jnp.array(trial.x), jnp.array(trial.y),
            jnp.array(trial.mask), noise_key)

        if (step + 1) % display_step == 0:
            eval_trial = generate_trials(task_type, task_list, mode='random',
                                         batch_size=2 * batch_size, rng=rng)
            key, eval_key = jax.random.split(key)
            ys_pred, _ = state.apply_fn(
                {'params': state.params}, jnp.array(eval_trial.x),
                noise_input=True, noise_rec=True,
                rngs={'noise': eval_key})
            ys_pred = np.array(ys_pred)
            is_go   = eval_trial.is_go

            pre_peak  = (ys_pred[:stim_on, :, 0].max(axis=0) if stim_on > 0
                         else np.zeros(ys_pred.shape[1]))
            post_peak = ys_pred[stim_on : stim_on + reaction_steps, :, 0].max(axis=0)
            is_resp   = (post_peak > response_thresh) & (pre_peak < 0.3)

            hit = float(np.sum(is_resp[is_go])  / is_go.sum())    if is_go.any()    else np.nan
            fa  = float(np.sum(is_resp[~is_go]) / (~is_go).sum()) if (~is_go).any() else np.nan

            traces_go   = ys_pred[:, is_go,  0].T if is_go.any()   else np.empty((0, tdim))
            traces_nogo = ys_pred[:, ~is_go, 0].T if (~is_go).any() else np.empty((0, tdim))

            log['step'].append(step + 1)
            log['loss'].append(float(loss))
            log['hit_rate'].append(hit)
            log['fa_rate'].append(fa)
            log['resp_go'].append(traces_go.mean(axis=0).tolist() if is_go.any()
                                  else [np.nan] * tdim)
            log['resp_nogo'].append(traces_nogo.mean(axis=0).tolist() if (~is_go).any()
                                    else [np.nan] * tdim)
            log['traces_go'].append(traces_go.tolist())
            log['traces_nogo'].append(traces_nogo.tolist())

            print(f'step {step+1:6d}  loss={float(loss):.4f}  '
                  f'hit={hit:.2f}  FA={fa:.2f}')

            ckpt = {
                'params': state.params, 'step': step + 1,
                'log': log, 'task_type': task_type, 'task_list': task_list,
                'trainable': trainable,
                'stim_on': stim_on, 'stim_off': stim_off,
                'resp_onset': resp_onset,
                'reaction_steps': reaction_steps, 'dt': dt,
                'response_thresh': response_thresh,
            }
            with open(save_path / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(ckpt, f)

    return state, log


# ── Notebook API ──────────────────────────────────────────────────────────────

def setup_training(train_task, task_list, n_steps, k, seed, save_dir=None,
                   init_params=None, trainable='all',
                   training_params_path=None, task_params_path=None,
                   source_step=None):
    """
    Load hyperparams, build model + train state, write config to disk.

    Parameters
    ----------
    train_task   : str — 'pure' | 'harmonic' | 'wn'
    task_list    : list[str] — all tasks (determines shared freq grid)
    n_steps      : int — total training batches
    k            : int — evaluate and checkpoint every k steps
    seed         : int
    save_dir     : str — base directory; a named subfolder is created inside it
    init_params  : dict | None — replace random init with these weights (transfer)
    trainable    : str | list | None — saved to config['training']['model']['trainable']
    training_params_path : str | None
    task_params_path     : str | None
    source_step  : int | None — transfer learning: checkpoint step of the source model
                   the weights came from.  If not None, '_src<source_step>' is appended
                   to the run's folder name (and saved in config.json).

    Returns
    -------
    state : flax TrainState
    cfg   : dict  (timing values, grid info, paths, hyperparam sub-dicts)
    """
    hp          = load_training_params(training_params_path)
    hp['training']['n_steps']      = n_steps
    hp['training']['display_step'] = k
    hp['training']['random_seed']  = seed
    if trainable is not None:
        hp['model']['trainable'] = (trainable if isinstance(trainable, str)
                                    else list(trainable))
    task_params = load_task_params(task_params_path)

    m_hp = hp['model']
    o_hp = hp['optimizer']
    t_hp = hp['training']

    timing      = task_params['timing']
    dt          = timing['dt']
    alpha       = dt / m_hp['tau']
    n_pulses    = timing['n_pulses']
    pulse_steps = max(1, math.ceil(timing['pulse_dur_ms'] / dt))
    stim_on     = int(timing['pre_stim_dur_ms'] / dt)
    stim_off    = stim_on + n_pulses * pulse_steps
    tdim        = stim_off + int(timing['post_stim_dur_ms'] / dt)

    grid_hz, _ = get_stimulus_vectors(task_list, task_params)
    n_freq = len(grid_hz)
    n_in   = n_freq + 4   # freq + fixation(1) + context(3)

    model = LeakyRNN(
        n_rnn       = m_hp['n_rnn'],
        n_out       = 1,
        alpha       = alpha,
        sigma_rec   = m_hp['sigma_rec'],
        sigma_x     = m_hp['sigma_x'],
        activation  = m_hp['activation'],
        w_rec_coeff = m_hp['w_rec_coeff'],
    )
    state = create_train_state(
        model,
        n_in          = n_in,
        batch_size    = t_hp['batch_size'],
        tdim          = tdim,
        learning_rate = o_hp['learning_rate'],
        grad_clip     = o_hp['grad_clip'],
        seed          = seed,
    )
    if init_params is not None:
        state = state.replace(params=init_params)

    subfolder = (f"{train_task}_{seed}_lr{o_hp['learning_rate']}"
                 f"_bs{t_hp['batch_size']}_noise{m_hp['sigma_x']}_nrnn{m_hp['n_rnn']}")
    if source_step is not None:
        subfolder += f'_src{source_step}'
    save_path = Path(save_dir or results_dir()) / train_task / subfolder
    save_path.mkdir(parents=True, exist_ok=True)

    # Store train_task as a list when it encodes multiple tasks (e.g. 'pure-harmonic')
    config_train_task = (train_task.split('-')
                         if isinstance(train_task, str) and '-' in train_task
                         else train_task)
    config = {
        'train_task': config_train_task, 'task_list': task_list,
        'n_in': n_in, 'tdim': tdim, 'stim_on': stim_on,
        'stim_off': stim_off, 'pulse_steps': pulse_steps, 'seed': seed,
        'source_step': source_step,
        'training': hp, 'task_params': task_params,
    }
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    with open(save_path / 'train_params.json', 'w') as f:
        json.dump(hp, f, indent=2)

    stim_dur_ms = n_pulses * pulse_steps * dt
    print(f'task_list : {task_list}')
    print(f'train_task: {train_task}')
    print(f'n_freq={n_freq}  n_in={n_in} (freq+fixation+context)'
          f'  n_rnn={m_hp["n_rnn"]}  alpha={alpha:.3f}')
    print(f'pulse_dur={pulse_steps*dt}ms  n_pulses={n_pulses}'
          f'  stim_dur={stim_dur_ms}ms  tdim={tdim} ({tdim*dt}ms)')
    print(f'save_dir  : {save_path.resolve()}')

    cfg = dict(
        hp=hp, task_params=task_params,
        m_hp=m_hp, o_hp=o_hp, t_hp=t_hp,
        timing=timing, dt=dt, alpha=alpha,
        n_pulses=n_pulses, pulse_steps=pulse_steps,
        stim_on=stim_on, stim_off=stim_off, tdim=tdim,
        n_freq=n_freq, n_in=n_in, save_path=save_path,
        n_steps=n_steps, k=k, seed=seed,
        train_task=train_task, task_list=task_list,
    )
    return state, cfg


def run_training_loop(state, train_task, task_list, cfg,
                      train_step_fn=None, snapshot_every=2000):
    """
    Train only — no evaluation during the loop.

    Saves a copy of model parameters at each eval step (every cfg['k'] steps)
    and returns them as ``snapshots``.  Call ``evaluate_training`` afterwards
    to compute hit/FA rates and response traces across any set of tasks.

    Parameters
    ----------
    state           : flax TrainState
    train_task      : str
    task_list       : list[str]
    cfg             : dict returned by setup_training
    train_step_fn   : callable | None — JIT step; defaults to _train_step
    snapshot_every  : int — also save a named checkpoint_{step}.pkl every
                      this many steps (must be a multiple of cfg['k'])

    Returns
    -------
    state     : updated TrainState
    snapshots : list of {'step': int, 'params': dict, 'loss': float}
    """
    if train_step_fn is None:
        train_step_fn = _train_step

    t_hp        = cfg['t_hp']
    timing      = cfg['timing']
    dt          = cfg['dt']
    stim_on     = cfg['stim_on']
    stim_off    = cfg['stim_off']
    pulse_steps = cfg['pulse_steps']
    save_path   = cfg['save_path']
    n_steps     = cfg['n_steps']
    k           = cfg['k']
    seed        = cfg['seed']

    resp_onset_pulses = timing.get('resp_onset_pulses', 2)
    resp_onset        = stim_on + resp_onset_pulses * pulse_steps
    response_thresh   = cfg['task_params'].get('response', {}).get('response_thresh', 0.5)

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)

    snapshots = []

    for step in range(n_steps):
        trial = generate_trials(train_task, task_list, mode='random',
                                batch_size=t_hp['batch_size'], rng=rng)
        key, noise_key = jax.random.split(key)
        state, loss = train_step_fn(
            state, jnp.array(trial.x), jnp.array(trial.y),
            jnp.array(trial.mask), noise_key)

        if (step + 1) % k == 0:
            snap = {
                'step':   step + 1,
                'params': jax.tree_util.tree_map(np.array, state.params),
                'loss':   float(loss),
            }
            snapshots.append(snap)
            print(f'step {step+1:6d}  loss={float(loss):.4f}')

            ckpt = {
                'params': state.params, 'step': step + 1,
                'train_task': train_task, 'task_list': task_list,
                'stim_on': stim_on, 'stim_off': stim_off,
                'resp_onset': resp_onset,
                'dt': dt, 'k': k,
                'response_thresh': response_thresh,
            }
            with open(save_path / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(ckpt, f)
            if snapshot_every and (step + 1) % snapshot_every == 0:
                with open(save_path / f'checkpoint_{step+1}.pkl', 'wb') as f:
                    pickle.dump(ckpt, f)

    print('Done.')
    return state, snapshots


def evaluate_training(state, snapshots, tasks, cfg,
                      eval_with_noise=True, batch_size=None, seed=0):
    """
    Evaluate each training snapshot on one or more tasks.

    Parameters
    ----------
    state      : flax TrainState — used only for apply_fn; params are swapped
                 per snapshot
    snapshots  : list of {'step', 'params', 'loss'} from run_training_loop
    tasks      : list[str] — tasks to evaluate; first entry is treated as the
                 target (training) task, the rest as source/transfer tasks
    cfg        : dict returned by setup_training
    eval_with_noise : bool
    batch_size : int | None — defaults to 2 × cfg['t_hp']['batch_size']
    seed       : int

    Returns
    -------
    log : dict with keys 'step', 'loss', 'tasks', and for each task name t:
          f'{t}_hit_rate', f'{t}_fa_rate',
          f'{t}_resp_go',  f'{t}_resp_nogo',
          f'{t}_traces_go', f'{t}_traces_nogo'
    """
    t_hp        = cfg['t_hp']
    dt          = cfg['dt']
    stim_on     = cfg['stim_on']
    stim_off    = cfg['stim_off']
    tdim        = cfg['tdim']
    timing      = cfg['timing']
    pulse_steps = cfg['pulse_steps']
    task_list   = cfg['task_list']
    save_path   = cfg['save_path']

    bsize             = batch_size or 2 * t_hp['batch_size']
    resp_onset_pulses = timing.get('resp_onset_pulses', 2)
    response_thresh   = cfg['task_params'].get('response', {}).get('response_thresh', 0.5)

    # Fixed alignment window around go cue, consistent across variable pre-stim batches.
    # fixed_pre: steps of pre-stim kept before go cue = minimum guaranteed by pre_stim_range_ms.
    # fixed_post: stim + post-stim steps after go cue.
    pre_range    = timing.get('pre_stim_range_ms', [timing['pre_stim_dur_ms']] * 2)
    fixed_pre    = int(min(pre_range) / dt)
    n_pulses     = timing['n_pulses']
    stim_dur     = n_pulses * pulse_steps
    post_steps   = int(timing['post_stim_dur_ms'] / dt)
    fixed_post   = stim_dur + post_steps
    aligned_tdim = fixed_pre + fixed_post
    resp_onset   = fixed_pre + resp_onset_pulses * pulse_steps   # in aligned frame
    reaction_steps = min(int(2000 / dt), fixed_post)

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)

    # If checkpoint_init.pkl exists (transfer learning), prepend it as step 0
    eval_snapshots = list(snapshots)
    ckpt_init_path = save_path / 'checkpoint_init.pkl'
    if ckpt_init_path.exists():
        with open(ckpt_init_path, 'rb') as f:
            init_ckpt = pickle.load(f)
        eval_snapshots = [{'step': 0, 'params': init_ckpt['params'],
                           'loss': float('nan')}] + eval_snapshots

    log = {'step': [], 'loss': [], 'tasks': tasks}
    for t in tasks:
        log[f'{t}_hit_rate']    = []
        log[f'{t}_fa_rate']     = []
        log[f'{t}_resp_go']     = []
        log[f'{t}_resp_nogo']   = []
        log[f'{t}_traces_go']   = []
        log[f'{t}_traces_nogo'] = []

    for snap in eval_snapshots:
        params = jax.tree_util.tree_map(jnp.array, snap['params'])
        log['step'].append(snap['step'])
        log['loss'].append(snap['loss'])

        task_strs = []
        for t in tasks:
            trial = generate_trials(t, task_list, mode='random',
                                    batch_size=bsize, rng=rng)
            key, nk = jax.random.split(key)
            ys, _ = state.apply_fn(
                {'params': params}, jnp.array(trial.x),
                noise_input=eval_with_noise, noise_rec=True,
                rngs={'noise': nk})
            ys    = np.array(ys)
            is_go = trial.is_go

            # Align to go cue: trim a fixed window around trial.stim_on.
            stim_on_arr = trial.stim_on  # (bsize,) per-trial go cue steps
            ys_aln = np.stack([
                ys[so - fixed_pre : so + fixed_post, i, :]
                for i, so in enumerate(stim_on_arr)
            ], axis=1)  # (aligned_tdim, bsize, 1)

            pre  = (ys_aln[:fixed_pre, :, 0].max(axis=0) if fixed_pre > 0
                    else np.zeros(bsize))
            post = ys_aln[fixed_pre : fixed_pre + reaction_steps, :, 0].max(axis=0)
            resp = (post > response_thresh) & (pre < 0.3)

            hit = float(resp[is_go].mean())  if is_go.any()    else np.nan
            fa  = float(resp[~is_go].mean()) if (~is_go).any() else np.nan

            tgo   = ys_aln[:, is_go,  0].T if is_go.any()    else np.empty((0, aligned_tdim))
            tnogo = ys_aln[:, ~is_go, 0].T if (~is_go).any() else np.empty((0, aligned_tdim))

            log[f'{t}_hit_rate'].append(hit)
            log[f'{t}_fa_rate'].append(fa)
            log[f'{t}_resp_go'].append(
                (tgo.mean(axis=0) if len(tgo) > 0 else np.full(aligned_tdim, np.nan)).tolist())
            log[f'{t}_resp_nogo'].append(
                (tnogo.mean(axis=0) if len(tnogo) > 0 else np.full(aligned_tdim, np.nan)).tolist())
            log[f'{t}_traces_go'].append(tgo.tolist())
            log[f'{t}_traces_nogo'].append(tnogo.tolist())
            task_strs.append(f'{t}: hit={hit:.2f} FA={fa:.2f}')

        print(f'step {snap["step"]:6d}  ' + '  |  '.join(task_strs))

    # Persist the full log into checkpoint.pkl
    ckpt_path = save_path / 'checkpoint.pkl'
    if ckpt_path.exists():
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        ckpt['log']              = log
        ckpt['resp_onset']       = resp_onset
        ckpt['reaction_steps']   = reaction_steps
        ckpt['fixed_pre']        = fixed_pre
        ckpt['aligned_tdim']     = aligned_tdim
        with open(ckpt_path, 'wb') as f:
            pickle.dump(ckpt, f)

    # Also persist per-snapshot eval into each named checkpoint_{step}.pkl
    for i, snap in enumerate(snapshots):
        snap_path = save_path / f'checkpoint_{snap["step"]}.pkl'
        if snap_path.exists():
            with open(snap_path, 'rb') as f:
                snap_ckpt = pickle.load(f)
            snap_ckpt['resp_onset']     = resp_onset
            snap_ckpt['reaction_steps'] = reaction_steps
            snap_ckpt['fixed_pre']      = fixed_pre
            snap_ckpt['aligned_tdim']   = aligned_tdim
            snap_ckpt['snap_eval'] = {
                t: {
                    'hit_rate':   log[f'{t}_hit_rate'][i],
                    'fa_rate':    log[f'{t}_fa_rate'][i],
                    'resp_go':    log[f'{t}_resp_go'][i],
                    'resp_nogo':  log[f'{t}_resp_nogo'][i],
                    'traces_go':  log[f'{t}_traces_go'][i],
                    'traces_nogo': log[f'{t}_traces_nogo'][i],
                }
                for t in tasks
            }
            with open(snap_path, 'wb') as f:
                pickle.dump(snap_ckpt, f)

    return log


def plot_training_log(log, cfg, title='', last_snap_step=None, n_snaps=6):
    """
    Plot training performance from an evaluation log.

    Parameters
    ----------
    log            : dict returned by evaluate_training
    cfg            : dict returned by setup_training
    title          : str — figure suptitle
    last_snap_step : int | None — cap step index for snapshot curves (None = all)
    n_snaps        : int — number of snapshot curves per panel (default 6)

    Returns
    -------
    fig : matplotlib Figure
    """
    dt          = cfg['dt']
    k           = cfg['k']
    timing      = cfg['timing']
    pulse_steps = cfg['pulse_steps']
    save_path   = cfg['save_path']
    response_thresh   = cfg['task_params'].get('response', {}).get('response_thresh', 0.5)
    resp_onset_pulses = timing.get('resp_onset_pulses', 2)

    # Aligned-frame geometry (must match evaluate_training).
    pre_range    = timing.get('pre_stim_range_ms', [timing['pre_stim_dur_ms']] * 2)
    fixed_pre    = int(min(pre_range) / dt)
    n_pulses     = timing['n_pulses']
    stim_dur     = n_pulses * pulse_steps
    post_steps   = int(timing['post_stim_dur_ms'] / dt)
    fixed_post   = stim_dur + post_steps
    aligned_tdim = fixed_pre + fixed_post
    stim_on      = fixed_pre                                        # go cue in aligned frame
    stim_off     = fixed_pre + stim_dur
    resp_onset   = fixed_pre + resp_onset_pulses * pulse_steps
    reaction_steps = min(int(2000 / dt), fixed_post)

    tasks  = log.get('tasks', [])
    steps  = np.array(log['step'])
    t_ms   = np.arange(aligned_tdim) * dt

    end_idx  = (int(np.searchsorted(steps, last_snap_step))
                if last_snap_step is not None else len(steps))
    n_snaps  = min(n_snaps, max(end_idx, 1))
    snap_idx = np.round(
        np.logspace(0, np.log10(max(end_idx - 1, 1)), n_snaps)).astype(int)
    snap_idx = np.clip(snap_idx, 0, len(steps) - 1)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, n_snaps))

    n_rows = max(len(tasks), 1)
    fig, axes_all = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes_all = axes_all[np.newaxis, :]

    def _draw_response_panel(ax, mean_arr, traces_key, panel_title):
        for i, si in enumerate(snap_idx):
            mu     = np.array(mean_arr[si])
            traces = np.array(log[traces_key][si])
            sd     = traces.std(axis=0) if len(traces) > 1 else np.zeros_like(mu)
            ax.plot(t_ms, mu, color=cmap[i], lw=1.8, label=f'step {steps[si]}')
            ax.fill_between(t_ms, mu - sd, mu + sd, color=cmap[i], alpha=0.2)
        ax.axvline(stim_on    * dt, color='k',          ls='--', lw=1,   alpha=0.6, label='stim on')
        ax.axvline(stim_off   * dt, color='k',          ls=':',  lw=1,   alpha=0.6, label='stim off')
        ax.axvline(resp_onset * dt, color='tab:orange', ls='--', lw=1.2, alpha=0.8, label='resp onset')
        ax.axhline(response_thresh, color='gray', ls='--', lw=1)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Mean ± SD output')
        ax.set_title(panel_title)
        ax.set_ylim(-0.3, 1.3)
        ax.legend(fontsize=7)

    # ── Row 0: hit/FA/correct for all tasks + response curves for first task ──
    ax0 = axes_all[0, 0]
    line_styles = ['-', '--', ':', '-.']
    task_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(tasks), 1)))
    for ti, t in enumerate(tasks):
        ls   = line_styles[ti % len(line_styles)]
        col  = task_colors[ti]
        hr   = np.array(log.get(f'{t}_hit_rate', []))
        fr   = np.array(log.get(f'{t}_fa_rate',  []))
        corr = (hr + (1.0 - fr)) / 2.0
        ax0.plot(steps, hr,   color='g',        lw=1.5, ls=ls,     label=f'Hit — {t}',     alpha=0.7)
        ax0.plot(steps, fr,   color='r',        lw=1.5, ls=ls,     label=f'FA — {t}',      alpha=0.7)
        ax0.plot(steps, corr, color='b',        lw=2.5, ls=ls,     label=f'Correct — {t}')
        # shade between hit and correct-rejection to show the two components
        # ax0.fill_between(steps, fr, hr, color=col, alpha=0.08)
    ax0.axhline(0.5, color='gray', ls='--', lw=1)
    ax0.set_xlabel(f'Training step (eval every {k})')
    ax0.set_ylabel('Rate')
    ax0.set_title('Hit / FA / Correct rate')
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(fontsize=7)

    if tasks:
        t0 = tasks[0]
        _draw_response_panel(axes_all[0, 1],
                             log[f'{t0}_resp_go'],   f'{t0}_traces_go',
                             f'{t0} — Go trials')
        _draw_response_panel(axes_all[0, 2],
                             log[f'{t0}_resp_nogo'], f'{t0}_traces_nogo',
                             f'{t0} — No-Go trials')

    # ── Rows 1+: one row per additional (source) task ─────────────────────────
    for row, t in enumerate(tasks[1:], start=1):
        ax_left = axes_all[row, 0]
        if row == 1:
            ax_left.plot(steps, np.array(log['loss']), color='tab:purple', lw=2)
            ax_left.set_xlabel(f'Training step (eval every {k})')
            ax_left.set_ylabel('Loss (MSE)')
            ax_left.set_title('Training loss')
            ax_left.set_xlim(steps[0] if len(steps) else 0, steps[-1] if len(steps) else 1)
        else:
            ax_left.axis('off')
        _draw_response_panel(axes_all[row, 1],
                             log[f'{t}_resp_go'],   f'{t}_traces_go',
                             f'{t} — Go trials')
        _draw_response_panel(axes_all[row, 2],
                             log[f'{t}_resp_nogo'], f'{t}_traces_nogo',
                             f'{t} — No-Go trials')

    fig.suptitle(title or str(save_path), fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path / 'perf_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── Evaluation helper ──────────────────────────────────────────────────────────

def _evaluate(state, tasks, task_params, batch_size, rng):
    """Return per-task fraction correct (threshold at 0.5 at stim onset)."""
    perfs = {}
    for task_type in tasks:
        trial = generate_trials(task_type, tasks, mode='random',
                                batch_size=batch_size,
                                params_path=None, rng=rng)
        xs_eval = jnp.array(trial.x)
        ys_pred, _ = state.apply_fn(
            {'params': state.params}, xs_eval)
        response = np.array(ys_pred)[trial.stim_on, :, 0]   # value at stim onset
        correct  = (response > 0.5) == trial.is_go
        perfs[task_type] = float(correct.mean())
    return perfs


# ── Main training function ─────────────────────────────────────────────────────

def train(tasks, save_dir=None, training_params_path=None,
          task_params_path=None, seed=0):
    """
    Train a LeakyRNN on the specified auditory go/no-go tasks.

    Parameters
    ----------
    tasks : list of str
        Task types to train on; also determines the shared frequency grid.
        Choices: 'pure', 'harmonic', 'wn'.
    save_dir : str
        Directory for checkpoints and log (created if absent).
    training_params_path : str, optional
        Path to training.json (defaults to code/training.json).
    task_params_path : str, optional
        Path to task_params.json (defaults to code/task_params.json).
    seed : int

    Returns
    -------
    state : flax TrainState  (final parameters)
    log   : dict             (loss and per-task performance history)
    """
    hp          = load_training_params(training_params_path)
    task_params = load_task_params(task_params_path)

    m_hp = hp['model']
    o_hp = hp['optimizer']
    t_hp = hp['training']

    dt    = task_params['timing']['dt']
    alpha = dt / m_hp['tau']

    # Frequency grid dimensions
    grid_hz, _ = get_stimulus_vectors(tasks, task_params)
    n_in = len(grid_hz)

    # Trial time dimensions
    timing   = task_params['timing']
    stim_on  = int(timing['pre_stim_dur_ms']  / dt)
    stim_off = stim_on + int(timing['stim_dur_ms'] / dt)
    tdim     = stim_off + int(timing['post_stim_dur_ms'] / dt)

    # Build model and initial state
    model = LeakyRNN(
        n_rnn       = m_hp['n_rnn'],
        n_out       = 1,
        alpha       = alpha,
        sigma_rec   = m_hp['sigma_rec'],
        sigma_x     = m_hp['sigma_x'],
        activation  = m_hp['activation'],
        w_rec_coeff = m_hp['w_rec_coeff'],
    )
    state = create_train_state(
        model,
        n_in        = n_in,
        batch_size  = t_hp['batch_size'],
        tdim        = tdim,
        learning_rate = o_hp['learning_rate'],
        grad_clip   = o_hp['grad_clip'],
        seed        = seed,
    )

    save_path = Path(save_dir or results_dir())
    save_path.mkdir(parents=True, exist_ok=True)

    # Save run config alongside checkpoint
    config = {'tasks': tasks, 'training': hp, 'task_params': task_params,
              'n_in': n_in, 'seed': seed}
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)

    log = {'step': [], 'loss': [], 'perf': {tt: [] for tt in tasks}}

    print(f"Tasks : {tasks}")
    print(f"n_freq: {n_in}   n_rnn: {m_hp['n_rnn']}   alpha: {alpha:.3f}")
    print(f"Steps : {t_hp['n_steps']}   batch: {t_hp['batch_size']}   "
          f"display every {t_hp['display_step']}")
    print()

    t0 = time.time()
    for step in range(t_hp['n_steps']):
        task_type = str(rng.choice(tasks))
        trial     = generate_trials(task_type, tasks, mode='random',
                                    batch_size=t_hp['batch_size'], rng=rng)

        xs       = jnp.array(trial.x)
        ys_target = jnp.array(trial.y)

        key, noise_key = jax.random.split(key)
        state, loss    = _train_step(state, xs, ys_target,
                                     jnp.array(trial.mask), noise_key)

        if (step + 1) % t_hp['display_step'] == 0 or step == 0:
            perfs    = _evaluate(state, tasks, task_params, t_hp['batch_size'], rng)
            elapsed  = time.time() - t0
            perf_str = '  '.join(f'{tt}={v:.2f}' for tt, v in perfs.items())
            print(f"step {step+1:6d}  loss={float(loss):.4f}  {perf_str}  "
                  f"({elapsed:.0f}s)")

            log['step'].append(step + 1)
            log['loss'].append(float(loss))
            for tt, v in perfs.items():
                log['perf'][tt].append(v)

            ckpt = {'params': state.params, 'step': step + 1,
                    'log': log, 'tasks': tasks}
            with open(save_path / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(ckpt, f)

            if all(v >= t_hp['target_perf'] for v in perfs.values()):
                print(f"Target performance {t_hp['target_perf']} reached.")
                break

    return state, log


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train auditory go/no-go network.')
    parser.add_argument('--tasks',     nargs='+', default=['pure', 'harmonic', 'wn'],
                        help='Task types to train on')
    parser.add_argument('--save_dir',  default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--training_params', default=None,
                        help='Path to training.json')
    parser.add_argument('--task_params',     default=None,
                        help='Path to task_params.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    train(tasks=args.tasks, save_dir=args.save_dir,
          training_params_path=args.training_params,
          task_params_path=args.task_params,
          seed=args.seed)
