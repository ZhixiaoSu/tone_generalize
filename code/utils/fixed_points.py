"""
Fixed-point finding for the LeakyRNN.

Theory
------
A fixed point h* satisfies F(h*; x) = h*, where:
    F(h; x) = (1 - alpha) * h + alpha * relu(W_in @ x + W_rec @ h + b_rec)

We find h* by minimising q(h) = (1/2N) * ||F(h; x) - h||^2 using Adam (JAX/optax).
A fixed point is *stable* if all eigenvalues of dF/dh|_{h*} have |λ| < 1.

Supported trial phases
----------------------
'fixation'   — last pre-stimulus step (fixation=1, no stimulus, context on)
'stim_on'    — stimulus onset (fixation off, pulse 1 starts)
'stim_off'   — right after the last pulse (silence + context)
'pulse_on'   — start of the N-th pulse, input held constant  (set pulse_num, 1-indexed)
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax


# ── Timing helpers ────────────────────────────────────────────────────────────

def get_phase_step(phase, stim_on, stim_off, pulse_steps, pulse_num=2):
    """
    Return the timestep index for a named trial phase.

    Parameters
    ----------
    phase       : 'fixation' | 'stim_on' | 'stim_off' | 'pulse_on'
    stim_on     : int
    stim_off    : int
    pulse_steps : int   timesteps per pulse
    pulse_num   : int   1-indexed pulse number (used by 'pulse_on')

    Returns
    -------
    int timestep
    """
    if phase == 'fixation':
        return stim_on - 1   # last pre-stimulus step: fixation=1, no stimulus
    elif phase == 'stim_on':
        return stim_on
    elif phase == 'stim_off':
        return stim_off
    elif phase == 'pulse_on':
        return stim_on + (pulse_num - 1) * pulse_steps
    else:
        raise ValueError(
            f"Unknown phase '{phase}'. "
            "Choose from: 'fixation', 'stim_on', 'stim_off', 'pulse_on'")


def phase_label(phase, pulse_num=None):
    """Human-readable label for a phase."""
    if phase in ('fixation', 'stim_on', 'stim_off'):
        return phase
    return f"{phase}  (pulse {pulse_num})"


# ── Core dynamics ─────────────────────────────────────────────────────────────

def _make_step_fn(params, alpha):
    """Return a JAX function (h, x) -> h_next (no noise)."""
    W_in  = jnp.array(params['W_in'])
    W_rec = jnp.array(params['W_rec'])
    b_rec = jnp.array(params['b_rec'])

    def step(h, x):
        pre   = x @ W_in + h @ W_rec + b_rec
        h_new = (1.0 - alpha) * h + alpha * jax.nn.relu(pre)
        return h_new

    return step


# ── Main entry point ──────────────────────────────────────────────────────────

def find_fixed_points(params, x_frozen, h_inits, alpha,
                      n_iters=3000, lr=1e-2, tol=1e-12,
                      unique_tol=1e-3, verbose=True):
    """
    Find fixed points of the LeakyRNN for a frozen input vector.

    Parameters
    ----------
    params     : dict   {W_in, W_rec, b_rec, ...}  (numpy or JAX arrays)
    x_frozen   : (n_in,)        input held constant during optimisation
    h_inits    : (n_inits, n_rnn)  initial hidden states (use actual network states)
    alpha      : float            dt / tau
    n_iters    : int              Adam optimisation steps
    lr         : float            Adam learning rate
    tol        : float            q threshold for "converged"
    unique_tol : float            L2 distance for deduplication
    verbose    : bool

    Returns
    -------
    dict with keys
      'h'        : (n_fps, n_rnn)          unique converged fixed points
      'q'        : (n_fps,)                q-values  (should be ~0)
      'is_stable': (n_fps,) bool           all Jacobian eigenvalues |λ| < 1
      'jacobians': (n_fps, n_rnn, n_rnn)   recurrent Jacobian at each FP
    """
    step_fn = _make_step_fn(params, alpha)
    x     = jnp.array(x_frozen, dtype=jnp.float32)
    n_rnn = h_inits.shape[1]

    def q_fn(h):
        h_next = step_fn(h, x)
        return 0.5 * jnp.mean((h_next - h) ** 2)

    batch_q    = jax.jit(jax.vmap(q_fn))
    batch_grad = jax.jit(jax.vmap(jax.grad(q_fn)))

    optimizer = optax.adam(lr)
    h = jnp.array(h_inits, dtype=jnp.float32)
    opt_state = optimizer.init(h)

    @jax.jit
    def _update(h, opt_state):
        grads = batch_grad(h)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(h, updates), opt_state

    for i in range(n_iters):
        h, opt_state = _update(h, opt_state)
        if verbose and (i + 1) % 500 == 0:
            qs = np.array(batch_q(h))
            print(f"    iter {i+1:4d}  min_q={qs.min():.2e}  "
                  f"converged={int((qs < tol).sum())}/{len(qs)}")

    q_vals = np.array(batch_q(h))
    h_np   = np.array(h)

    converged = q_vals < tol
    h_conv = h_np[converged]
    q_conv = q_vals[converged]

    if len(h_conv) == 0:
        if verbose:
            print(f"    No FPs found (min_q={q_vals.min():.2e}). "
                  "Consider raising tol or n_iters.")
        return {'h':         np.zeros((0, n_rnn)),
                'q':         np.zeros(0),
                'is_stable': np.zeros(0, dtype=bool),
                'jacobians': np.zeros((0, n_rnn, n_rnn))}

    h_unique, q_unique = _deduplicate(h_conv, q_conv, unique_tol)

    # Jacobian of F(h; x) at each unique FP
    jac_fn  = jax.jit(jax.vmap(jax.jacobian(lambda h: step_fn(h, x))))
    jacs    = np.array(jac_fn(jnp.array(h_unique, dtype=jnp.float32)))
    eigvals = np.array([np.linalg.eigvals(J) for J in jacs])
    is_stable = np.abs(eigvals).max(axis=1) < 1.0

    if verbose:
        print(f"    → {len(h_unique)} unique FP(s): "
              f"{is_stable.sum()} stable, {(~is_stable).sum()} unstable")

    return {'h': h_unique, 'q': q_unique,
            'is_stable': is_stable, 'jacobians': jacs}


def _deduplicate(h, q, tol):
    """Keep one representative per cluster of FPs within L2 distance tol."""
    keep = []
    for i in range(len(h)):
        if all(np.linalg.norm(h[i] - h[j]) > tol for j in keep):
            keep.append(i)
    return h[keep], q[keep]
