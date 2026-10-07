"""
Leaky RNN model in JAX/Flax.

Requires: jax, flax, optax
  pip install jax[cuda12] flax optax        # GPU (Linux/WSL)
  pip install jax flax optax                # CPU-only

Architecture
------------
  h_t = (1 - alpha) * h_{t-1}
        + alpha * act(W_in @ x_t + W_rec @ h_{t-1} + b_rec + noise_rec)
  y_t = W_out @ h_t + b_out

where alpha = dt / tau.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state


# ── Model ─────────────────────────────────────────────────────────────────────

class LeakyRNN(nn.Module):
    n_rnn:      int
    n_out:      int
    alpha:      float        # dt / tau
    sigma_rec:  float        # recurrent noise std (at unit variance)
    sigma_x:    float        # input noise std
    activation: str  = 'relu'
    w_rec_coeff: float = 0.8

    @nn.compact
    def __call__(self, xs: jnp.ndarray,
                 noise_input: bool = False,
                 noise_rec:   bool = False) -> tuple:
        """
        Parameters
        ----------
        xs          : (tdim, batch, n_in)
        noise_input : if True, add Gaussian noise to the input (sigma_x)
        noise_rec   : if True, add Gaussian noise to recurrent neurons (sigma_rec)

        Returns
        -------
        ys : (tdim, batch, n_out)  readout at every time step
        hs : (tdim, batch, n_rnn) hidden state at every time step
        """
        n_in  = xs.shape[-1]
        tdim  = xs.shape[0]
        batch = xs.shape[1]

        W_in  = self.param('W_in',
                    nn.initializers.normal(stddev=1.0 / jnp.sqrt(n_in)),
                    (n_in, self.n_rnn))
        W_rec = self.param('W_rec',
                    nn.initializers.orthogonal(scale=self.w_rec_coeff),
                    (self.n_rnn, self.n_rnn))
        b_rec = self.param('b_rec', nn.initializers.zeros, (self.n_rnn,))
        W_out = self.param('W_out',
                    nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.n_rnn)),
                    (self.n_rnn, self.n_out))
        b_out = self.param('b_out', nn.initializers.zeros, (self.n_out,))

        act = {'relu': nn.relu, 'tanh': jnp.tanh,
               'softplus': nn.softplus}[self.activation]

        # Generate all noise up-front so scan stays pure
        if noise_input or noise_rec:
            key = self.make_rng('noise')
            k_x, k_rec = jax.random.split(key)
        if noise_input:
            xs = xs + self.sigma_x * jax.random.normal(k_x, xs.shape)
        if noise_rec:
            # std scales with sqrt(2/alpha) so per-step variance = sigma_rec^2
            rec_noise = (jnp.sqrt(2.0 / self.alpha) * self.sigma_rec
                         * jax.random.normal(k_rec, (tdim, batch, self.n_rnn)))
        else:
            rec_noise = jnp.zeros((tdim, batch, self.n_rnn))

        alpha = self.alpha

        def _step(h, inputs):
            x_t, eta_t = inputs
            pre   = x_t @ W_in + h @ W_rec + b_rec + eta_t
            h_new = (1.0 - alpha) * h + alpha * act(pre)
            y_t   = h_new @ W_out + b_out
            return h_new, (h_new, y_t)

        h0 = jnp.zeros((batch, self.n_rnn))
        _, (hs, ys) = jax.lax.scan(_step, h0, (xs, rec_noise))
        return ys, hs   # (tdim, batch, n_out), (tdim, batch, n_rnn)


# ── TrainState factory ─────────────────────────────────────────────────────────

def create_train_state(model: LeakyRNN, n_in: int, batch_size: int, tdim: int,
                       learning_rate: float, grad_clip: float, seed: int = 0
                       ) -> train_state.TrainState:
    """Initialise parameters and create an Optax Adam train state."""
    k_params, k_noise = jax.random.split(jax.random.PRNGKey(seed))
    dummy = jnp.zeros((tdim, batch_size, n_in))
    variables = model.init({'params': k_params, 'noise': k_noise}, dummy)
    params = variables['params']

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)
