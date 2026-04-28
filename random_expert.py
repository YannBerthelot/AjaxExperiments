"""Random-action 'expert' for the random_expert ablation.

Mirrors the shape of a real expert's state pytree so the actor input
(env_obs + flatten(expert_state)) keeps the same dimension as edge-qa,
but the expert's action and state evolution carry no useful information
about the task. Tests "the expert provides signal, not just structured
noise" cleanly: same architecture, only expert content scrambled.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


class RandomScrambledExpert:
    """Stateful random expert that mimics a PID/CPG expert's state shape.

    Constructed from a template expert (the env's real expert) and an
    action dim. ``init_state(num_envs)`` returns a pytree with the SAME
    leaf shapes as the template, so ``flatten_expert_state`` produces an
    obs-aug vector of identical dim.

    ``__call__(state, obs) -> (action, new_state)``:
        - action: per-step uniform U[-1, 1]^action_dim, derived from a
          state-dependent fold_in of a fixed seed (no Python-side RNG;
          deterministic given (state, obs); distinct each step).
        - new_state: small bounded noise update of the carry leaves.
          Stays in tanh range so flatten_expert_state output magnitudes
          remain comparable to a real PID's integrator (after the
          agent's running normalisation).
    """

    def __init__(self, template_expert, action_dim: int, key_seed: int = 42):
        # We don't actually call the template; we just use its zero_state.
        self._zero_state = template_expert._zero_state
        self._action_dim = int(action_dim)
        self._key_seed = int(key_seed)
        self._params = getattr(template_expert, "_params", None)

    def init_state(self, num_envs: int):
        """Zero-initialise state with a leading batch dimension matching
        the template expert's init_state contract."""
        return jax.tree.map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), (num_envs,) + jnp.shape(x)),
            self._zero_state,
        )

    def _state_seed(self, state) -> jax.Array:
        """Derive a per-step PRNGKey from the state itself. Differs each
        step because state changes; jit-friendly via fold_in."""
        leaves = jax.tree_util.tree_leaves(state)
        if not leaves:
            seed = jnp.uint32(0)
        else:
            stacked = jnp.stack([jnp.sum(jnp.asarray(l)) for l in leaves])
            seed = jnp.uint32(jnp.abs(jnp.sum(stacked) * 1e6))
        return jax.random.fold_in(jax.random.PRNGKey(self._key_seed), seed)

    def __call__(self, *args):
        # Two-arg form: (state, obs) -> (action, new_state).
        # One-arg form: (obs,) -> action  (for stateless callers).
        if len(args) == 2:
            state, obs = args
            stateful = True
        else:
            (obs,) = args
            state = self._zero_state
            stateful = False

        key = self._state_seed(state)
        action_key, state_key = jax.random.split(key)

        batch_shape = obs.shape[:-1]
        action = jax.random.uniform(
            action_key, batch_shape + (self._action_dim,),
            minval=-1.0, maxval=1.0,
        )

        new_state = jax.tree.map(
            lambda x: jnp.tanh(
                x * 0.99
                + jax.random.normal(state_key, jnp.shape(x)) * 0.05
            ),
            state,
        )
        if stateful:
            return action, new_state
        return action

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
