"""Every orchestrator env's expert can be degraded, and sigma = 0 leaves it
unchanged (run_degradation_eval.py degrades the expert of each env)."""

import os
import sys

import jax
import jax.numpy as jnp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def test_every_orchestrator_expert_degrades():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from envs import make_env_by_name  # type: ignore
    from expert_degradation import degrade_expert  # type: ignore
    from run_full_study import ENVS  # type: ignore
    from sac_hyperparam_search import _load_expert_policy  # type: ignore

    failed = []
    for name in ENVS:
        env, _ = make_env_by_name(name)
        expert = _load_expert_policy(name, env)
        try:
            degrade_expert(expert, 0.5, jax.random.PRNGKey(0))
        except Exception as e:
            failed.append((name, type(e).__name__, str(e)))
    assert not failed, f"Degradation failed for: {failed}"


def test_zero_sigma_keeps_the_parameters_and_a_positive_one_moves_them():
    from cpg_experts import make_cheetah_cpg_expert  # type: ignore
    from expert_degradation import degrade_expert  # type: ignore

    expert = make_cheetah_cpg_expert()
    same = degrade_expert(expert, 0.0, jax.random.PRNGKey(0))
    moved = degrade_expert(expert, 0.5, jax.random.PRNGKey(0))
    assert jnp.allclose(same._params.amp_0, expert._params.amp_0)
    assert not jnp.allclose(moved._params.amp_0, expert._params.amp_0)
