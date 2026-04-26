"""Regression: every env name in the orchestrator's ENVS list must resolve
through ``envs.make_env_by_name``.

Before this test: the orchestrator used ``NuclearReactor`` but ``envs.py``
only registered ``Reactor``. The mismatch produced an immediate crash in
every Reactor HPO trial, which the HPO master silently absorbed as 8
FAILED trials before the study was inspected.
"""

import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def test_every_orchestrator_env_resolves():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_disable_hlo_passes=layout_assignment"
    )
    from run_full_study import ENVS  # type: ignore
    from envs import make_env_by_name  # type: ignore

    unresolved = []
    for name in ENVS:
        try:
            env, _ = make_env_by_name(name)
        except Exception as e:
            unresolved.append((name, type(e).__name__, str(e)))
    assert not unresolved, (
        "run_full_study.ENVS contains names that envs.make_env_by_name "
        f"cannot resolve: {unresolved}"
    )


def test_expert_resolver_covers_all_envs():
    """The HPO script's ``_load_expert_policy`` must return a usable expert
    for every env in the orchestrator. A missing CPG lookup or a bad env
    name would leave the expert-using methods (edge, ibrl, jsrl, residual)
    unable to construct a SAC agent.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_disable_hlo_passes=layout_assignment"
    )
    from run_full_study import ENVS  # type: ignore
    from envs import make_env_by_name  # type: ignore
    from sac_hyperparam_search import _load_expert_policy  # type: ignore

    missing = []
    for name in ENVS:
        try:
            env, _ = make_env_by_name(name)
            exp = _load_expert_policy(name, env)
        except Exception as e:
            missing.append((name, type(e).__name__, str(e)))
            continue
        if exp is None:
            missing.append((name, "None", "_load_expert_policy returned None"))
    assert not missing, (
        f"Expert resolution failed for: {missing}"
    )
