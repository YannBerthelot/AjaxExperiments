"""Env registry — per-environment constants shared across experiment scripts.

Any script that takes a ``--env`` flag should call :func:`get_env_config` and
read every env-varying value from the returned :class:`EnvConfig`.  Adding a
new TargetGym environment means adding one entry to ``_build_registry``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class EnvConfig:
    """All environment-specific values needed by the experiment scripts."""

    name: str  # canonical name — must match the key in hyperparams/ajax_sac.yml
    env_cls: Callable
    params_cls: Callable
    params_kwargs: dict = field(default_factory=dict)

    # Noisy-expert study: target-channel metadata.
    # noise_target_obs_index : index of the observation channel holding the target signal
    # noise_target_clip      : (low, high) clip bounds for the perturbed target
    # noise_target_range     : full range of the target (σ = noise_pct / 100 × range)
    # noise_action_fn        : (perturbed_target, env_params) -> action vector
    noise_target_obs_index: Optional[int] = None
    noise_target_clip: Optional[tuple[float, float]] = None
    noise_target_range: Optional[float] = None
    noise_action_fn: Optional[Callable] = None

    @property
    def slug(self) -> str:
        """Lowercase name used as a filename / project suffix."""
        return self.name.lower()

    def make_env(self):
        env = self.env_cls()
        params = self.params_cls(**self.params_kwargs)
        return env, params


def _plane_noise_action(interpolator, perturbed_target):
    import jax.numpy as jnp

    power = interpolator(perturbed_target)[..., None]
    return jnp.concatenate([power, jnp.zeros_like(power)], axis=-1)


def _build_registry() -> dict[str, EnvConfig]:
    import math

    from target_gym import Plane, PlaneParams
    from target_gym.plane3d.env import PlaneParams3D
    from target_gym.plane3d.env_jax import Plane3DHeading

    return {
        "Plane": EnvConfig(
            name="Plane",
            env_cls=Plane,
            params_cls=PlaneParams,
            params_kwargs=dict(
                target_altitude_range=(3_000, 8_000),
                initial_altitude_range=(3_000, 8_000),
                max_steps_in_episode=10_000,
            ),
            noise_target_obs_index=6,
            noise_target_clip=(3_000.0, 8_000.0),
            noise_target_range=5_000.0,
            noise_action_fn=_plane_noise_action,
        ),
        "Plane3DHeading": EnvConfig(
            name="Plane3DHeading",
            env_cls=Plane3DHeading,
            params_cls=PlaneParams3D,
            params_kwargs=dict(
                target_heading_range=(-math.pi, math.pi),
                max_steps_in_episode=10_000,
            ),
            noise_target_obs_index=11,
            noise_target_clip=(-math.pi, math.pi),
            noise_target_range=2.0 * math.pi,
            noise_action_fn=None,
        ),
    }


_REGISTRY: Optional[dict[str, EnvConfig]] = None


def _registry() -> dict[str, EnvConfig]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_env_config(env_name: str) -> EnvConfig:
    reg = _registry()
    if env_name not in reg:
        raise ValueError(
            f"Unknown env {env_name!r}. Registered: {sorted(reg)}. "
            f"Add it to envs.py:_build_registry."
        )
    return reg[env_name]


def registered_envs() -> list[str]:
    return sorted(_registry())


# (env_module, env_class, params_module, params_class)
_ENV_IMPORT_MAP: dict[str, tuple[str, str, str, str]] = {
    "Plane":             ("target_gym.plane.env_jax",              "Airplane2D",         "target_gym.plane.env",              "PlaneParams"),
    "Plane3D":           ("target_gym.plane3d.env_jax",            "Plane3DHeading",     "target_gym.plane3d.env",            "PlaneParams3D"),
    "Plane3DHeading":    ("target_gym.plane3d.env_jax",            "Plane3DHeading",     "target_gym.plane3d.env",            "PlaneParams3D"),
    "Plane3DCircle":     ("target_gym.plane3d.env_jax",            "Plane3DCircle",      "target_gym.plane3d.env",            "PlaneParams3D"),
    "Plane3DFigureEight":("target_gym.plane3d.env_jax",            "Plane3DFigureEight", "target_gym.plane3d.env",            "PlaneParams3D"),
    "CSTR":              ("target_gym.pc_gym.cstr.env_jax",        "CSTR",               "target_gym.pc_gym.cstr.env_jax",    "CSTRParams"),
    "FirstOrderSystem":  ("target_gym.pc_gym.first_order.env_jax", "FirstOrderSystem",   "target_gym.pc_gym.first_order.env_jax", "FirstOrderParams"),
    "FourTank":          ("target_gym.pc_gym.four_tank.env_jax",   "FourTank",           "target_gym.pc_gym.four_tank.env_jax",   "FourTankParams"),
    "GlassFurnace":      ("target_gym.glass_furnace.env_jax",      "GlassFurnace",       "target_gym.glass_furnace.env",      "GlassFurnaceParams"),
    "Reactor":           ("target_gym.reactor.env_jax",            "Reactor",            "target_gym.reactor.env",            "ReactorParams"),
}


# Mujoco Playground locomotion envs we route through ``mujoco_playground.registry.load``.
# They don't go through ``_registry()`` (which is for ``target_gym`` only) and don't have
# separate params classes: Playground envs carry their params inside the Env object.
_PLAYGROUND_ENVS = {"CheetahRun", "WalkerRun", "HopperHop", "HopperStand",
                    "BarkourJoystick", "HumanoidStand", "HumanoidWalk",
                    "HumanoidRun"}


def make_env_by_name(env_name: str):
    """Instantiate an env by name. Supports three sources:

    1. ``target_gym`` registered env (via ``_build_registry``).
    2. ``target_gym`` importable env (via ``_ENV_IMPORT_MAP``).
    3. ``mujoco_playground`` locomotion env (via ``registry.load``).

    Returns ``(env, env_params)``. For Playground envs ``env_params`` is ``None``
    because their config is carried inside the env object.
    """
    if env_name in _registry():
        return get_env_config(env_name).make_env()
    if env_name in _PLAYGROUND_ENVS:
        # Delegate to Ajax's own playground builder, which applies the full
        # wrapper stack Ajax's SAC expects (EpisodeWrapper + VmapWrapper +
        # FinalObsWrapper + BraxAutoResetWrapper + BatchRngWrapper). Bypassing
        # this and hand-rolling the stack in AjaxExperiments introduced a vmap
        # rank bug caught only at runtime; use the canonical path instead.
        from ajax.environments.create import build_env_from_id
        env, _env_params = build_env_from_id(
            env_name, n_envs=1, episode_length=1000
        )
        return env, None
    entry = _ENV_IMPORT_MAP.get(env_name)
    if entry is None:
        raise ValueError(
            f"Unknown env {env_name!r}. Known target_gym envs: {sorted(_ENV_IMPORT_MAP)}. "
            f"Known mujoco_playground envs: {sorted(_PLAYGROUND_ENVS)}. "
            f"To add a new target_gym env, register it in envs.py:_build_registry."
        )
    import importlib
    env_mod, env_cls_name, params_mod, params_cls_name = entry
    env_cls = getattr(importlib.import_module(env_mod), env_cls_name)
    params_cls = getattr(importlib.import_module(params_mod), params_cls_name)
    return env_cls(), params_cls()
