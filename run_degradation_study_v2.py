"""Multi-GPU orchestrator for the degradation v2 study.

Sweeps (env x method x degradation_type x intensity) and dispatches each
cell to ``run_degradation_v2.py`` on a free GPU from ``--gpus``. Skips
cells whose output JSON already exists so the script can be safely
re-launched after interruption.

Designed for the rented remote instance where only v2 work runs (no
parallel HPO, no contention from sibling jobs). Simpler than
``run_full_study.py``: no GPU-busy probing, no ps-scan dedup, no
priority logic — just a worker thread per GPU pulling from a queue.

Usage::

    python run_degradation_study_v2.py --gpus 0 1 2 \\
        [--envs Plane3DCircle CheetahRun ...] \\
        [--methods sac_quality_aware ibrl ...] \\
        [--types action_bias latency obs_noise] \\
        [--n-timesteps 1000000] [--n-seeds 50]

Add ``--dry-run`` to print the work plan and exit.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "degradation_v2_results"
LOGS_DIR = ROOT / "study_v2_logs"

# Sweep defaults. Match run_full_study.py for env / method coverage.
DEFAULT_ENVS = [
    "Plane3DCircle",
    "CheetahRun",
    "FourTank",
    "GlassFurnace",
    "BarkourJoystick",
    "Plane3DFigureEight",
]
# Methods that take an expert (skip vanilla SAC; nothing to perturb).
DEFAULT_METHODS = [
    "sac_quality_aware",
    "ibrl",
    "jsrl_curriculum",
    "jsrl",
    "residual",
]
DEFAULT_TYPES = ("action_bias", "obs_noise")

# Per-type intensity sweeps. Keep three points per axis to match v1's
# (sigma in {0.25, 0.5, 0.75}) cadence.
INTENSITIES = {
    "action_bias": [0.10, 0.25, 0.50],   # uniform half-range on per-seed bias
    "obs_noise":   [0.05, 0.15, 0.30],    # std multiplier on per-dim obs std
    # "action_dropout": [0.1, 0.3, 0.5],  # deferred; uncomment when wrapper lands
}


@dataclass(frozen=True)
class V2Unit:
    env: str
    method: str
    deg_type: str
    intensity: float

    def _tag(self) -> str:
        return f"{self.intensity}"

    def name(self) -> str:
        return f"{self.env}/{self.method}/{self.deg_type}/{self._tag()}"

    def output_path(self) -> Path:
        return RESULTS_DIR / (
            f"{self.env.lower()}__{self.method}__"
            f"{self.deg_type}__{self._tag()}.json"
        )

    def log_path(self) -> Path:
        return LOGS_DIR / (
            f"{self.env.lower()}__{self.method}__"
            f"{self.deg_type}__{self._tag()}.log"
        )

    def is_done(self) -> bool:
        return self.output_path().exists()


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

def _set_pdeathsig():
    """Linux-only: send SIGTERM to subprocess if the parent dies."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        SIGTERM = 15
        libc.prctl(PR_SET_PDEATHSIG, SIGTERM, 0, 0, 0)
    except Exception:
        pass


def _env_with_gpu(gpu_id: int) -> dict:
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    e.pop("JAX_PLATFORMS", None)
    return e


def _build_cmd(unit: V2Unit, gpu_id: int, n_timesteps: int,
               n_seeds: int, deg_seed: int) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "run_degradation_v2.py"),
        "--env", unit.env,
        "--method", unit.method,
        "--gpu", str(gpu_id),
        "--degradation-type", unit.deg_type,
        "--intensity", str(unit.intensity),
        "--n-timesteps", str(n_timesteps),
        "--n-seeds", str(n_seeds),
        "--degradation-seed", str(deg_seed),
    ]


def _dispatch(unit: V2Unit, gpu_id: int, n_timesteps: int,
              n_seeds: int, deg_seed: int) -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(unit, gpu_id, n_timesteps, n_seeds, deg_seed)
    log_path = unit.log_path()
    with open(log_path, "w") as log:
        log.write(f"[study_v2] launching {unit.name()} on GPU {gpu_id}\n")
        log.write(f"[study_v2] cmd: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, env=_env_with_gpu(gpu_id),
            stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=_set_pdeathsig,
        )
        ret = proc.wait()
        log.write(f"[study_v2] exited with {ret}\n")
    return ret


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class _GPUWorker(threading.Thread):
    def __init__(self, gpu_id: int, work_q: queue.Queue, done_counter: dict,
                 lock: threading.Lock, stop_event: threading.Event,
                 n_timesteps: int, n_seeds: int, deg_seed: int):
        super().__init__(daemon=True, name=f"gpu-{gpu_id}")
        self.gpu_id = gpu_id
        self.work_q = work_q
        self.done_counter = done_counter
        self.lock = lock
        self.stop_event = stop_event
        self.n_timesteps = n_timesteps
        self.n_seeds = n_seeds
        self.deg_seed = deg_seed

    def run(self):
        while not self.stop_event.is_set():
            try:
                unit = self.work_q.get(timeout=2.0)
            except queue.Empty:
                if self.work_q.empty():
                    return
                continue
            try:
                if unit.is_done():
                    print(f"[gpu{self.gpu_id}] skip {unit.name()} (done)")
                    with self.lock:
                        self.done_counter["SKIP"] = (
                            self.done_counter.get("SKIP", 0) + 1
                        )
                    continue
                print(f"[gpu{self.gpu_id}] start {unit.name()}")
                t0 = time.time()
                ret = _dispatch(unit, self.gpu_id, self.n_timesteps,
                                self.n_seeds, self.deg_seed)
                dt = time.time() - t0
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[gpu{self.gpu_id}] {unit.name()} -> {status} "
                      f"({dt/60:.1f} min)")
                with self.lock:
                    self.done_counter[status] = (
                        self.done_counter.get(status, 0) + 1
                    )
            finally:
                self.work_q.task_done()


# ---------------------------------------------------------------------------
# Unit list
# ---------------------------------------------------------------------------

def _hpo_available(env: str, method: str) -> bool:
    """True if hp_results/<env>/<method>/optuna.db exists locally.

    Workers fail with exit 3 when this is missing (no Optuna study to load
    best params from). Pre-flighting here avoids wasted dispatch attempts.
    """
    return (ROOT / "hp_results" / env.lower() / method / "optuna.db").exists()


def _build_units(envs: List[str], methods: List[str],
                 types: Tuple[str, ...]) -> List[V2Unit]:
    units: List[V2Unit] = []
    for env in envs:
        for method in methods:
            if not _hpo_available(env, method):
                continue
            for deg_type in types:
                for intensity in INTENSITIES[deg_type]:
                    units.append(V2Unit(
                        env=env, method=method,
                        deg_type=deg_type, intensity=float(intensity),
                    ))
    return units


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", required=True,
                        help="GPU IDs to dispatch across (e.g. --gpus 0 1 2).")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--types", nargs="+", default=list(DEFAULT_TYPES),
                        choices=DEFAULT_TYPES)
    parser.add_argument("--n-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--degradation-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the work plan and exit (no dispatch).")
    args = parser.parse_args()

    units = _build_units(args.envs, args.methods, tuple(args.types))
    pending = [u for u in units if not u.is_done()]
    skipped_no_hpo = []
    for env in args.envs:
        for method in args.methods:
            if not _hpo_available(env, method):
                skipped_no_hpo.append(f"{env}/{method}")
    print(f"[study_v2] {len(units)} total cells (after HPO-availability filter), "
          f"{len(pending)} pending, "
          f"{len(units) - len(pending)} already done.")
    if skipped_no_hpo:
        print(f"[study_v2] skipped {len(skipped_no_hpo)} (env, method) pairs without HPO:")
        for s in skipped_no_hpo:
            print(f"  - {s}")
    print(f"[study_v2] envs={args.envs}")
    print(f"[study_v2] methods={args.methods}")
    print(f"[study_v2] types={args.types}")
    print(f"[study_v2] gpus={args.gpus}")

    if args.dry_run:
        print("--- WORK PLAN ---")
        for u in pending:
            print(f"  {u.name()} -> {u.output_path()}")
        return 0

    if not pending:
        print("[study_v2] nothing to do; exiting.")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    work_q: queue.Queue[V2Unit] = queue.Queue()
    for u in pending:
        work_q.put(u)

    done_counter: dict = {}
    lock = threading.Lock()
    stop_event = threading.Event()

    workers = [
        _GPUWorker(
            gpu_id=gid, work_q=work_q, done_counter=done_counter,
            lock=lock, stop_event=stop_event,
            n_timesteps=args.n_timesteps, n_seeds=args.n_seeds,
            deg_seed=args.degradation_seed,
        )
        for gid in args.gpus
    ]
    for w in workers:
        w.start()

    try:
        work_q.join()
    except KeyboardInterrupt:
        print("[study_v2] Ctrl-C: stopping workers and waiting for in-flight cells")
        stop_event.set()
        for w in workers:
            w.join(timeout=300.0)
        return 130

    stop_event.set()
    for w in workers:
        w.join(timeout=10.0)

    print("[study_v2] done.")
    print(f"[study_v2] summary: {done_counter}")
    return 0 if done_counter.get("OK", 0) + done_counter.get("SKIP", 0) >= len(pending) else 1


if __name__ == "__main__":
    sys.exit(main())
