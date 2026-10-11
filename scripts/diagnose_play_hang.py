"""Pinpoint which call in the `isaaclab play` sequence blocks.

Run from the Isaac Lab checkout:

    python scripts/diagnose_play_hang.py --task Tracking-Flat-G1-v0 --checkpoint <path>

Prints a timestamped line around every step the play entrypoint performs, so the first missing
timestamp identifies the blocking call. Uses `faulthandler` to dump the Python stacks of all threads
if a step exceeds `--stuck_after` seconds.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time

# Unbuffered and crash-safe: a hang here must still leave the markers on disk.
sys.stdout.reconfigure(line_buffering=True, write_through=True)
faulthandler.enable()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--stuck_after", type=float, default=45.0, help="Dump all thread stacks after this many seconds in a step.")
parser.add_argument("--motion", default=os.environ.get("WBT_MOTION", ""))
parser.add_argument("--agent", default=None)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

_T0 = time.time()


def mark(label: str) -> float:
    """Print a timestamped marker and return the current elapsed time."""
    now = time.time() - _T0
    print(f"[{now:8.2f}s] {label}", flush=True)
    return now


def step(label: str, fn):
    """Run fn, arming faulthandler so a hang dumps every thread stack."""
    start = mark(f"BEGIN {label}")
    # Write the dump to its own file: it would otherwise be lost in buffered redirected stdout.
    dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "play_hang_traceback.txt")
    handle = open(dump_path, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(args_cli.stuck_after, exit=False, file=handle)
    try:
        result = fn()
    except BaseException as exc:  # never swallow: a raised error is a result too
        import traceback

        mark(f"RAISED {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        handle.close()
    mark(f"END   {label}  ({time.time() - _T0 - start:.2f}s)")
    return result


mark("launching app")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
mark("app launched")

if args_cli.motion:
    hydra_overrides = list(hydra_overrides) + [f"env.commands.motion.motion_file={args_cli.motion}"]
mark(f"hydra overrides: {hydra_overrides}")

import whole_body_tracking.tasks  # noqa: F401, E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import (  # noqa: E402
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

_orig_argv = sys.argv
sys.argv = [sys.argv[0]] + list(hydra_overrides)
try:
    env_cfg, agent_cfg = step(
        "resolve_task_config", lambda: resolve_task_config(args_cli.task, args_cli.agent or "rsl_rl_cfg_entry_point")
    )
finally:
    sys.argv = _orig_argv
mark(f"env_cfg.agent = {args_cli.agent or 'rsl_rl_cfg_entry_point'}")

# REQUIRED before create_rsl_rl_runner: the real play entrypoint does this at play_rsl_rl.py:113.
# Without it, agent_cfg.actor still carries the legacy stochastic/init_noise_std keys and
# rsl_rl's construct_algorithm fails with TypeError. Omitting it here was a bug in this diagnostic,
# not in the play path.
agent_cfg = step("handle_deprecated_rsl_rl_cfg", lambda: handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version()))
mark(f"actor keys after normalisation: {sorted(agent_cfg.to_dict()['actor'])}")

env_cfg.scene.num_envs = args_cli.num_envs
mark(f"num_envs set to {args_cli.num_envs}")
mark(f"device = {env_cfg.sim.device}")

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

env = step("ManagerBasedRLEnv(...)", lambda: ManagerBasedRLEnv(cfg=env_cfg))
mark("env created")

wrapped = step("RslRlVecEnvWrapper", lambda: RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions))
mark("env wrapped")

runner = step("create_rsl_rl_runner", lambda: create_rsl_rl_runner(wrapped, agent_cfg))
mark("runner created")

step("runner.load", lambda: runner.load(args_cli.checkpoint))
mark("checkpoint loaded")

policy = step("get_inference_policy", lambda: runner.get_inference_policy(device=env.unwrapped.device))
mark("inference policy ready")

step("export_policy_to_jit", lambda: runner.export_policy_to_jit(path=os.path.join("logs", "play_diag"), filename="policy.pt"))
mark("jit exported")

step("export_policy_to_onnx", lambda: runner.export_policy_to_onnx(path=os.path.join("logs", "play_diag"), filename="policy.onnx"))
mark("onnx exported")

obs = step("env.get_observations", lambda: wrapped.get_observations())
mark("observations ready")

step("one env.step", lambda: wrapped.step(policy(obs)))
mark("one step done")

mark("closing env")
env.close()
mark("env closed")
mark("DONE - play sequence completes")
simulation_app.close()
