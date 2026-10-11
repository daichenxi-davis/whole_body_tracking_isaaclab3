"""Diff the resolved env/agent config against the upstream BeyondMimic values.

Purpose: prove that the Isaac Lab 3.0 migration introduced no hidden parameter drift beyond the
changes that were made deliberately. The reference column below is the upstream whole_body_tracking
configuration at the values the repository shipped (Isaac Lab 2.1.0 era). Any difference reported is
classified as either:

* EXPECTED  -- one of the deliberate migration changes (documented in the repo's migration notes)
* UNEXPECTED -- parameter drift that was not intended; should be investigated

Requires the patched ``whole_body_tracking`` package to be importable (installed editable, or its
source directory pointed at with the ``WBT_SOURCE`` environment variable).

Usage (from the Isaac Lab checkout)::

    .venv\\Scripts\\python.exe <this file>
"""

from __future__ import annotations

import os
import sys

# Allow running against an uninstalled source tree:
#   set WBT_SOURCE=<...>\whole_body_tracking\source\whole_body_tracking
_source = os.environ.get("WBT_SOURCE")
if _source and _source not in sys.path:
    sys.path.insert(0, _source)

import whole_body_tracking.tasks  # noqa: F401, E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "Tracking-Flat-G1-v0"

# Upstream values from the un-migrated repository. Only fields that a migration could plausibly
# disturb are listed; everything else is compared against the resolved config itself.
UPSTREAM = {
    "scene.num_envs": 4096,
    "scene.env_spacing": 2.5,
    "scene.contact_forces.history_length": 3,
    "scene.contact_forces.track_air_time": True,
    "scene.contact_forces.force_threshold": 10.0,
    "scene.contact_forces.debug_vis": True,
    "decimation": 4,
    "episode_length_s": 10.0,
    "sim.dt": 0.005,
    "commands.motion.resampling_time_range": (1.0e9, 1.0e9),
    "commands.motion.debug_vis": True,
    "commands.motion.anchor_body_name": "torso_link",
    "commands.motion.joint_position_range": (-0.1, 0.1),
    "commands.motion.adaptive_kernel_size": 1,
    "commands.motion.adaptive_lambda": 0.8,
    "commands.motion.adaptive_uniform_ratio": 0.1,
    "commands.motion.adaptive_alpha": 0.001,
    "observations.policy.enable_corruption": True,
    "observations.policy.concatenate_terms": True,
    "actions.joint_pos.use_default_offset": True,
    "rewards.motion_global_anchor_pos.weight": 0.5,
    "rewards.motion_global_anchor_ori.weight": 0.5,
    "rewards.motion_body_pos.weight": 1.0,
    "rewards.motion_body_ori.weight": 1.0,
    "rewards.motion_body_lin_vel.weight": 1.0,
    "rewards.motion_body_ang_vel.weight": 1.0,
    "rewards.action_rate_l2.weight": -1e-1,
    "rewards.joint_limit.weight": -10.0,
    "rewards.undesired_contacts.weight": -0.1,
    "terminations.anchor_pos.params.threshold": 0.25,
    "terminations.anchor_ori.params.threshold": 0.8,
    "terminations.ee_body_pos.params.threshold": 0.25,
}

# Deliberate migration changes. Everything here is expected to differ.
EXPECTED_CHANGES = {
    "scene.num_envs": "reduced 4096 -> 64 for the local 4 GB GPU",
    "scene.contact_forces.history_length": "reduced 3 -> 2 to save VRAM",
    "scene.contact_forces.debug_vis": "disabled to save VRAM",
    "commands.motion.debug_vis": "disabled to save VRAM",
    "sim.physics": "new explicit backend config (self.sim.physx no longer exists)",
}

AGENT_UPSTREAM = {
    "num_steps_per_env": 24,
    "max_iterations": 30000,
    "save_interval": 500,
    "experiment_name": "g1_flat",
    "algorithm.value_loss_coef": 1.0,
    "algorithm.use_clipped_value_loss": True,
    "algorithm.clip_param": 0.2,
    "algorithm.entropy_coef": 0.005,
    "algorithm.num_learning_epochs": 5,
    "algorithm.num_mini_batches": 4,
    "algorithm.learning_rate": 1.0e-3,
    "algorithm.schedule": "adaptive",
    "algorithm.gamma": 0.99,
    "algorithm.lam": 0.95,
    "algorithm.desired_kl": 0.01,
    "algorithm.max_grad_norm": 1.0,
}

AGENT_EXPECTED_CHANGES = {
    "policy": "replaced by actor/critic model cfgs (rsl-rl >= 4.0 deprecates 'policy')",
    "empirical_normalization": "folded into actor.obs_normalization / critic.obs_normalization",
    "obs_groups": "now explicit instead of MISSING (auto-filled to the same value)",
}


def dig(obj, dotted: str):
    """Resolve a dotted path on a nested config object.

    Supports both attribute access and dict keys, because manager term parameters live inside the
    term's ``params`` dict (e.g. ``terminations.anchor_pos.params.threshold``) rather than on the
    term cfg object itself.
    """
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
    return cur


def compare(label, cfg, reference):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    drift, expected, same = [], [], 0
    for path, want in sorted(reference.items()):
        got = dig(cfg, path)
        if got == want:
            same += 1
            continue
        if path in EXPECTED_CHANGES:
            expected.append((path, want, got))
        else:
            drift.append((path, want, got))
    print(f"fields compared: {len(reference)}   identical: {same}")
    if expected:
        print(f"\nEXPECTED changes ({len(expected)}):")
        for p, w, g in expected:
            print(f"  {p}\n      upstream: {w!r}\n      now     : {g!r}\n      reason  : {EXPECTED_CHANGES[p]}")
    if drift:
        print(f"\nUNEXPECTED drift ({len(drift)}) -- investigate:")
        for p, w, g in drift:
            print(f"  {p}: upstream={w!r} now={g!r}")
    else:
        print("\nUNEXPECTED drift: none")
    return len(drift)


def main() -> int:
    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")

    n = compare("ENV CONFIG", env_cfg, UPSTREAM)
    n += compare("AGENT CONFIG", agent_cfg, AGENT_UPSTREAM)

    # Report the new fields the migration introduced, so nothing is silently unaccounted for.
    print(f"\n{'=' * 78}\nSTRUCTURAL ADDITIONS (not present upstream)\n{'=' * 78}")
    print(f"  sim.physics      = {type(env_cfg.sim.physics).__name__}  (upstream: sim.physx.*)")
    print(f"  sim.default_visualizer_cfg = {type(env_cfg.sim.default_visualizer_cfg).__name__}")
    print(f"  agent.actor/actor.critic   = {type(agent_cfg.actor).__name__} / {type(agent_cfg.critic).__name__}")
    print(f"  agent.obs_groups           = {agent_cfg.obs_groups}")
    print(f"  agent.actor.hidden_dims    = {agent_cfg.actor.hidden_dims}  obs_normalization={agent_cfg.actor.obs_normalization}")
    print(f"  agent.critic.hidden_dims   = {agent_cfg.critic.hidden_dims}  obs_normalization={agent_cfg.critic.obs_normalization}")

    print(f"\n{'=' * 78}")
    print(f"RESULT: {'PASS - no unintended drift' if n == 0 else f'FAIL - {n} unexpected difference(s)'}")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
